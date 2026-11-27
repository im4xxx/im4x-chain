--[[
  im4x_chain_core.lua — shared engine for the two REAPER entry scripts:

    im4x_chain_overwrite.lua   source file is replaced in place
    im4x_chain_copy.lua        processed copy is added as a new take

  The audio itself is done by the `im4x-chain` command line tool (the chain from
  this repository). The scripts only:
    * take the active take of every selected item,
    * start `im4x-chain process ...` in the background,
    * read its JSON-lines progress file and draw a progress window,
    * reload the media / add a take when the file is ready.

  Install: copy the three .lua files into REAPER's Scripts folder and load the
  two entry scripts via Actions > Show action list > New action > Load ReaScript.
  The CLI is looked up in PATH, ~/.local/bin, the uv tool directory, and in a
  local checkout next to this file. Override with IM4X_CHAIN_CMD if needed.
--]]

local MODE = _G.IM4X_MODE
if MODE ~= "overwrite" and MODE ~= "copy" then
  reaper.ShowMessageBox(
    "Этот файл — общий движок. Запустите im4x_chain_overwrite.lua\n" ..
    "или im4x_chain_copy.lua (Actions > Load ReaScript).",
    "im4x-chain", 0)
  return
end

local IS_WIN = reaper.GetOS():match("Win") ~= nil
local IS_MAC = not IS_WIN and (reaper.GetOS():match("OSX") ~= nil or reaper.GetOS():match("mac") ~= nil)
local HOME = os.getenv(IS_WIN and "USERPROFILE" or "HOME") or ""

--==============================================================================
-- settings
--==============================================================================

local CONFIG = {
  -- Empty = auto-detect. May be a command prefix, e.g.
  --   "/Users/me/.local/bin/im4x-chain"
  --   "PYTHONPATH=/repo/src /repo/.venv/bin/python -m im4x_chain.cli"
  cli = "/Users/max/vibe-coding/im4x-chain/src/im4x_chain/cli.py",

  -- Extra arguments for `im4x-chain process`, e.g. "--target -16 --no-deplosive".
  -- Do not put -o here: the copy version adds its own output directory.
  extra_args = "",

  -- Copy version: suffix and destination ("" = next to the source file).
  copy_suffix = "_im4x",
  copy_dir = "",

  -- Ask before replacing source files.
  confirm_overwrite = true,

  -- Keep logs of successful runs too (logs of failed runs are always kept).
  keep_logs = false,

  window_w = 680,
  window_h = 500,
}

local env_cli = os.getenv("IM4X_CHAIN_CMD")
if env_cli and env_cli ~= "" then CONFIG.cli = env_cli end
local env_args = os.getenv("IM4X_CHAIN_ARGS")
if env_args and env_args ~= "" then CONFIG.extra_args = env_args end
if os.getenv("IM4X_CHAIN_NO_CONFIRM") == "1" then CONFIG.confirm_overwrite = false end
local env_copy_dir = os.getenv("IM4X_CHAIN_COPY_DIR")
if env_copy_dir and env_copy_dir ~= "" then CONFIG.copy_dir = env_copy_dir end

--==============================================================================
-- small helpers
--==============================================================================

local function trim(s)
  return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function basename(p)
  if IS_WIN then p = p:gsub("\\", "/") end
  return p:match("([^/]+)$") or p
end

local function dirname(p)
  if IS_WIN then p = p:gsub("\\", "/") end
  return p:match("^(.*)/[^/]*$") or "."
end

local function join(a, b)
  if a == nil or a == "" or a == "." then return b end
  if a:sub(-1) == "/" or a:sub(-1) == "\\" then return a .. b end
  return a .. "/" .. b
end

local function expand_home(p)
  if p == "~" then return HOME end
  if p:sub(1, 2) == "~/" or p:sub(1, 2) == "~\\" then return join(HOME, p:sub(3)) end
  return p
end

local function file_exists(p)
  local f = io.open(p, "rb")
  if f then f:close(); return true end
  return false
end

local function read_file(p)
  local f = io.open(p, "rb")
  if not f then return nil end
  local data = f:read("*a")
  f:close()
  return data
end

local function write_file(p, data)
  local f, err = io.open(p, "wb")
  if not f then return false, err end
  f:write(data)
  f:close()
  return true
end

local function ensure_dir(p)
  if p == nil or p == "" or p == "." then return end
  reaper.RecursiveCreateDirectory(p, 0)
end

local shq

local function remove_dir(p)
  -- only our own temporary directories
  if not p or not (p:find("im4x_chain_runs", 1, true) or p:find(".im4x-tmp-", 1, true)) then
    return
  end
  if IS_WIN then
    os.execute('rmdir /s /q ' .. shq(p) .. ' >nul 2>&1')
  else
    os.execute("rm -rf " .. shq(p) .. " >/dev/null 2>&1")
  end
end

-- Shell-quote one argument. Windows commands in this script are written to a
-- .cmd file, so double quotes and % are enough for ordinary paths.
shq = function(s)
  s = tostring(s)
  if IS_WIN then
    s = s:gsub("%%", "%%%%")
    return '"' .. s:gsub('"', '""') .. '"'
  end
  return "'" .. s:gsub("'", "'\\''") .. "'"
end

local function open_path(p)
  if IS_WIN then
    os.execute('start "" ' .. shq(p))
  elseif IS_MAC then
    os.execute("open " .. shq(p))
  else
    os.execute("xdg-open " .. shq(p) .. " >/dev/null 2>&1")
  end
end

local function process_alive(pid)
  if not pid or pid <= 0 then return false end
  if IS_WIN then
    local h = io.popen('tasklist /FI "PID eq ' .. pid .. '" /NH 2>nul')
    if not h then return false end
    local out = h:read("*a") or ""
    h:close()
    return out:find(tostring(pid), 1, true) ~= nil
  end
  local ok = os.execute("kill -0 " .. pid .. " >/dev/null 2>&1")
  return ok == true or ok == 0
end

local function kill_pid(pid, force)
  if not pid or pid <= 0 then return end
  if IS_WIN then
    os.execute("taskkill /PID " .. pid .. " /T " .. (force and "/F " or "") .. ">nul 2>&1")
  else
    os.execute("kill -" .. (force and "KILL" or "TERM") .. " " .. pid .. " >/dev/null 2>&1")
  end
end

local function child_pids(pid)
  local out = {}
  if IS_WIN then return out end
  local h = io.popen("pgrep -P " .. pid .. " 2>/dev/null")
  if h then
    for line in h:lines() do
      local n = tonumber(line:match("%d+"))
      if n then out[#out + 1] = n end
    end
    h:close()
  end
  return out
end

-- Kill a process and its descendants (denoise workers, `sleep` in a wrapper,
-- etc.). REAPER's Lua has no process groups, so walk the tree before killing.
local function kill_tree(pid, force, depth)
  if not pid or pid <= 0 then return end
  depth = depth or 0
  if not IS_WIN and depth < 8 then
    for _, kid in ipairs(child_pids(pid)) do
      kill_tree(kid, force, depth + 1)
    end
  end
  kill_pid(pid, force)
end

local function fmt_time(seconds)
  seconds = math.max(0, math.floor(seconds or 0))
  return string.format("%02d:%02d", math.floor(seconds / 60), seconds % 60)
end

--==============================================================================
-- CLI lookup and progress support
--==============================================================================

local function detect_cli()
  if CONFIG.cli ~= "" then return CONFIG.cli, CONFIG.cli end

  -- A checkout next to this file: reaper/im4x_chain_core.lua -> repo root.
  local self_path = debug.getinfo(1, "S").source:sub(2)
  local self_dir = self_path:match("^(.*[/\\])") or ""
  local repo = self_dir:gsub("[/\\]reaper[/\\]?$", "")
  if repo ~= self_dir and not IS_WIN then
    local py = join(repo, ".venv/bin/python")
    if file_exists(py) and file_exists(join(repo, "src/im4x_chain/cli.py")) then
      local cmd = "PYTHONPATH=" .. shq(join(repo, "src")) .. " " .. shq(py) .. " -m im4x_chain.cli"
      return cmd, cmd
    end
  end

  local exe = IS_WIN and "im4x-chain.exe" or "im4x-chain"
  local candidates = {
    join(HOME, ".local/bin/" .. exe),
    join(HOME, ".local/share/uv/tools/im4x-chain/bin/" .. exe),
    join(HOME, "bin/" .. exe),
    "/opt/homebrew/bin/im4x-chain",
    "/usr/local/bin/im4x-chain",
    "/usr/bin/im4x-chain",
  }
  for _, p in ipairs(candidates) do
    if p ~= "" and file_exists(p) then return shq(p), p end
  end

  local h = io.popen(IS_WIN and "where im4x-chain 2>nul" or "command -v im4x-chain 2>/dev/null")
  if h then
    local line = h:read("*l")
    h:close()
    if line and line ~= "" and file_exists(line) then return shq(line), line end
  end
  return nil, nil
end

local function supports_progress(cli)
  local h = io.popen(cli .. " process --help 2>&1")
  if not h then return false end
  local out = h:read("*a") or ""
  h:close()
  return out:find("--progress-file", 1, true) ~= nil
end

--==============================================================================
-- selected items -> jobs
--==============================================================================

local TAKE_PROPS = {"D_STARTOFFS", "D_PLAYRATE", "D_PITCH", "B_PPITCH", "B_LOOPSRC",
                    "I_CHANMODE", "D_VOL", "D_PAN", "B_MUTE"}

local function save_take(take)
  local p = {}
  for _, k in ipairs(TAKE_PROPS) do p[k] = reaper.GetMediaItemTakeInfo_Value(take, k) end
  local _, name = reaper.GetSetMediaItemTakeInfo_String(take, "P_NAME", "", false)
  p.name = name or ""
  return p
end

local function restore_take(take, p)
  for k, v in pairs(p) do
    if k ~= "name" and k ~= "source_path" and v ~= nil then
      reaper.SetMediaItemTakeInfo_Value(take, k, v)
    end
  end
  if p.name and p.name ~= "" then
    reaper.GetSetMediaItemTakeInfo_String(take, "P_NAME", p.name, true)
  end
end

local function resolve_source_path(p)
  if not p or p == "" then return nil end
  if IS_WIN then p = p:gsub("\\", "/") end
  if p:sub(1, 1) ~= "/" and not p:match("^%a:/") then
    local proj = reaper.GetProjectPath("")
    local cand = join(proj, p)
    if proj ~= "" and file_exists(cand) then p = cand end
  end
  return p
end

local function source_is_plain(src)
  if reaper.GetMediaSourceParent then
    local ok, parent = pcall(reaper.GetMediaSourceParent, src)
    if ok and parent then return false end
    return true
  end
  if not reaper.PCM_Source_GetSectionInfo then return true end
  local called, ok, offset, _length, reverse = pcall(reaper.PCM_Source_GetSectionInfo, src)
  if not called or not ok then return true end
  if reverse then return false end
  if offset and offset > 0.000001 then return false end
  return true
end

local function same_source(take, want)
  local src = reaper.GetMediaItemTake_Source(take)
  if not src then return false end
  if not source_is_plain(src) then return false end
  if (reaper.GetMediaSourceType(src, "") or ""):upper() == "SECTION" then return false end
  local p = resolve_source_path(reaper.GetMediaSourceFileName(src, "") or "") or ""
  want = resolve_source_path(want) or want or ""
  if IS_WIN then
    p, want = p:lower():gsub("\\", "/"), want:lower():gsub("\\", "/")
  end
  return p == want
end

local function collect_files()
  local files, order, skipped = {}, {}, {}
  local total = reaper.CountSelectedMediaItems(0)

  for i = 0, total - 1 do
    local item = reaper.GetSelectedMediaItem(0, i)
    local take = item and reaper.GetActiveTake(item)
    local reason = nil
    local path = nil

    if not item then
      reason = "айтем не найден"
    elseif not take then
      reason = "нет активного тейка (MIDI или пустой айтем)"
    else
      local src = reaper.GetMediaItemTake_Source(take)
      if not src then
        reason = "у тейка нет источника"
      else
        local typ = reaper.GetMediaSourceType(src, "")
        if not source_is_plain(src) or (typ or ""):upper() == "SECTION" then
          reason = "section/reverse источник не поддерживается (сначала Glue/склейте айтем)"
        else
          path = resolve_source_path(reaper.GetMediaSourceFileName(src, ""))
          local ext = path and path:lower():match("%.([^./\\]+)$") or ""
          if not path or path == "" then
            reason = "пустое имя файла"
          elseif ext ~= "wav" and ext ~= "flac" then
            reason = "формат ." .. (ext ~= "" and ext or "?") .. " не поддерживается (нужен WAV или FLAC)"
          elseif not file_exists(path) then
            reason = "файл не найден: " .. path
          end
        end
      end
    end

    if reason then
      skipped[#skipped + 1] = string.format("айтем %d: %s", i + 1, reason)
    else
      local key = IS_WIN and path:lower() or path
      local f = files[key]
      if not f then
        f = {path = path, takes = {}}
        files[key] = f
        order[#order + 1] = f
      end
      f.takes[#f.takes + 1] = {item = item, take = take}
    end
  end

  return order, skipped, total
end

local function target_for(path)
  local dir = CONFIG.copy_dir
  if dir == "" then
    dir = dirname(path)
  else
    dir = expand_home(dir)
    if dir:sub(1, 1) ~= "/" and not dir:match("^%a:") then
      dir = join(reaper.GetProjectPath(""), dir)
    end
  end
  local base = basename(path)
  local stem, ext = base:match("^(.*)(%.[^%.]+)$")
  if not stem or stem == "" then stem, ext = base, "" end
  local cand = join(dir, stem .. CONFIG.copy_suffix .. ext)
  local n = 2
  while file_exists(cand) do
    cand = join(dir, stem .. CONFIG.copy_suffix .. "-" .. n .. ext)
    n = n + 1
  end
  return cand
end

--==============================================================================
-- lock: only one chain run at a time in this REAPER profile
--==============================================================================

local LOCK = join(reaper.GetResourcePath(), "im4x_chain.lock")
local lock_owned = false

local function acquire_lock()
  local data = read_file(LOCK)
  if data and data ~= "" then
    local pid = tonumber(data:match("(%-?%d+)"))
    local ts = tonumber(data:match("\n(%d+)"))
    local fresh = ts ~= nil and (os.time() - ts) < 120
    if (pid and pid > 0 and process_alive(pid)) or (pid == -1 and fresh) then
      return false, pid
    end
    os.remove(LOCK)
  end
  local ok = write_file(LOCK, "-1\n" .. tostring(os.time()))
  lock_owned = ok
  return ok, nil
end

local function lock_update(pid)
  if not lock_owned or not pid then return end
  write_file(LOCK, tostring(pid) .. "\n" .. tostring(os.time()))
end

local function release_lock()
  if lock_owned then
    os.remove(LOCK)
    lock_owned = false
  end
end

--==============================================================================
-- run state
--==============================================================================

local cli_cmd = detect_cli()
if not cli_cmd then
  reaper.ShowMessageBox(
    "Не найден im4x-chain.\n\n" ..
    "Установите цепочку:\n" ..
    "  uv tool install git+https://github.com/im4xxx/im4x-chain\n" ..
    "и выполните  uv tool update-shell  (потом новый терминал / перезапуск REAPER).\n\n" ..
    "Или укажите путь к CLI в CONFIG.cli в начале im4x_chain_core.lua\n" ..
    "либо в переменной окружения IM4X_CHAIN_CMD.",
    "im4x-chain", 0)
  return
end

local has_progress = supports_progress(cli_cmd)
local files, skipped, selected_count = collect_files()

if selected_count == 0 then
  reaper.ShowMessageBox("Выберите хотя бы один айтем с аудио (WAV или FLAC).",
    "im4x-chain", 0)
  return
end

if #skipped > 0 then
  local msg = "Пропущено айтемов: " .. #skipped
  local n = math.min(#skipped, 8)
  for i = 1, n do msg = msg .. "\n  • " .. skipped[i] end
  if #skipped > n then msg = msg .. "\n  …" end
  reaper.ShowMessageBox(msg, "im4x-chain", 0)
end

if #files == 0 then
  reaper.ShowMessageBox("Среди выбранных айтемов нет файлов WAV/FLAC с активным тейком.",
    "im4x-chain", 0)
  return
end

if MODE == "overwrite" and CONFIG.confirm_overwrite then
  local names = {}
  for i = 1, math.min(#files, 6) do names[#names + 1] = "  • " .. files[i].path end
  if #files > 6 then names[#names + 1] = "  …" end
  local msg = "Исходные файлы будут перезаписаны (" .. #files .. "):\n" ..
              table.concat(names, "\n") ..
              "\n\nREAPER Undo не вернёт изменённые файлы на диске.\nПродолжить?"
  if reaper.ShowMessageBox(msg, "im4x-chain: перезапись", 4) ~= 6 then return end
end

local locked, lock_pid = acquire_lock()
if not locked then
  reaper.ShowMessageBox(
    "Уже выполняется другой процесс im4x-chain" ..
    (lock_pid and lock_pid > 0 and (" (PID " .. lock_pid .. ")") or "") ..
    ".\nДождитесь его завершения и запустите снова.",
    "im4x-chain", 0)
  return
end

math.randomseed(os.time() + math.floor((reaper.time_precise() * 1000) % 100000))
local run_dir = join(reaper.GetResourcePath(), "im4x_chain_runs/" ..
  string.format("%s_%d_%d", MODE, os.time(), math.random(100000, 999999)))
ensure_dir(run_dir)

local jobs = {}
for i, f in ipairs(files) do
  local job = {
    file = f,
    src = f.path,
    index = i,
    target = MODE == "copy" and target_for(f.path) or f.path,
    progress = join(run_dir, string.format("p%d.jsonl", i)),
    log = join(run_dir, string.format("p%d.log", i)),
    rc = join(run_dir, string.format("p%d.rc", i)),
    script = join(run_dir, string.format("p%d.%s", i, IS_WIN and "cmd" or "sh")),
    wrapper = join(run_dir, string.format("p%d.wrapper", i)),
    seen = 0,
    status = "pending",
  }
  if MODE == "copy" then
    -- The temporary output must be on the same volume as the target, otherwise
    -- os.rename() cannot move it (an external media drive, for example).
    job.tmpdir = join(dirname(job.target),
      string.format(".im4x-tmp-%d-%d", i, math.random(100000, 999999)))
  else
    job.tmpdir = join(run_dir, string.format("out%d", i))
  end
  jobs[#jobs + 1] = job
end

local ui = {
  stages = {},
  status = {},
  stage_seconds = {},
  file = "",
  target = "",
  message = "Подготовка…",
  summary = "",
  error = "",
  metrics = nil,
  current = nil,
  current_since = 0,
  done_count = 0,
  total_stages = 0,
  unknown = not has_progress,
  had_error = false,
}

local STAGE_LABELS = {
  read = "Чтение и замер",
  voice = "Высота голоса / срез низа",
  eq = "EQ",
  deplosive = "De-plosive (RX)",
  declick = "Mouth De-click (RX)",
  denoise = "Spectral De-noise (RX)",
  leveler = "Leveler",
  compressor = "Компрессор X-Comp",
  loudness = "Loudness Control",
  write = "Запись файла",
  measure = "Замер результата",
}

local phase = "starting"          -- starting | running | gap | done
local current = 0
local run_started_at = reaper.time_precise()
local start_at = run_started_at + 0.15
local next_job_at = run_started_at
local outcome = "ok"
local cancel_requested = false
local close_when_done = false
local window_closed = false
local quit_requested = false
local quit_done = false
local need_peaks = false
local keep_run_dir = false
local run_dir_removed = false
local undo_open = false
local noted_items = {}
local cancel_kill_at = 0
local cancel_force_at = 0
local cancel_giveup_at = 0
local mouse_was_down = false
local button_hit = {}
local trunc_cache = {}

--==============================================================================
-- undo
--==============================================================================

local function ensure_undo()
  if not undo_open then
    reaper.Undo_BeginBlock2(0)
    undo_open = true
  end
end

local function build_peaks(src)
  if reaper.PCM_Source_BuildPeaks then
    reaper.PCM_Source_BuildPeaks(src, 0)
  else
    need_peaks = true
  end
end

local function cleanup_partial(job)
  if not job then return end
  if MODE == "overwrite" then
    local base = basename(job.src)
    local stem, ext = base:match("^(.*)(%.[^%.]+)$")
    if stem then
      local tmp = join(dirname(job.src), "." .. stem .. ".im4xchain.tmp" .. ext)
      if file_exists(tmp) then os.remove(tmp) end
    end
  else
    remove_dir(job.tmpdir)
  end
end

local function close_undo()
  if undo_open then
    local desc = MODE == "copy" and "im4x-chain: обработка в новый тейк"
                            or "im4x-chain: обработка с перезаписью файла"
    reaper.Undo_EndBlock2(0, desc, -1)
    undo_open = false
  end
end

--==============================================================================
-- progress file (JSON lines)
--==============================================================================

local function field(line, key)
  return line:match('"' .. key .. '"%s*:%s*"([^"]*)"')
end

local function num(line, key)
  return tonumber(line:match('"' .. key .. '"%s*:%s*(-?%d+%.?%d*)'))
end

local function read_progress(job)
  local data = read_file(job.progress)
  if not data then return end
  local lines = {}
  for line in data:gmatch("([^\n]*)\n") do lines[#lines + 1] = line end
  for i = (job.seen or 0) + 1, #lines do
    local line = lines[i]
    if line ~= "" then
      local event = field(line, "event")
      if event == "begin" then
        local stages = field(line, "stages") or ""
        ui.stages = {}
        for s in stages:gmatch("[^,]+") do ui.stages[#ui.stages + 1] = trim(s) end
        ui.status = {}
        for _, s in ipairs(ui.stages) do ui.status[s] = "pending" end
        ui.total_stages = #ui.stages
        ui.done_count = 0
        ui.current = nil
        ui.current_since = reaper.time_precise()
        ui.unknown = (#ui.stages == 0)
        local pid = num(line, "pid")
        if pid and pid > 0 then job.pid = pid end
        if not cancel_requested then ui.message = "Обработка…" end
      elseif event == "stage" then
        local s, st = field(line, "stage"), field(line, "status")
        if s then
          if st == "start" then
            if ui.status[s] ~= "current" then
              ui.status[s] = "current"
              ui.current = s
              ui.current_since = reaper.time_precise()
              if not cancel_requested then
                ui.message = "Выполняется: " .. (STAGE_LABELS and STAGE_LABELS[s] or s)
              end
            end
          elseif st == "done" then
            if ui.status[s] ~= "done" then
              ui.status[s] = "done"
              ui.done_count = ui.done_count + 1
            end
            if ui.current == s then ui.current = nil end
            ui.stage_seconds[s] = num(line, "seconds")
          end
        end
      elseif event == "done" then
        job.metrics = {
          before_lufs = num(line, "before_lufs"),
          before_tp = num(line, "before_tp"),
          after_lufs = num(line, "after_lufs"),
          after_tp = num(line, "after_tp"),
          seconds = num(line, "seconds"),
        }
        job.warning = field(line, "warning") or job.warning
        ui.metrics = job.metrics
      elseif event == "error" then
        job.progress_error = field(line, "message") or ""
      end
    end
  end
  job.seen = #lines
end

local function read_rc(job)
  local data = read_file(job.rc)
  if not data then return nil end
  return tonumber(data:match("(%-?%d+)"))
end

local function last_lines(job, maxlen)
  local data = read_file(job.log)
  if not data then return "" end
  if #data > maxlen then data = data:sub(-maxlen) end
  local last = ""
  for line in data:gmatch("[^\r\n]+") do
    if line:match("%S") then last = line end
  end
  return last
end

local function metrics_from_log(job)
  local data = read_file(job.log) or ""
  local a, b, c, d, t = data:match(
    "(%-?%d+%.?%d*)%s+LUFS%s+TP%s+(%-?%d+%.?%d*)%s+dBTP%s*%-%>%s*" ..
    "(%-?%d+%.?%d*)%s+LUFS%s+TP%s+(%-?%d+%.?%d*)%s+dBTP%s*%[([%d%.]+)%s*s%]")
  if a then
    return {before_lufs = tonumber(a), before_tp = tonumber(b),
            after_lufs = tonumber(c), after_tp = tonumber(d), seconds = tonumber(t)}
  end
end

--==============================================================================
-- applying the result to the project
--==============================================================================

local function note_original(item, src)
  if noted_items[item] then return end
  noted_items[item] = true
  local _, notes = reaper.GetSetMediaItemInfo_String(item, "P_NOTES", "", false)
  notes = notes or ""
  local line = "im4x-chain: original file " .. src
  if not notes:find(line, 1, true) then
    local sep = (notes == "" or notes:sub(-1) == "\n") and "" or "\n"
    reaper.GetSetMediaItemInfo_String(item, "P_NOTES", notes .. sep .. line, true)
  end
end

local function add_take(item, props, src, target)
  local newsrc = reaper.PCM_Source_CreateFromFile(target)
  if not newsrc then return false, "не удалось открыть " .. target end
  local newtake = reaper.AddTakeToMediaItem(item)
  if not newtake then
    if reaper.PCM_Source_Destroy then reaper.PCM_Source_Destroy(newsrc) end
    return false, "не удалось добавить тейк"
  end
  reaper.SetMediaItemTake_Source(newtake, newsrc)
  build_peaks(newsrc)
  restore_take(newtake, props)
  local base = props.name ~= "" and props.name or basename(src)
  reaper.GetSetMediaItemTakeInfo_String(newtake, "P_NAME", base .. " [im4x]", true)
  reaper.SetActiveTake(newtake)
  reaper.UpdateItemInProject(item)
  note_original(item, src)
  return true
end

local function apply_copy(job)
  local produced = join(job.tmpdir, basename(job.src))
  if not file_exists(produced) then
    error("CLI не создал файл: " .. produced)
  end
  ensure_dir(dirname(job.target))
  local ok, err = os.rename(produced, job.target)
  if not ok then
    error("не удалось переименовать результат: " .. tostring(err))
  end
  remove_dir(job.tmpdir)

  ensure_undo()
  local added_count = 0
  for _, t in ipairs(job.file.takes) do
    if reaper.ValidatePtr(t.item, "MediaItem*") and reaper.ValidatePtr(t.take, "MediaItem_Take*")
       and same_source(t.take, job.src) then
      local added, err2 = add_take(t.item, save_take(t.take), job.src, job.target)
      if not added then error(err2) end
      added_count = added_count + 1
    end
  end
  if added_count == 0 then error("активный тейк уже указывает на другой файл") end
  job.changed = true
end

local function apply_overwrite(job)
  ensure_undo()
  local touched = 0
  -- Reload every take in the project that uses this file, not only the selected
  -- items: a split recording shares one source between many items, and all of
  -- them would otherwise keep the old audio in memory.
  for i = 0, reaper.CountMediaItems(0) - 1 do
    local item = reaper.GetMediaItem(0, i)
    if item then
      for ti = 0, reaper.CountTakes(item) - 1 do
        local take = reaper.GetTake(item, ti)
        if take and same_source(take, job.src) then
          local props = save_take(take)
          local newsrc = reaper.PCM_Source_CreateFromFile(job.src)
          if not newsrc then error("не удалось заново открыть " .. job.src) end
          reaper.SetMediaItemTake_Source(take, newsrc)
          build_peaks(newsrc)
          restore_take(take, props)
          reaper.UpdateItemInProject(item)
          touched = touched + 1
        end
      end
    end
  end
  if touched == 0 then error("айтемы удалены или их источник успел измениться") end
  job.changed = true
end

--==============================================================================
-- launching and finishing jobs
--==============================================================================

local function build_and_launch(job)
  if MODE == "copy" then ensure_dir(job.tmpdir) end

  local args = {cli_cmd, "process"}
  if MODE == "copy" then
    args[#args + 1] = "-o"
    args[#args + 1] = shq(job.tmpdir)
  end
  if CONFIG.extra_args ~= "" then args[#args + 1] = CONFIG.extra_args end
  if has_progress then
    args[#args + 1] = "--progress-file"
    args[#args + 1] = shq(job.progress)
  end
  args[#args + 1] = "--"
  args[#args + 1] = shq(job.src)
  local cmd = table.concat(args, " ")

  local script
  if IS_WIN then
    script = table.concat({
      "@echo off",
      "chcp 65001 >nul",
      cmd .. " > " .. shq(job.log) .. " 2>&1",
      "echo %ERRORLEVEL% > " .. shq(job.rc),
    }, "\r\n") .. "\r\n"
  else
    script = table.concat({
      "#!/bin/sh",
      "echo $$ > " .. shq(job.wrapper),
      cmd .. " > " .. shq(job.log) .. " 2>&1",
      "echo $? > " .. shq(job.rc),
    }, "\n") .. "\n"
  end

  local ok, err = write_file(job.script, script)
  if not ok then return false, "не удалось записать скрипт запуска: " .. tostring(err) end

  local launched
  if IS_WIN then
    launched = os.execute('start "" /b cmd /c ' .. shq(job.script) .. ' >nul 2>&1')
  else
    launched = os.execute("sh " .. shq(job.script) .. " >/dev/null 2>&1 &")
  end
  if launched == nil or launched == false then
    return false, "не удалось запустить im4x-chain"
  end
  return true
end

local function reset_ui(job)
  ui.file = job.src
  ui.target = job.target
  ui.stages = {}
  ui.status = {}
  ui.stage_seconds = {}
  ui.total_stages = 0
  ui.done_count = 0
  ui.current = nil
  ui.current_since = reaper.time_precise()
  ui.metrics = nil
  ui.unknown = not has_progress
  ui.message = "Запуск im4x-chain…"
  ui.error = ""
  ui.summary = ""
end

local finish_all
local cleanup_and_quit

local function advance_job()
  if cancel_requested then
    finish_all("cancelled")
    return
  end
  if current >= #jobs then
    finish_all("ok")
    return
  end
  phase = "gap"
  next_job_at = reaper.time_precise() + 0.2
end

local function on_job_finished(job, rc)
  read_progress(job)                       -- final lines, before the rc file
  job.exit_code = rc

  if rc ~= 0 and cancel_requested then
    job.status = "cancelled"
    cleanup_partial(job)
    -- The CLI may have replaced the file just before it was stopped; reload
    -- anyway so the project never shows stale audio for a changed file.
    if MODE == "overwrite" then
      local ok, err = pcall(apply_overwrite, job)
      close_undo()
      if not ok then job.warning = tostring(err) end
    end
  elseif rc ~= 0 then
    job.status = "error"
    local detail = job.progress_error or last_lines(job, 700)
    job.error = string.format("код %d: %s", rc, detail ~= "" and detail or "см. лог")
    ui.error = job.error
    if MODE == "overwrite" then
      local ok, err = pcall(apply_overwrite, job)
      close_undo()
      if not ok then
        job.error = job.error .. " · перезагрузка: " .. tostring(err)
        ui.error = job.error
      end
    end
  else
    local ok, err = pcall(function()
      if MODE == "copy" then apply_copy(job) else apply_overwrite(job) end
    end)
    close_undo()
    if ok then
      job.status = "ok"
      job.metrics = job.metrics or metrics_from_log(job)
      ui.metrics = job.metrics
      ui.message = "Готово: " .. basename(job.src)
    else
      job.status = "error"
      job.error = tostring(err)
      ui.error = job.error
    end
  end

  if job.status == "error" then
    ui.message = "Ошибка: " .. job.error
  end
  advance_job()
end

local function run_job()
  if cancel_requested then finish_all("cancelled"); return end
  current = current + 1
  if current > #jobs then finish_all("ok"); return end
  local job = jobs[current]
  reset_ui(job)
  ui.message = "Файл " .. current .. " из " .. #jobs .. " — запуск…"
  if MODE == "overwrite" then
    reaper.Main_OnCommand(1016, 0)          -- transport: stop before touching media files
  end
  phase = "running"
  local ok, err = build_and_launch(job)
  if not ok then
    job.status = "error"
    job.error = err
    ui.error = err
    ui.message = "Ошибка: " .. err
    advance_job()
  end
end

finish_all = function(why)
  if phase == "done" then return end
  phase = "done"
  outcome = why or "ok"
  ui.finished_at = reaper.time_precise()
  ui.current = nil
  close_undo()
  reaper.UpdateArrange()

  if need_peaks then
    reaper.Main_OnCommand(40047, 0)      -- build missing peaks
    need_peaks = false
  end

  local ok_count, err_count, cancel_count = 0, 0, 0
  for _, j in ipairs(jobs) do
    if j.status == "ok" then ok_count = ok_count + 1
    elseif j.status == "error" then err_count = err_count + 1
    elseif j.status == "cancelled" then
      cancel_count = cancel_count + 1
      cleanup_partial(j)
    end
  end

  ui.had_error = err_count > 0
  if outcome == "cancelled" or cancel_requested then
    ui.message = "Остановлено"
  elseif err_count > 0 then
    ui.message = string.format("Завершено с ошибками: %d из %d", err_count, #jobs)
  else
    ui.message = "Готово"
  end

  local last_ok = nil
  for i = #jobs, 1, -1 do
    if jobs[i].status == "ok" then last_ok = jobs[i]; break end
  end
  if last_ok and last_ok.metrics and last_ok.metrics.after_lufs then
    local m = last_ok.metrics
    ui.summary = string.format("Итог: %.2f LUFS / TP %.2f dBTP", m.after_lufs, m.after_tp or 0)
  elseif last_ok then
    ui.summary = "Результат: " .. basename(last_ok.target)
  else
    ui.summary = ""
  end

  if MODE == "copy" and last_ok then
    ui.summary = ui.summary .. " · копия: " .. last_ok.target
  elseif MODE == "overwrite" and last_ok then
    ui.summary = ui.summary .. " · файл перезаписан"
  end
  if last_ok and last_ok.warning and last_ok.warning ~= "" then
    ui.summary = ui.summary .. " · " .. last_ok.warning
  end

  keep_run_dir = CONFIG.keep_logs or err_count > 0 or cancel_count > 0 or outcome ~= "ok"
  if not keep_run_dir then
    remove_dir(run_dir)
    run_dir_removed = true
  end
  release_lock()

  if close_when_done or window_closed then
    cleanup_and_quit()
  end
end

--==============================================================================
-- cancellation and window closing
--==============================================================================

cleanup_and_quit = function()
  if quit_done then return end
  quit_done = true
  close_undo()
  release_lock()
  if run_dir and not keep_run_dir and not run_dir_removed then
    remove_dir(run_dir)
    run_dir_removed = true
  end
  gfx.quit()
end

local close_prompt_at = 0

local function request_cancel()
  if phase ~= "starting" and phase ~= "running" and phase ~= "gap" then return end
  if cancel_requested then return end
  cancel_requested = true
  ui.message = "Останавливаю…"
  cancel_giveup_at = reaper.time_precise() + 8.0
  if phase == "running" then
    local job = jobs[current]
    if job and job.pid then
      kill_tree(job.pid, false)
      cancel_kill_at = reaper.time_precise() + 1.5
      cancel_force_at = reaper.time_precise() + 4.0
    elseif job and job.wrapper_pid then
      -- No progress file (old CLI): stop its child, the wrapper then writes rc.
      kill_tree(job.wrapper_pid, false)
      cancel_kill_at = reaper.time_precise() + 1.5
      cancel_force_at = reaper.time_precise() + 4.0
    else
      cancel_kill_at = reaper.time_precise() + 1.0
      cancel_force_at = reaper.time_precise() + 4.0
    end
  else
    finish_all("cancelled")
  end
end

local function request_close()
  if phase == "done" then
    quit_requested = true
    return
  end
  if cancel_requested or close_when_done then return end
  local now = reaper.time_precise()
  if now - close_prompt_at < 0.6 then return end
  close_prompt_at = now
  local tail = MODE == "overwrite"
    and "Уже перезаписанные файлы останутся перезаписанными."
    or "Уже добавленные копии останутся в проекте."
  local r = reaper.ShowMessageBox(
    "Остановить обработку и закрыть окно?\n" .. tail,
    "im4x-chain", 4)
  if r == 6 then
    close_when_done = true
    request_cancel()
  end
end

--==============================================================================
-- progress window
--==============================================================================

local function utf8_cut(s, n)
  if utf8 and utf8.offset then
    local ok, pos = pcall(utf8.offset, s, n + 1)
    if ok and pos then return s:sub(1, pos - 1) end
    return s
  end
  return s:sub(1, n)
end

local function fit(s, maxw)
  if not s or s == "" then return "" end
  local key = maxw .. ":" .. s
  if trunc_cache[key] then return trunc_cache[key] end
  if gfx.measurestr(s) <= maxw then
    trunc_cache[key] = s
    return s
  end
  local len = (utf8 and utf8.len and utf8.len(s)) or #s
  local out = "..."
  for n = len, 1, -1 do
    local cut = utf8_cut(s, n) .. "..."
    if gfx.measurestr(cut) <= maxw then out = cut; break end
  end
  trunc_cache[key] = out
  return out
end

local function set_rgb(r, g, b, a)
  gfx.set(r, g, b, a or 1)
end

local function text_right(s, right, y)
  gfx.x = right - gfx.measurestr(s)
  gfx.y = y
  gfx.drawstr(s)
end

local function draw_button(id, label, x, y, w, h, enabled, primary)
  local mx, my = gfx.mouse_x or 0, gfx.mouse_y or 0
  local hover = enabled and mx >= x and mx <= x + w and my >= y and my <= y + h
  if primary then
    if hover then set_rgb(0.33, 0.80, 0.58)
    else set_rgb(0.22, 0.62, 0.44) end
  else
    if hover then set_rgb(0.22, 0.25, 0.30)
    else set_rgb(0.16, 0.18, 0.22) end
  end
  gfx.rect(x, y, w, h, true)
  set_rgb(0.90, 0.93, 0.96)
  gfx.setfont(1, "Arial", 13)
  local tw = gfx.measurestr(label)
  gfx.x = x + (w - tw) / 2
  gfx.y = y + (h - 15) / 2
  gfx.drawstr(label)
  button_hit[id] = {x = x, y = y, w = w, h = h, enabled = enabled}
end

local function draw_bar(x, y, w, h, frac, now)
  set_rgb(0.16, 0.18, 0.21)
  gfx.rect(x, y, w, h, true)
  local f = math.max(0, math.min(1, frac or 0))
  if f > 0 then
    set_rgb(0.24, 0.72, 0.52)
    gfx.rect(x, y, w * f, h, true)
  end
  if ui.current and phase == "running" then
    local total = math.max(ui.total_stages, 1)
    local seg = w / total
    local px = x + ui.done_count * seg + seg * ((now * 0.55) % 1)
    set_rgb(0.45, 0.90, 0.66, 0.55)
    gfx.rect(math.max(x, px - 10), y - 1, math.min(20, x + w - math.max(x, px - 10)), h + 2, true)
  end
  set_rgb(0.07, 0.08, 0.10)
  for i = 1, math.max(ui.total_stages - 1, 0) do
    local tx = x + w * i / ui.total_stages
    gfx.rect(tx, y, 1, h, true)
  end
end

local function draw_indeterminate(x, y, w, h, now)
  set_rgb(0.16, 0.18, 0.21)
  gfx.rect(x, y, w, h, true)
  local band = w * 0.28
  local px = x + (w + band) * ((now * 0.45) % 1) - band
  set_rgb(0.24, 0.72, 0.52, 0.85)
  gfx.rect(math.max(x, px), y, math.min(band, x + w - math.max(x, px)), h, true)
end

local function draw(now)
  button_hit = {}
  local W, H = CONFIG.window_w, CONFIG.window_h

  set_rgb(0.075, 0.085, 0.105)
  gfx.rect(0, 0, W, H, true)
  set_rgb(0.105, 0.125, 0.155)
  gfx.rect(0, 0, W, 78, true)
  set_rgb(0.24, 0.62, 0.44)
  gfx.rect(0, 78, W, 2, true)

  -- header
  gfx.setfont(1, "Arial", 20)
  set_rgb(0.92, 0.95, 0.98)
  gfx.x, gfx.y = 20, 12
  gfx.drawstr("im4x-chain")

  gfx.setfont(1, "Arial", 11)
  local badge = MODE == "overwrite" and "ПЕРЕЗАПИСЬ" or "КОПИЯ"
  local bw = gfx.measurestr(badge) + 18
  local bx = W - 20 - bw
  if MODE == "overwrite" then set_rgb(0.55, 0.24, 0.22) else set_rgb(0.18, 0.45, 0.32) end
  gfx.rect(bx, 16, bw, 20, true)
  set_rgb(0.95, 0.95, 0.95)
  gfx.x = bx + 9
  gfx.y = 21
  gfx.drawstr(badge)

  gfx.setfont(1, "Arial", 12)
  set_rgb(0.62, 0.68, 0.75)
  gfx.x, gfx.y = 20, 42
  gfx.drawstr(fit(basename(ui.file), W - 180))

  if #jobs > 0 and current > 0 then
    gfx.setfont(1, "Arial", 11)
    set_rgb(0.55, 0.62, 0.70)
    text_right(string.format("файл %d из %d", math.min(current, #jobs), #jobs), W - 20, 44)
  end

  gfx.setfont(1, "Arial", 11)
  set_rgb(0.45, 0.52, 0.60)
  gfx.x, gfx.y = 20, 60
  gfx.drawstr(fit(dirname(ui.file), W - 40))

  -- stage list
  local y = 96
  if #ui.stages == 0 then
    gfx.setfont(1, "Arial", 14)
    set_rgb(0.80, 0.84, 0.88)
    gfx.x, gfx.y = 30, 140
    if ui.unknown then
      gfx.drawstr("Обработка (этот CLI не сообщает этапы)…")
    else
      gfx.drawstr("Подготовка…")
    end
    draw_indeterminate(30, 180, W - 60, 10, now)
  else
    gfx.setfont(1, "Arial", 13)
    for i, s in ipairs(ui.stages) do
      local row = y + (i - 1) * 23
      local st = ui.status[s] or "pending"
      if st == "current" then
        set_rgb(0.13, 0.20, 0.17)
        gfx.rect(20, row - 2, W - 40, 21, true)
      end
      -- status dot
      if st == "done" then
        set_rgb(0.30, 0.78, 0.55)
        gfx.circle(30, row + 8, 4, true)
        set_rgb(0.07, 0.09, 0.11)
        gfx.line(28, row + 8, 29, row + 10)
        gfx.line(29, row + 10, 33, row + 5)
      elseif st == "current" then
        local pulse = 1.5 + 1.5 * ((math.sin(now * 4) + 1) / 2)
        set_rgb(0.35, 0.85, 0.60, 0.35)
        gfx.circle(30, row + 8, 7 + pulse, true)
        set_rgb(0.45, 0.92, 0.66)
        gfx.circle(30, row + 8, 4, true)
      else
        set_rgb(0.32, 0.36, 0.42)
        gfx.circle(30, row + 8, 4, true)
      end

      if st == "done" then set_rgb(0.72, 0.78, 0.82)
      elseif st == "current" then set_rgb(0.88, 0.98, 0.92)
      else set_rgb(0.48, 0.53, 0.60) end
      gfx.x, gfx.y = 46, row
      gfx.drawstr(fit(STAGE_LABELS[s] or s, W - 200))

      gfx.setfont(1, "Arial", 11)
      set_rgb(0.42, 0.48, 0.55)
      if st == "done" and ui.stage_seconds[s] then
        text_right(string.format("%.1f с", ui.stage_seconds[s]), W - 24, row + 1)
      elseif st == "current" then
        text_right(string.format("%.0f с", now - ui.current_since), W - 24, row + 1)
      end
      gfx.setfont(1, "Arial", 13)
    end
  end

  -- progress bar
  local by = H - 108
  local total = math.max(ui.total_stages, 0)
  local frac = total > 0 and (ui.done_count / total) or 0
  if phase == "done" and outcome ~= "cancelled" and not ui.had_error then frac = 1 end
  draw_bar(20, by, W - 40, 10, frac, now)
  gfx.setfont(1, "Arial", 11)
  set_rgb(0.55, 0.62, 0.70)
  gfx.x, gfx.y = 20, by + 16
  if total > 0 then
    gfx.drawstr(string.format("%d из %d этапов", ui.done_count, total))
  else
    gfx.drawstr("выполняется…")
  end
  local elapsed = (phase == "done" and (ui.finished_at or now) or now) - run_started_at
  text_right(fmt_time(elapsed), W - 20, by + 16)

  -- messages
  gfx.setfont(1, "Arial", 12)
  if ui.error ~= "" then
    set_rgb(0.95, 0.50, 0.45)
    gfx.x, gfx.y = 20, by + 36
    gfx.drawstr(fit(ui.error, W - 40))
  else
    set_rgb(0.75, 0.80, 0.86)
    gfx.x, gfx.y = 20, by + 36
    gfx.drawstr(fit(ui.message, W - 40))
    if ui.summary ~= "" then
      set_rgb(0.50, 0.58, 0.66)
      gfx.x, gfx.y = 20, by + 54
      gfx.drawstr(fit(ui.summary, W - 40))
    end
  end

  -- buttons
  local bh = 28
  local byy = H - 38
  if phase == "running" or phase == "starting" or phase == "gap" then
    draw_button("cancel", "Отмена", W - 130, byy, 110, bh, true, false)
  else
    draw_button("close", "Закрыть", W - 130, byy, 110, bh, true, true)
  end
  if not run_dir_removed and run_dir then
    draw_button("log", "Открыть лог", 20, byy, 120, bh, true, false)
  end
end

local function handle_mouse()
  local down = (gfx.mouse_cap or 0) % 2 == 1
  if down and not mouse_was_down then
    local mx, my = gfx.mouse_x or 0, gfx.mouse_y or 0
    for id, r in pairs(button_hit) do
      if r.enabled and mx >= r.x and mx <= r.x + r.w
         and my >= r.y and my <= r.y + r.h then
        if id == "cancel" then request_cancel()
        elseif id == "close" then request_close()
        elseif id == "log" then open_path(run_dir) end
        break
      end
    end
  end
  mouse_was_down = down
end

--==============================================================================
-- main loop
--==============================================================================

local function loop()
  local now = reaper.time_precise()

  if phase == "starting" and now >= start_at then
    run_job()
  elseif phase == "gap" and now >= next_job_at then
    run_job()
  elseif phase == "running" then
    local job = jobs[current]
    if job then
      read_progress(job)
      if not job.wrapper_pid and not IS_WIN then
        local w = tonumber((read_file(job.wrapper) or ""):match("%d+"))
        if w then
          job.wrapper_pid = w
          if not job.pid then lock_update(w) end
        end
      end
      if job.pid and not job.lock_updated then
        lock_update(job.pid)
        job.lock_updated = true
      end

      local rc = read_rc(job)
      if rc ~= nil then
        on_job_finished(job, rc)
      elseif cancel_requested then
        local pid = job.pid or job.wrapper_pid
        if not pid and now >= cancel_kill_at then
          pid = tonumber((read_file(job.wrapper) or ""):match("%d+"))
          if pid then
            job.wrapper_pid = pid
            kill_tree(pid, false)
          end
          cancel_kill_at = now + 1.0
        elseif pid and now >= cancel_force_at then
          if job.pid then
            kill_tree(job.pid, true)
          else
            kill_tree(pid, true)
          end
        end
        if now >= cancel_giveup_at then
          job.status = "cancelled"
          cleanup_partial(job)
          if MODE == "overwrite" then
            pcall(apply_overwrite, job)
            close_undo()
          end
          finish_all("cancelled")
        end
      end
    end
  end

  if quit_done then return end

  if not window_closed then
    local ch = gfx.getchar()
    if ch == -1 then
      window_closed = true
      request_cancel()
    elseif ch == 27 then
      request_close()
    end
    draw(now)
    handle_mouse()
    gfx.update()
  end

  if quit_requested or (phase == "done" and (close_when_done or window_closed)) then
    if phase == "done" then ui.finished_at = now end
    cleanup_and_quit()
    return
  end
end

local function safe_loop()
  local ok, err = xpcall(loop, debug.traceback)
  if not ok then
    local job = jobs[current]
    local pid = job and (job.pid or job.wrapper_pid)
    if pid then kill_tree(pid, true) end
    release_lock()
    close_undo()
    reaper.ShowMessageBox("Ошибка скрипта im4x-chain:\n\n" .. tostring(err),
      "im4x-chain", 0)
    gfx.quit()
    return
  end
  if not quit_done and not quit_requested then reaper.defer(safe_loop) end
end

gfx.init("im4x-chain · " .. (MODE == "overwrite" and "перезапись" or "копия"),
         CONFIG.window_w, CONFIG.window_h, 0, 120, 100)
ui.message = "Подготовка…"
safe_loop()
