-- im4x-chain for REAPER (Claude's version): shared engine of the two actions.
--
-- The entry scripts set IM4X_MODE ("overwrite" or "take") and dofile() this file.
-- For every selected item the active take's WAV/FLAC file goes through the im4x-chain CLI,
-- one file at a time, in the background; a window shows the stage and progress bars.
--   overwrite: the file is processed in place (the CLI writes a temp file and swaps it in),
--              then every item that plays this file is reloaded.
--   take:      the result is written next to the source as <name>_im4x.<ext> and added to
--              the item as a new, active take; the original file and take stay as they are.
-- Esc or closing the window cancels: the CLI is killed before it swaps files in, so the
-- file being processed stays untouched; files already finished stay finished.

local MODE = _G.IM4X_MODE or "take"
local TITLE = MODE == "overwrite" and "im4x-chain — перезапись файла" or "im4x-chain — новый тейк"
local EXT_SECTION = "im4x_claude"
local IS_WIN = reaper.GetOS():match("^Win") ~= nil

-- stage names from the CLI's progress events, and rough shares of the processing time
-- (measured on a 24-minute voice file); a stage's bar creeps by time inside the stage
local STAGES = {
  read = { "Чтение", 2 }, voice = { "Тон голоса", 3 }, eq = { "EQ", 2 },
  deplosive = { "De-plosive", 12 }, declick = { "Mouth De-click", 15 },
  denoise = { "Spectral De-noise", 35 }, leveler = { "Leveler", 8 },
  compressor = { "Компрессор", 3 }, loudness = { "Loudness", 5 },
  write = { "Запись", 5 }, measure = { "Замер", 4 },
}
local SECONDS_PER_AUDIO_SECOND = 0.02   -- ~25 s for 24 min of audio on an M-series Mac
local STARTUP_SECONDS = 4               -- Python and plugin start-up

---------------------------------------------------------------------------- helpers

local function shq(s)                        -- quote for /bin/sh
  return "'" .. tostring(s):gsub("'", "'\\''") .. "'"
end

local function exists(path)
  local f = io.open(path, "rb")
  if f then f:close() return true end
  return false
end

local function read_all(path)
  local f = io.open(path, "rb")
  if not f then return nil end
  local s = f:read("a")
  f:close()
  return s
end

local function write_all(path, s)
  local f = assert(io.open(path, "wb"))
  f:write(s)
  f:close()
end

local function dirname(p) return p:match("^(.*)[/\\][^/\\]*$") or "." end
local function basename(p) return p:match("([^/\\]+)$") or p end
local function splitext(name) local s, e = name:match("^(.*)(%.[^.]+)$") return s or name, e or "" end

local function fmt_time(sec)
  sec = math.max(0, math.floor(sec + 0.5))
  if sec >= 3600 then return string.format("%d:%02d:%02d", sec // 3600, sec % 3600 // 60, sec % 60) end
  return string.format("%d:%02d", sec // 60, sec % 60)
end

-- the progress file is JSON lines written by our own CLI: flat objects with strings and numbers
local function json_field(line, key)
  if line:match('"' .. key .. '"%s*:%s*""') then return "" end
  local s = line:match('"' .. key .. '"%s*:%s*"(.-[^\\])"')
  if s then return (s:gsub('\\"', '"'):gsub("\\\\", "\\")) end
  local n = line:match('"' .. key .. '"%s*:%s*(-?[%d.eE+-]+)')
  if n then return tonumber(n) end
  local b = line:match('"' .. key .. '"%s*:%s*(%a+)')
  if b == "true" then return true elseif b == "false" then return false end
  return nil
end

local function msg(text, kind) return reaper.ShowMessageBox(text, "im4x-chain", kind or 0) end

---------------------------------------------------------------------------- the CLI

local function find_cli()
  local candidates = {}
  local saved = reaper.GetExtState(EXT_SECTION, "cli")
  if saved ~= "" then candidates[#candidates + 1] = saved end
  local env = os.getenv("IM4X_CHAIN_CLI")
  if env and env ~= "" then candidates[#candidates + 1] = env end
  local home = os.getenv("HOME") or os.getenv("USERPROFILE") or ""
  for _, p in ipairs({ home .. "/.local/bin/im4x-chain", "/opt/homebrew/bin/im4x-chain",
                       "/usr/local/bin/im4x-chain" }) do
    candidates[#candidates + 1] = p
  end
  for _, p in ipairs(candidates) do
    if exists(p) then return p end
  end
  -- REAPER started from the Dock has a short PATH: ask a login shell
  local h = io.popen("/bin/sh -lc 'command -v im4x-chain' 2>/dev/null")
  if h then
    local p = (h:read("l") or ""):gsub("%s+$", "")
    h:close()
    if p ~= "" and exists(p) then return p end
  end
  return nil
end

local function cli_has_progress(cli)
  local h = io.popen(shq(cli) .. " process --help 2>&1")
  if not h then return false end
  local out = h:read("a") or ""
  h:close()
  return out:find("--progress-file", 1, true) ~= nil
end

---------------------------------------------------------------------------- processes

local function children(pid)
  local kids = {}
  local h = io.popen("pgrep -P " .. pid .. " 2>/dev/null")
  if h then
    for line in h:lines() do kids[#kids + 1] = tonumber(line) end
    h:close()
  end
  return kids
end

-- the CLI spawns render workers; collect the whole tree first, then kill it
local function kill_tree(pid)
  if not pid then return end
  local all, queue = {}, { pid }
  while #queue > 0 do
    local p = table.remove(queue)
    all[#all + 1] = p
    for _, k in ipairs(children(p)) do queue[#queue + 1] = k end
  end
  for i = #all, 1, -1 do os.execute("kill -TERM " .. all[i] .. " >/dev/null 2>&1") end
end

---------------------------------------------------------------------------- project

local function take_file(take)
  local src = reaper.GetMediaItemTake_Source(take)
  if not src then return nil end
  local parent = reaper.GetMediaSourceParent(src)          -- section / reversed sources
  while parent do src = parent parent = reaper.GetMediaSourceParent(src) end
  local path = reaper.GetMediaSourceFileName(src, "")
  if not path or path == "" then return nil end
  return path, reaper.GetMediaSourceLength(src)
end

local function all_takes_of(path)
  local list = {}
  for i = 0, reaper.CountMediaItems(0) - 1 do
    local item = reaper.GetMediaItem(0, i)
    for t = 0, reaper.CountTakes(item) - 1 do
      local take = reaper.GetTake(item, t)
      if take and not reaper.TakeIsMIDI(take) and take_file(take) == path then
        list[#list + 1] = { item = item, take = take }
      end
    end
  end
  return list
end

local function with_selection(items, fn)
  local saved = {}
  for i = 0, reaper.CountSelectedMediaItems(0) - 1 do saved[#saved + 1] = reaper.GetSelectedMediaItem(0, i) end
  reaper.SelectAllMediaItems(0, false)
  for _, it in ipairs(items) do reaper.SetMediaItemSelected(it, true) end
  fn()
  reaper.SelectAllMediaItems(0, false)
  for _, it in ipairs(saved) do
    if reaper.ValidatePtr2(0, it, "MediaItem*") then reaper.SetMediaItemSelected(it, true) end
  end
end

local function items_of(takes)
  local seen, items = {}, {}
  for _, t in ipairs(takes) do
    if not seen[t.item] then seen[t.item] = true items[#items + 1] = t.item end
  end
  return items
end

local function set_offline(items, offline)
  -- 40440 / 40439: Item: Set selected media temporarily offline / online
  with_selection(items, function() reaper.Main_OnCommand(offline and 40440 or 40439, 0) end)
end

local function unique_output(path)
  local stem, ext = splitext(path)
  local out = stem .. "_im4x" .. ext
  local n = 2
  while exists(out) do
    out = stem .. "_im4x" .. n .. ext
    n = n + 1
  end
  return out
end

local function add_take(item, old, file)
  local src = reaper.PCM_Source_CreateFromFile(file)
  if not src then return false end
  local take = reaper.AddTakeToMediaItem(item)
  reaper.SetMediaItemTake_Source(take, src)
  for _, k in ipairs({ "D_STARTOFFS", "D_PLAYRATE", "D_PITCH", "B_PPITCH", "I_PITCHMODE",
                       "I_CHANMODE", "D_VOL", "D_PAN" }) do
    reaper.SetMediaItemTakeInfo_Value(take, k, reaper.GetMediaItemTakeInfo_Value(old, k))
  end
  reaper.GetSetMediaItemTakeInfo_String(take, "P_NAME", basename(file), true)
  reaper.SetActiveTake(take)
  return true
end

---------------------------------------------------------------------------- jobs

local function collect_jobs()
  local jobs, by_path, skipped = {}, {}, {}
  for i = 0, reaper.CountSelectedMediaItems(0) - 1 do
    local item = reaper.GetSelectedMediaItem(0, i)
    local take = reaper.GetActiveTake(item)
    if take and not reaper.TakeIsMIDI(take) then
      local path, len = take_file(take)
      local ext = path and select(2, splitext(path)):lower()
      if path and (ext == ".wav" or ext == ".flac") and exists(path) then
        local job = by_path[path]
        if not job then
          job = { path = path, length = len or 0, items = {} }
          by_path[path] = job
          jobs[#jobs + 1] = job
        end
        job.items[#job.items + 1] = { item = item, take = take }
      else
        skipped[#skipped + 1] = path and basename(path) or "(айтем без файла)"
      end
    end
  end
  return jobs, skipped
end

local work_root
local function new_workdir()
  local base = (os.getenv("TMPDIR") or "/tmp"):gsub("/$", "")
  work_root = string.format("%s/im4x-reaper-%d-%d", base, math.floor(reaper.time_precise() * 1000), math.random(1000000))
  os.execute("mkdir -p " .. shq(work_root))
end

local function start_job(job, cli, n, with_progress)
  local w = work_root .. "/" .. n
  os.execute("mkdir -p " .. shq(w))
  job.progress_file, job.log, job.exit_file, job.pid_file = w .. "/progress.jsonl", w .. "/log.txt",
    w .. "/exit", w .. "/pid"
  local cmd = shq(cli) .. " process " .. shq(job.path)
  if MODE == "take" then
    job.stage_dir = dirname(job.path) .. "/.im4x-" .. basename(work_root) .. "-" .. n
    cmd = cmd .. " -o " .. shq(job.stage_dir)
  end
  if with_progress then cmd = cmd .. " --progress-file " .. shq(job.progress_file) end
  local script = w .. "/run.sh"
  write_all(script, table.concat({
    "#!/bin/sh",
    'export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"',
    "echo $$ > " .. shq(job.pid_file),
    cmd .. " > " .. shq(job.log) .. " 2>&1",
    "echo $? > " .. shq(job.exit_file) .. ".tmp && mv " .. shq(job.exit_file) .. ".tmp " .. shq(job.exit_file),
    "",
  }, "\n"))
  os.execute("/bin/sh " .. shq(script) .. " >/dev/null 2>&1 &")
  job.started = reaper.time_precise()
  job.stage, job.stage_started, job.done_weight, job.total_weight = nil, job.started, 0, 0
  job.expected = STARTUP_SECONDS + job.length * SECONDS_PER_AUDIO_SECOND
  job.offset = 0
end

-- read new progress lines; returns nothing, updates the job
local function poll_progress(job)
  local f = io.open(job.progress_file, "rb")
  if not f then return end
  f:seek("set", job.offset)
  while true do
    local line = f:read("l")
    if not line then break end
    if line:sub(-1) ~= "}" then break end               -- a half-written line: next time
    job.offset = f:seek()
    local ev = json_field(line, "event")
    if ev == "begin" then
      job.cli_pid = json_field(line, "pid")
      job.total_weight = 0
      for name in tostring(json_field(line, "stages") or ""):gmatch("[^,]+") do
        job.total_weight = job.total_weight + (STAGES[name] and STAGES[name][2] or 2)
      end
    elseif ev == "stage" then
      local name = json_field(line, "stage")
      local w = STAGES[name] and STAGES[name][2] or 2
      if json_field(line, "status") == "start" then
        job.stage, job.stage_started = name, reaper.time_precise()
        job.stage_index, job.stage_total = json_field(line, "index"), json_field(line, "total")
      else
        job.done_weight = job.done_weight + w
        job.stage = nil
      end
    elseif ev == "done" then
      job.after = string.format("%.2f LUFS, TP %.2f dBTP", json_field(line, "after_lufs") or 0,
        json_field(line, "after_tp") or 0)
      local warn = json_field(line, "warning")
      if warn and warn ~= "" then job.warning = warn end
    elseif ev == "error" then
      job.error = json_field(line, "message")
    end
  end
  f:close()
end

-- 0..1 for the file being processed
local function job_fraction(job, now)
  if job.total_weight > 0 then
    local f = job.done_weight
    if job.stage then
      local w = STAGES[job.stage] and STAGES[job.stage][2] or 2
      local expected = math.max(0.5, job.expected * w / job.total_weight)
      f = f + w * math.min(0.95, math.max(0, now - job.stage_started) / expected)
    end
    return math.min(1, f / job.total_weight)
  end
  return math.min(0.95, math.max(0, now - job.started) / job.expected)   -- no stage events (yet)
end

---------------------------------------------------------------------------- finishing a job

local function finish_job(job)
  local code = tonumber((read_all(job.exit_file) or ""):match("%d+"))
  job.ok = code == 0 and not job.error
  if not job.ok and not job.error then
    local log = (read_all(job.log) or ""):gsub("%s+$", "")
    job.error = log ~= "" and log:sub(-600) or ("код выхода " .. tostring(code))
  end
  if MODE == "overwrite" then
    set_offline(items_of(all_takes_of(job.path)), false)
    if job.ok then
      with_selection(items_of(all_takes_of(job.path)), function() reaper.Main_OnCommand(40441, 0) end) -- rebuild peaks
    end
  else
    local staged = job.stage_dir .. "/" .. basename(job.path)
    if job.ok and exists(staged) then
      local out = unique_output(job.path)
      local moved, err = os.rename(staged, out)
      if moved then
        job.output = out
        reaper.Undo_BeginBlock()
        for _, t in ipairs(job.items) do
          if reaper.ValidatePtr2(0, t.item, "MediaItem*") and reaper.ValidatePtr2(0, t.take, "MediaItem_Take*") then
            add_take(t.item, t.take, out)
          end
        end
        reaper.Undo_EndBlock("im4x-chain: новый тейк " .. basename(out), -1)
        reaper.Main_OnCommand(40047, 0)                    -- build any missing peaks
      else
        job.ok, job.error = false, "не удалось переименовать результат: " .. tostring(err)
      end
    elseif job.ok then
      job.ok, job.error = false, "CLI не создал файл " .. staged
    end
    os.execute("rm -rf " .. shq(job.stage_dir))
  end
  reaper.UpdateArrange()
end

local function cancel_job(job)
  local sh_pid = tonumber((read_all(job.pid_file) or ""):match("%d+"))
  kill_tree(sh_pid)
  kill_tree(job.cli_pid)
  if MODE == "overwrite" then
    local stem, ext = splitext(basename(job.path))       -- the CLI's temp file, left by the kill
    os.remove(dirname(job.path) .. "/." .. stem .. ".im4xchain.tmp" .. ext)
    set_offline(items_of(all_takes_of(job.path)), false)
  elseif job.stage_dir then
    os.execute("rm -rf " .. shq(job.stage_dir))
  end
  reaper.UpdateArrange()
end

---------------------------------------------------------------------------- window

local W, H = 520, 196
local function bar(x, y, w, h, frac, r, g, b)
  gfx.set(0.22, 0.22, 0.24) gfx.rect(x, y, w, h, 1)
  gfx.set(r, g, b) gfx.rect(x, y, math.floor(w * math.max(0, math.min(1, frac)) + 0.5), h, 1)
end

local function text(x, y, s, r, g, b)
  gfx.set(r or 0.9, g or 0.9, b or 0.9)
  gfx.x, gfx.y = x, y
  gfx.drawstr(s)
end

local function fit(s, w)                  -- cut a long name from the left, keep the end
  if gfx.measurestr(s) <= w then return s end
  local cut = s
  while #cut > 1 and gfx.measurestr("…" .. cut) > w do
    cut = cut:sub(2)
    while #cut > 0 and cut:byte(1) >= 0x80 and cut:byte(1) < 0xC0 do cut = cut:sub(2) end
  end
  return "…" .. cut
end

---------------------------------------------------------------------------- main

local function main()
  local jobs, skipped = collect_jobs()
  if #jobs == 0 then
    msg("Выделите айтемы с WAV или FLAC: обрабатывается активный тейк каждого айтема." ..
      (#skipped > 0 and ("\n\nПропущены: " .. table.concat(skipped, ", ")) or ""))
    return
  end
  local cli = find_cli()
  if not cli then
    msg("Не найден CLI im4x-chain.\n\nУстановите его (uv tool install …) или укажите путь:\n" ..
      "reaper.SetExtState(\"" .. EXT_SECTION .. "\", \"cli\", \"/путь/к/im4x-chain\", true)")
    return
  end
  local with_progress = cli_has_progress(cli)

  if MODE == "overwrite" then
    local lines, others = {}, 0
    for _, j in ipairs(jobs) do
      lines[#lines + 1] = "• " .. basename(j.path)
      others = others + #items_of(all_takes_of(j.path)) - #items_of(j.items)
    end
    local text_ = "Файлы будут перезаписаны результатом цепочки. Undo их не вернёт.\n\n" ..
      table.concat(lines, "\n", 1, math.min(#lines, 12)) .. (#lines > 12 and "\n…" or "")
    if others > 0 then
      text_ = text_ .. "\n\nЭти файлы играют ещё " .. others .. " невыделенных айтема(ов): они тоже изменятся."
    end
    -- IM4X_ASSUME_YES: set by test harnesses only
    if not _G.IM4X_ASSUME_YES and msg(text_ .. "\n\nПродолжить?", 1) ~= 1 then return end
  end
  if #skipped > 0 then
    msg("Пропущены (не WAV/FLAC или файла нет):\n" .. table.concat(skipped, "\n"))
  end

  new_workdir()
  local total_len = 0
  for _, j in ipairs(jobs) do total_len = total_len + math.max(1, j.length) end

  local current, done_len, t0 = 0, 0, reaper.time_precise()
  local job, cancelled, finished_at
  local results = {}

  local function next_job()
    current = current + 1
    job = jobs[current]
    if not job then return end
    if MODE == "overwrite" then set_offline(items_of(all_takes_of(job.path)), true) end
    start_job(job, cli, current, with_progress)
  end

  local function cleanup()
    if job and not job.finished then cancel_job(job) end
    if work_root then os.execute("rm -rf " .. shq(work_root)) end
  end
  reaper.atexit(cleanup)

  local function summary()
    local ok, bad = 0, {}
    for _, j in ipairs(results) do
      if j.ok then ok = ok + 1 elseif not j.cancelled then bad[#bad + 1] = basename(j.path) .. ":\n" .. tostring(j.error) end
    end
    local warns = {}
    for _, j in ipairs(results) do
      if j.warning then warns[#warns + 1] = basename(j.path) .. ": " .. j.warning end
    end
    if #bad > 0 or #warns > 0 or cancelled then
      local s = string.format("Готово: %d из %d.", ok, #jobs)
      if cancelled then s = s .. " Остановлено: текущий файл не изменён, следующие не обработаны." end
      if #bad > 0 then s = s .. "\n\nОшибки:\n" .. table.concat(bad, "\n\n") end
      if #warns > 0 then s = s .. "\n\nПредупреждения:\n" .. table.concat(warns, "\n") end
      msg(s)
    end
  end

  gfx.init(TITLE, W, H)
  gfx.setfont(1, "Arial", 15)
  gfx.setfont(2, "Arial", 13)
  next_job()

  local function loop()
    local now = reaper.time_precise()
    local ch = gfx.getchar()

    if not finished_at then
      if ch == -1 or ch == 27 then                          -- closed or Esc
        cancelled = true
        if job then
          cancel_job(job)
          job.finished, job.ok, job.cancelled = true, false, true
          results[#results + 1] = job
        end
        finished_at = now
      elseif job then
        if with_progress then poll_progress(job) end
        if exists(job.exit_file) then
          if with_progress then poll_progress(job) end
          finish_job(job)
          job.finished = true
          results[#results + 1] = job
          done_len = done_len + math.max(1, job.length)
          next_job()
          if not job then finished_at = now end
        end
      end
    end

    -- draw
    gfx.set(0.13, 0.13, 0.14) gfx.rect(0, 0, gfx.w, gfx.h, 1)
    local x, w = 16, gfx.w - 32
    gfx.setfont(1)
    if finished_at then
      local ok = 0
      for _, j in ipairs(results) do if j.ok then ok = ok + 1 end end
      text(x, 16, cancelled and "Остановлено" or "Готово", 1, 1, 1)
      gfx.setfont(2)
      text(x, 42, string.format("Обработано файлов: %d из %d, время %s", ok, #jobs, fmt_time(finished_at - t0)), 0.75, 0.75, 0.78)
      local last = results[#results]
      if last and last.after then text(x, 62, "Последний: " .. last.after, 0.75, 0.75, 0.78) end
      bar(x, 90, w, 12, cancelled and done_len / total_len or 1, 0.35, 0.75, 0.45)
      if ch == -1 or ch == 27 or ch == 13 or now - finished_at > (cancelled and 0.3 or 1.5) then
        gfx.quit()
        summary()
        return
      end
    else
      local frac = job_fraction(job, now)
      text(x, 14, fit(string.format("Файл %d из %d: %s", current, #jobs, basename(job.path)), w), 1, 1, 1)
      gfx.setfont(2)
      local stage = job.stage and STAGES[job.stage] and STAGES[job.stage][1] or job.stage
      local line
      if not with_progress then
        line = "Обработка… (CLI без --progress-file: этапы не видны, полоса по оценке времени)"
      elseif stage then
        line = string.format("Этап %d из %d: %s", job.stage_index or 0, job.stage_total or 0, stage)
      else
        line = job.total_weight > 0 and "…" or "Запуск…"
      end
      text(x, 38, fit(line, w), 0.75, 0.75, 0.78)
      bar(x, 60, w, 14, frac, 0.32, 0.6, 0.95)
      text(x, 80, string.format("%d%%   %s", math.floor(frac * 100), fmt_time(now - job.started)), 0.6, 0.6, 0.64)
      if #jobs > 1 then
        local all = (done_len + math.max(1, job.length) * frac) / total_len
        text(x, 106, "Всего", 0.75, 0.75, 0.78)
        bar(x, 126, w, 10, all, 0.35, 0.75, 0.45)
        text(x, 140, string.format("%d%%   %s", math.floor(all * 100), fmt_time(now - t0)), 0.6, 0.6, 0.64)
      end
      text(x, gfx.h - 24, "Esc или закрыть окно — отмена (текущий файл не изменится)", 0.5, 0.5, 0.54)
    end
    gfx.update()
    reaper.defer(loop)
  end
  loop()
end

if IS_WIN then
  msg("Эта версия скриптов работает на macOS и Linux.")
else
  math.randomseed(math.floor(reaper.time_precise() * 1000))
  main()
end
