-- @description im4x-chain: обработать выбранные айтемы, перезаписав файлы
-- @author im4x
-- @version 1.0.0
-- @about
--   Запускает цепочку im4x-chain для активного тейка каждого выбранного айтема.
--   Результат заменяет исходный WAV/FLAC-файл на диске (через временный файл
--   и атомарную замену). В окне виден этап обработки; REAPER Undo не вернёт
--   перезаписанные файлы.
--
--   Нужен установленный CLI: uv tool install git+https://github.com/im4xxx/im4x-chain

_G.IM4X_MODE = "overwrite"

local src = debug.getinfo(1, "S").source:sub(2)
local dir = src:match("^(.*[/\\])") or ""
local ok, err = pcall(dofile, dir .. "im4x_chain_core.lua")
if not ok then
  reaper.ShowMessageBox("im4x-chain: не удалось загрузить im4x_chain_core.lua\n\n" ..
    tostring(err), "im4x-chain", 0)
end
