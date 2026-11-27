-- @description im4x-chain: обработать выбранные айтемы в копию (без перезаписи)
-- @author im4x
-- @version 1.0.0
-- @about
--   Запускает цепочку im4x-chain для активного тейка каждого выбранного айтема.
--   Результат пишется рядом с исходником как <имя>_im4x.wav/.flac и добавляется
--   в айтем новым тейком; исходный файл и исходный тейк остаются на месте.
--   В окне виден этап обработки.
--
--   Нужен установленный CLI: uv tool install git+https://github.com/im4xxx/im4x-chain

_G.IM4X_MODE = "copy"

local src = debug.getinfo(1, "S").source:sub(2)
local dir = src:match("^(.*[/\\])") or ""
local ok, err = pcall(dofile, dir .. "im4x_chain_core.lua")
if not ok then
  reaper.ShowMessageBox("im4x-chain: не удалось загрузить im4x_chain_core.lua\n\n" ..
    tostring(err), "im4x-chain", 0)
end
