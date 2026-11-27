-- @description im4x-chain: обработать выбранные айтемы, перезаписав исходные файлы
-- @about
--   Активный тейк каждого выбранного айтема (WAV/FLAC) проходит цепочку im4x-chain,
--   результат заменяет исходный файл, все айтемы с этим файлом перезагружаются.
--   Окно показывает этап и прогресс; Esc — отмена. Нужен CLI im4x-chain.

_G.IM4X_MODE = "overwrite"
local dir = debug.getinfo(1, "S").source:sub(2):match("^(.*[/\\])") or ""
dofile(dir .. "im4x_claude_core.lua")
