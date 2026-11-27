#!/bin/bash
# im4x-chain: installation on macOS. Double-click in Finder (the first time: right-click ->
# Open) or run `bash install.command`. Installs uv if needed, the im4x-chain CLI from this
# folder, copies the REAPER scripts and creates the settings file. Run again to update.
cd "$(dirname "$0")" || exit 1
SRC="$(pwd)"
step() { printf '\n== %s\n' "$1"; }
finish() { printf '\n'; read -r -p "Нажмите Enter, чтобы закрыть" _; exit "$1"; }
fail() { printf '\nОшибка: %s\n' "$1"; finish 1; }

step "uv (устанавливает Python и библиотеки)"
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
    echo "uv уже установлен"
else
    echo "Устанавливаю uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv не установился"
    command -v uv >/dev/null 2>&1 || fail "uv не установился"
fi

step "CLI im4x-chain"
uv tool install --force --reinstall "$SRC" || fail "uv tool install"
uv tool update-shell >/dev/null 2>&1
CLI="$HOME/.local/bin/im4x-chain"
[ -x "$CLI" ] || CLI="$(command -v im4x-chain)" || fail "im4x-chain не найден после установки"
echo "Установлен: $CLI"

step "Скрипты REAPER"
R="$HOME/Library/Application Support/REAPER"
if [ -d "$R" ]; then
    mkdir -p "$R/Scripts/im4x_chain" && cp -R im4x_chain/. "$R/Scripts/im4x_chain/" || fail "копирование скриптов"
    echo "Скопированы в $R/Scripts/im4x_chain"
else
    echo "REAPER не найден в $R — скопируйте папку im4x_chain в папку Scripts REAPER вручную"
    echo "(в REAPER: Options → Show REAPER resource path)."
fi

step "Настройки"
CFG="$("$CLI" config --path)"
if [ -f "$CFG" ]; then echo "Файл настроек уже есть: $CFG"; else "$CLI" config --init; fi

step "Готово"
echo "В REAPER один раз: Actions → Show action list → New action → Load ReaScript…"
echo "и выберите im4x_chain_overwrite.lua и im4x_chain_take.lua из папки Scripts/im4x_chain."
echo "Команда im4x-chain работает в новых окнах терминала."
finish 0
