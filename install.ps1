# im4x-chain: installation on Windows. Started by install.cmd (double-click).
# Installs uv if needed, the im4x-chain CLI from this folder, copies the REAPER scripts and
# creates the settings file. Run it again from a newer archive to update.
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8   # paths from the CLI may be non-Latin
$env:PYTHONUTF8 = '1'
$src = $PSScriptRoot
$code = 0
function Step($t) { Write-Host "`n== $t" -ForegroundColor Cyan }

try {
    Step 'uv (устанавливает Python и библиотеки)'
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Host 'uv уже установлен'
    } else {
        Write-Host 'Устанавливаю uv (права администратора не нужны)...'
        powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv не установился' }
    }

    Step 'CLI im4x-chain'
    & uv tool install --force --reinstall "$src"
    if ($LASTEXITCODE -ne 0) { throw "uv tool install завершился с кодом $LASTEXITCODE" }
    & uv tool update-shell *> $null
    $cli = Join-Path $env:USERPROFILE '.local\bin\im4x-chain.exe'
    if (-not (Test-Path $cli)) { $cli = (Get-Command im4x-chain -ErrorAction Stop).Source }
    Write-Host "Установлен: $cli"

    Step 'Скрипты REAPER'
    $reaper = Join-Path $env:APPDATA 'REAPER'
    if (Test-Path $reaper) {
        $dst = Join-Path $reaper 'Scripts\im4x_chain'
        New-Item -ItemType Directory -Force -Path $dst | Out-Null
        Copy-Item -Path (Join-Path $src 'im4x_chain\*') -Destination $dst -Recurse -Force
        Write-Host "Скопированы в $dst"
    } else {
        Write-Host "REAPER не найден в $reaper - скопируйте папку im4x_chain в папку Scripts REAPER вручную"
        Write-Host '(в REAPER: Options -> Show REAPER resource path).'
    }

    Step 'Настройки'
    $cfg = (& $cli config --path | Out-String).Trim()
    if (Test-Path $cfg) { Write-Host "Файл настроек уже есть: $cfg" } else { & $cli config --init }

    Step 'Готово'
    Write-Host 'В REAPER один раз: Actions -> Show action list -> New action -> Load ReaScript...'
    Write-Host 'и выберите im4x_chain_overwrite.lua и im4x_chain_take.lua из папки Scripts\im4x_chain.'
    Write-Host 'Команда im4x-chain работает в новых окнах терминала.'
} catch {
    Write-Host "`nОшибка: $_" -ForegroundColor Red
    $code = 1
}
Read-Host "`nНажмите Enter, чтобы закрыть"
exit $code
