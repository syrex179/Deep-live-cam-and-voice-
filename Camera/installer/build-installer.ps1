[CmdletBinding()]
param([switch]$InstallInnoSetup)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$isccCandidates = @(
    'C:\Program Files (x86)\Inno Setup 6\ISCC.exe',
    'C:\Program Files\Inno Setup 6\ISCC.exe',
    (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe')
)
$iscc = $isccCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $iscc -and $InstallInnoSetup) {
    winget install --id JRSoftware.InnoSetup --exact --accept-package-agreements --accept-source-agreements
    $iscc = $isccCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $iscc) {
    throw 'Inno Setup 6 не найден. Установите его или запустите: .\installer\build-installer.ps1 -InstallInnoSetup'
}

& $iscc (Join-Path $root 'installer\Deep-Live-Studio.iss')
if ($LASTEXITCODE -ne 0) { throw 'Не удалось собрать установщик.' }
Write-Host "Готово: $(Join-Path $root 'installer-output')"
