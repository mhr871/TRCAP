param(
    [string]$Output = "..\2025tasviret_upd_colab.zip"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $root $Output))
$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("2025tasviret_upd_colab_" + [System.Guid]::NewGuid().ToString("N"))

$excludeNames = @(
    ".git",
    "__pycache__",
    "checkpoints",
    "Data",
    "experiments",
    "eval_outputs",
    "flagged"
)

New-Item -ItemType Directory -Path $tempDir | Out-Null

Get-ChildItem -Path $root -Recurse -Force -File | ForEach-Object {
    $relative = $_.FullName.Substring($root.Length).TrimStart("\", "/")
    $parts = $relative -split "[\\/]"
    foreach ($part in $parts) {
        if ($excludeNames -contains $part) {
            return
        }
    }

    $dest = Join-Path $tempDir $relative
    $destDir = Split-Path -Parent $dest
    if (!(Test-Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }
    Copy-Item -LiteralPath $_.FullName -Destination $dest
}

if (Test-Path $outputPath) {
    Remove-Item -LiteralPath $outputPath -Force
}

Compress-Archive -Path (Join-Path $tempDir "*") -DestinationPath $outputPath
Remove-Item -LiteralPath $tempDir -Recurse -Force

Write-Host "Created: $outputPath"
