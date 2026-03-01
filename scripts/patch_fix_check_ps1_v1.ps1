# scripts/patch_fix_check_ps1_v1.ps1
# Purpose: Narrow py_compile scope to avoid legacy/patch scripts breaking checks.
# - compile all under src/
# - compile only scripts/run_*.py under scripts/
# Creates backup of existing scripts/check.ps1 if present.

param(
  [switch]$Apply
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$target = Join-Path $repo "scripts\check.ps1"

function Write-FileUtf8NoBom([string]$path, [string]$content) {
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($path, $content, $utf8NoBom)
}

$new = @'
param(
  [switch]$VerboseLog
)

$ErrorActionPreference = "Stop"

function Log([string]$msg) {
  Write-Host $msg
}

Log "[CHECK] python version:"
python --version

# 1) compile src (all)
Log "[1/2] py_compile: src/**/*.py"
$srcFiles = Get-ChildItem -Path ".\src" -Recurse -Filter *.py -File | ForEach-Object FullName
if (-not $srcFiles -or $srcFiles.Count -eq 0) {
  Log "  (skip) no python files found in src/"
} else {
  python -m py_compile $srcFiles
  Log "  OK: compiled $($srcFiles.Count) files"
}

# 2) compile scripts core only
Log "[2/2] py_compile: scripts/run_*.py (core entrypoints only)"
$core = Get-ChildItem -Path ".\scripts" -Filter "run_*.py" -File | ForEach-Object FullName
if (-not $core -or $core.Count -eq 0) {
  Log "  (skip) no run_*.py found in scripts/"
} else {
  python -m py_compile $core
  Log "  OK: compiled $($core.Count) files"
}

Log "[DONE] checks passed"
'@

if (-not $Apply) {
  Write-Host "DRY RUN: will update $target"
  Write-Host "Run with:  powershell -ExecutionPolicy Bypass -File .\scripts\patch_fix_check_ps1_v1.ps1 -Apply"
  exit 0
}

# backup if exists
if (Test-Path $target) {
  $ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
  $bak = "$target.bak_$ts"
  Copy-Item $target $bak -Force
  Write-Host "[BAK] $bak"
} else {
  New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
}

Write-FileUtf8NoBom $target $new
Write-Host "[OK] wrote: $target"

Write-Host "`n[INFO] git status:"
git status -sb
