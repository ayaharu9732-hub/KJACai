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