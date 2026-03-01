param(
  [string]$Repo = "C:\Users\Futamura\KJACai",
  [switch]$ContinueRebase
)

$ErrorActionPreference = "Stop"

Set-Location $Repo

$path = Join-Path $Repo ".gitignore"
if (!(Test-Path $path)) {
  throw ".gitignore not found: $path"
}

# backup
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$bak = "$path.bak_$ts"
Copy-Item $path $bak -Force
Write-Host "[BAK] $bak"

# read all lines
$lines = Get-Content $path -Encoding UTF8

# If conflict markers exist, merge both sides.
# Strategy:
# - Remove lines between markers but keep BOTH sides content (excluding markers).
# - Then de-duplicate while keeping order.
$inConflict = $false
$mode = "normal"  # normal | ours | theirs
$merged = New-Object System.Collections.Generic.List[string]
$bufOurs = New-Object System.Collections.Generic.List[string]
$bufTheirs = New-Object System.Collections.Generic.List[string]

function Flush-Conflict {
  param($mergedRef, $oursRef, $theirsRef)
  # keep both sides; order: ours then theirs
  foreach ($l in $oursRef) { $mergedRef.Add($l) }
  foreach ($l in $theirsRef) { $mergedRef.Add($l) }
  $oursRef.Clear()
  $theirsRef.Clear()
}

foreach ($l in $lines) {
  if ($l -match '^<<<<<<< ') {
    $inConflict = $true
    $mode = "ours"
    continue
  }
  if ($inConflict -and $l -match '^=======') {
    $mode = "theirs"
    continue
  }
  if ($inConflict -and $l -match '^>>>>>>> ') {
    # end conflict
    Flush-Conflict -mergedRef $merged -oursRef $bufOurs -theirsRef $bufTheirs
    $inConflict = $false
    $mode = "normal"
    continue
  }

  if ($inConflict) {
    if ($mode -eq "ours") { $bufOurs.Add($l) | Out-Null }
    elseif ($mode -eq "theirs") { $bufTheirs.Add($l) | Out-Null }
    continue
  }

  $merged.Add($l) | Out-Null
}

# If file ended while still in conflict (rare), flush anyway
if ($inConflict) {
  Flush-Conflict -mergedRef $merged -oursRef $bufOurs -theirsRef $bufTheirs
}

# Normalize: trim trailing spaces, drop empty duplicates
# De-duplicate while preserving order (case-sensitive)
$seen = @{}
$final = New-Object System.Collections.Generic.List[string]

foreach ($l in $merged) {
  $x = $l.TrimEnd()
  # keep empty lines but avoid multiple consecutive empty lines
  if ($x -eq "") {
    if ($final.Count -gt 0 -and $final[$final.Count-1] -eq "") { continue }
    $final.Add("") | Out-Null
    continue
  }
  if (-not $seen.ContainsKey($x)) {
    $seen[$x] = $true
    $final.Add($x) | Out-Null
  }
}

# Ensure MUST-have ignores (safety)
$must = @(
  "",
  "# --- Python ---",
  "__pycache__/",
  "*.pyc",
  "",
  "# --- Secrets ---",
  ".env",
  ".env.*",
  "src/GPTsupo-tu.txt",
  "",
  "# --- Logs ---",
  "*.log",
  "error.log",
  "",
  "# --- Outputs / heavy ---",
  "output/",
  "models/",
  "artifacts/",
  "",
  "# --- External repos / vendor ---",
  "AlphaPose/",
  "",
  "# --- ML weights ---",
  "*.pt",
  "*.pth"
)

foreach ($m in $must) {
  if ($m -eq "") {
    # ensure there is at least one blank line before a section (optional)
    continue
  }
  if (-not $seen.ContainsKey($m)) {
    $seen[$m] = $true
    $final.Add($m) | Out-Null
  }
}

# Write back
Set-Content -Path $path -Value $final -Encoding UTF8
Write-Host "[OK] .gitignore conflict markers removed + merged + ensured must-have ignores."

# Sanity check: verify no conflict markers remain
$check = Get-Content $path -Encoding UTF8 | Select-String -Pattern '^(<<<<<<<|=======|>>>>>>>)' -Quiet
if ($check) {
  throw "Conflict markers still remain in .gitignore. Open it and fix manually."
}

# Stage .gitignore
git add .gitignore | Out-Null
Write-Host "[OK] staged .gitignore"

if ($ContinueRebase) {
  Write-Host "[RUN] git rebase --continue"
  git rebase --continue
  Write-Host "[OK] rebase continued."
  Write-Host "If it stops again with another .gitignore conflict, re-run this script."
} else {
  Write-Host "Next: git rebase --continue"
}
