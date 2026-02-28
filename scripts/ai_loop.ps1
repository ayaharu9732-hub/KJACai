param(
  [Parameter(Mandatory=$true)]
  [string]$Goal,

  [string]$CodexCmd = "codex",

  [string]$TestCmd = "powershell -ExecutionPolicy Bypass -File .\scripts\run_tests.ps1 -ResultPath .\artifacts\test-results.xml",

  [switch]$IncludeGitDiff,

  [string]$EnvId,
  [string]$CodexConfigPath,

  # 明示したい場合だけ使う（通常は自動検出）
  [string]$Branch,

  [int]$WaitMinutes = 10,
  [int]$PollSeconds = 5
)

$ErrorActionPreference = "Stop"

function Require-Cmd($name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "Command not found: $name"
  }
}

function Get-RepoRoot {
  $This = if ($PSCommandPath) { $PSCommandPath } else { $MyInvocation.MyCommand.Path }
  if (-not $This) { throw "Cannot resolve script path." }
  return (Resolve-Path (Join-Path (Split-Path -Parent $This) "..")).Path
}

function CmdUtf8($cmdline) {
  $full = "chcp 65001 >nul & $cmdline"
  $out = cmd /c $full 2>&1
  return ($out -join "`n")
}

function Save-Text {
  param([string]$Path, [string]$Text)
  $dir = Split-Path -Parent $Path
  if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
  $Text | Out-File -Encoding utf8 $Path
}

function Get-DefaultCodexConfigPath {
  if ($CodexConfigPath) { return $CodexConfigPath }
  return (Join-Path $env:USERPROFILE ".codex\config.toml")
}

function Get-CodexEnvIdFromConfig {
  param([string]$ConfigPath)
  if (-not (Test-Path $ConfigPath)) { return $null }
  $text = Get-Content -Raw -Encoding utf8 $ConfigPath

  $m = [regex]::Match($text, "(env_[A-Za-z0-9]+)")
  if ($m.Success) { return $m.Groups[1].Value }

  $m2 = [regex]::Match($text, '(?im)^\s*(default_env|env|environment)\s*=\s*"([^"]+)"\s*$')
  if ($m2.Success) { return $m2.Groups[2].Value }

  return $null
}

function Get-CodexEnvIdFromFile {
  param([string]$RepoRoot)
  $envFile = Join-Path $RepoRoot "artifacts\.codex_env_id.txt"
  if (-not (Test-Path $envFile)) { return $null }
  $v = (Get-Content -Raw -Encoding ascii $envFile).Trim()
  if ($v) { return $v }
  return $null
}

function Extract-TaskInfo {
  param([string]$Text)

  $taskId = $null
  $taskUrl = $null

  $mUrl = [regex]::Match($Text, "(https?://\S+/codex/tasks/(task_e_[a-zA-Z0-9]+))")
  if ($mUrl.Success) {
    $taskUrl = $mUrl.Groups[1].Value
    $taskId = $mUrl.Groups[2].Value
    return @{ taskId = $taskId; taskUrl = $taskUrl }
  }

  $mId = [regex]::Match($Text, "(task_e_[a-zA-Z0-9]+)")
  if ($mId.Success) { $taskId = $mId.Groups[1].Value }

  return @{ taskId = $taskId; taskUrl = $taskUrl }
}

function Get-RemoteDefaultBranch {
  # 1) origin/HEAD から取る（これが一番正確）
  try {
    $head = (git symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>$null).Trim()
    if ($head -match "^origin/(.+)$") { return $Matches[1] }
  } catch { }

  # 2) remote show origin から HEAD branch を読む
  try {
    $txt = (git remote show origin 2>$null)
    foreach ($line in $txt) {
      if ($line -match "HEAD branch:\s+(.+)$") {
        return $Matches[1].Trim()
      }
    }
  } catch { }

  # 3) 最後にローカル現在ブランチ
  try {
    $cur = (git branch --show-current).Trim()
    if ($cur) { return $cur }
  } catch { }

  return "main"
}

function Wait-TaskReady {
  param(
    [string]$CodexCmd,
    [string]$TaskId,
    [int]$WaitMinutes,
    [int]$PollSeconds,
    [string]$ArtifactsDir,
    [string]$TaskUrl
  )

  $deadline = (Get-Date).AddMinutes($WaitMinutes)

  while ($true) {
    $stText = CmdUtf8 "$CodexCmd cloud status $TaskId"
    $stText = $stText.Trim()
    Write-Host "[AI_LOOP] status: $stText"

    if ($stText -match "\[ERROR\]" -or $stText -match "ERROR") {
      $statusPath = Join-Path $ArtifactsDir ("codex_status_{0}.log" -f $TaskId)
      Save-Text -Path $statusPath -Text $stText

      # diff は取れないことがあるので try
      try {
        $diffPath = Join-Path $ArtifactsDir ("codex_diff_{0}.patch" -f $TaskId)
        $diffText = CmdUtf8 "$CodexCmd cloud diff $TaskId"
        Save-Text -Path $diffPath -Text $diffText
      } catch {
        # ignore
      }

      if ($TaskUrl) { Write-Host "[AI_LOOP] Task URL: $TaskUrl" }
      throw "Codex task ERROR: $TaskId (see $statusPath)"
    }

    if ($stText -match "\[READY\]" -or $stText -match "READY" -or $stText -match "COMPLETED" -or $stText -match "DONE") {
      return
    }

    if ((Get-Date) -gt $deadline) {
      $statusPath = Join-Path $ArtifactsDir ("codex_status_{0}_timeout.log" -f $TaskId)
      Save-Text -Path $statusPath -Text $stText
      throw "Timeout waiting for task to become READY: $TaskId (see $statusPath)"
    }

    Start-Sleep -Seconds $PollSeconds
  }
}

function Save-TaskDiff {
  param(
    [string]$CodexCmd,
    [string]$TaskId,
    [string]$ArtifactsDir
  )

  $diffPath = Join-Path $ArtifactsDir ("codex_diff_{0}.patch" -f $TaskId)
  $diffText = CmdUtf8 "$CodexCmd cloud diff $TaskId"
  Save-Text -Path $diffPath -Text $diffText
  Write-Host "[AI_LOOP] diff saved: $diffPath"
  return $diffPath
}

function Apply-TaskDiffIfAllowed {
  param(
    [string]$CodexCmd,
    [string]$TaskId,
    [string]$ArtifactsDir,
    [string]$Goal
  )

  $isNoop = ($Goal -match "NOOP") -or ($Goal -match "コード変更禁止")
  if ($isNoop) {
    Write-Host "[AI_LOOP] NOOP detected; skipping apply."
    return $false
  }

  $applyLog = Join-Path $ArtifactsDir ("codex_apply_{0}.log" -f $TaskId)
  Write-Host "[AI_LOOP] applying task diff..."
  $applyText = CmdUtf8 "$CodexCmd cloud apply $TaskId"
  Save-Text -Path $applyLog -Text $applyText
  Write-Host "[AI_LOOP] apply log: $applyLog"
  return $true
}

function Safe-Run($label, $cmd) {
  Write-Host "[AI_LOOP] $label"
  $out = cmd /c $cmd 2>&1
  $out | ForEach-Object { $_ }
  return ($out -join "`r`n")
}

# ---- main ----
Require-Cmd $CodexCmd

$RepoRoot = Get-RepoRoot
Set-Location $RepoRoot

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$Artifacts = Join-Path $RepoRoot "artifacts"
New-Item -ItemType Directory -Force -Path $Artifacts | Out-Null

$taskFile   = Join-Path $Artifacts "codex_task_$ts.txt"
$codexLog   = Join-Path $Artifacts "codex_out_$ts.log"
$testLog    = Join-Path $Artifacts "tests_out_$ts.log"
$reportPath = Join-Path $Artifacts "ai_loop_report_$ts.txt"

# EnvId 決定：手動 > artifacts > config
if (-not $EnvId) {
  $EnvId = Get-CodexEnvIdFromFile -RepoRoot $RepoRoot
  if ($EnvId) { Write-Host "[AI_LOOP] Loaded EnvId from artifacts\.codex_env_id.txt : $EnvId" }
}
if (-not $EnvId) {
  $cfgPath = Get-DefaultCodexConfigPath
  $EnvId = Get-CodexEnvIdFromConfig -ConfigPath $cfgPath
  if ($EnvId) { Write-Host "[AI_LOOP] Detected EnvId from config: $EnvId" }
}
if (-not $EnvId) {
  Write-Error "Could not auto-detect Codex Cloud ENV_ID."
  exit 11
}

# Branch 決定：手動指定 > origin/HEAD > current > main
if (-not $Branch) {
  $Branch = Get-RemoteDefaultBranch
}
Write-Host "[AI_LOOP] RepoRoot: $RepoRoot"
Write-Host "[AI_LOOP] EnvId:    $EnvId"
Write-Host "[AI_LOOP] Branch:   $Branch"

$task = @"
Repo: $RepoRoot

GOAL:
$Goal

Hard constraints:
- Keep existing tests passing.
- Prefer minimal diffs.
- Do not change unrelated files.
- If GOAL says "NOOP" or "コード変更禁止", DO NOT modify any files.
- After modifications, run:
  $TestCmd

Deliver:
- What changed (files + brief reasoning)
- How to verify (commands)
"@
$task | Out-File -Encoding utf8 $taskFile
Write-Host "[AI_LOOP] Task saved: $taskFile"

# Codex Cloud exec（--env と --branch を必ず渡す）
Write-Host "[AI_LOOP] Running Codex cloud exec..."
$execText = ""

try {
  $execLines = (& $CodexCmd cloud exec --env $EnvId --branch $Branch -- "$task" 2>&1)
  $execLines | Tee-Object -FilePath $codexLog | Out-Host
  $execText = ($execLines -join "`n")
} catch {
  Write-Error "Codex cloud exec failed: $($_.Exception.Message)"
  Write-Error "Codex log: $codexLog"
  exit 10
}

$ti = Extract-TaskInfo -Text $execText
$taskId = $ti.taskId
$taskUrl = $ti.taskUrl

$diffPath = $null
$didApply = $false

if ($taskId) {
  Write-Host "[AI_LOOP] Codex task id: $taskId"
  if ($taskUrl) { Write-Host "[AI_LOOP] Codex task url: $taskUrl" }

  Wait-TaskReady -CodexCmd $CodexCmd -TaskId $taskId -WaitMinutes $WaitMinutes -PollSeconds $PollSeconds -ArtifactsDir $Artifacts -TaskUrl $taskUrl

  $diffPath = Save-TaskDiff -CodexCmd $CodexCmd -TaskId $taskId -ArtifactsDir $Artifacts
  $didApply = Apply-TaskDiffIfAllowed -CodexCmd $CodexCmd -TaskId $taskId -ArtifactsDir $Artifacts -Goal $Goal
} else {
  Write-Warning "Could not extract task id; skipping status/diff/apply automation."
}

# テスト実行（ローカル）
Write-Host "[AI_LOOP] Running tests..."
$testOutText = ""
try {
  $testOutText = Safe-Run "TestCmd => $TestCmd" $TestCmd
  $testOutText | Out-File -Encoding utf8 $testLog
} catch {
  $testOutText = $_.Exception.Message
  $testOutText | Out-File -Encoding utf8 $testLog
}

# git diff（任意）
$gitDiffText = ""
if ($IncludeGitDiff) {
  if (Get-Command git -ErrorAction SilentlyContinue) {
    try {
      $gitDiffText = (git diff --stat) + "`r`n`r`n" + (git diff)
    } catch {
      $gitDiffText = "[WARN] git diff failed: $($_.Exception.Message)"
    }
  } else {
    $gitDiffText = "[WARN] git not found; skipping git diff."
  }
}

# レポート
$codexOutText = ""
try { $codexOutText = Get-Content -Raw -Encoding utf8 $codexLog } catch { $codexOutText = "[WARN] Could not read $codexLog" }

$report = @"
=== AI LOOP REPORT ($ts) ===
Repo: $RepoRoot
Env:  $EnvId
Branch: $Branch

GOAL:
$Goal

Task:
- id:      $taskId
- url:     $taskUrl
- diff:    $diffPath
- applied: $didApply

--- Codex output (log) ---
$codexOutText

--- Test output (log) ---
$testOutText

--- Git diff (optional) ---
$gitDiffText

Files:
- Task:   $taskFile
- Codex:  $codexLog
- Tests:  $testLog
- Report: $reportPath
"@

$report | Out-File -Encoding utf8 $reportPath

try {
  Set-Clipboard -Value $report
  Write-Host "[AI_LOOP] Report copied to clipboard."
} catch {
  Write-Warning "Set-Clipboard failed. Report saved at: $reportPath"
}

Write-Host "[AI_LOOP] Report saved: $reportPath"
Write-Host "[AI_LOOP] DONE"