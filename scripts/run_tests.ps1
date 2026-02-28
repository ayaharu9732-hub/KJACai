param(
    [string]$ResultPath
)

$ErrorActionPreference = "Stop"

${This} = if ($PSCommandPath) { $PSCommandPath } else { $MyInvocation.MyCommand.Path }
if (-not ${This}) {
    Write-Error "Could not resolve script path from PSCommandPath/MyInvocation."
    exit 2
}

$ScriptsDir = Split-Path -Parent ${This}
try {
    $RepoRoot = Resolve-Path (Join-Path $ScriptsDir '..')
    if ($RepoRoot -is [System.Management.Automation.PathInfo]) {
        $RepoRoot = $RepoRoot.Path
    }
} catch {
    Write-Error "Failed to resolve repo root from script path '$This'. Expected layout: <repo>\\scripts\\run_tests.ps1"
    exit 2
}

$TestsPath = Join-Path $RepoRoot 'tests'

Write-Host "[TEST] Running Pester 5.7.1 with TestRegistry disabled"
Write-Host "[TEST] Tests path: $TestsPath"

try {
    Remove-Module Pester -ErrorAction SilentlyContinue
    Import-Module Pester -RequiredVersion 5.7.1 -Force
} catch {
    Write-Error "Failed to load Pester 5.7.1: $($_.Exception.Message)"
    exit 2
}

function New-BasePesterConfig {
    $c = New-PesterConfiguration
    $c.Run.Path = $TestsPath
    $c.Run.PassThru = $true
    $c.Output.Verbosity = "Detailed"
    $c.TestRegistry.Enabled = $false
    return $c
}

$result = $null
$firstErr = $null
$hadResultExportAttempt = $false

if ($ResultPath) {
    if (-not [System.IO.Path]::IsPathRooted($ResultPath)) {
        $ResultPath = Join-Path $RepoRoot $ResultPath
    }
    $resultDir = Split-Path -Parent $ResultPath
    if ($resultDir) {
        New-Item -ItemType Directory -Path $resultDir -Force | Out-Null
    }
    Write-Host "[TEST] Result path: $ResultPath"

    $cfg = New-BasePesterConfig
    $cfg.TestResult.Enabled = $true
    $cfg.TestResult.OutputPath = $ResultPath
    $cfg.TestResult.OutputFormat = "NUnitXml"
    $hadResultExportAttempt = $true

    try {
        $result = Invoke-Pester -Configuration $cfg
    } catch {
        $firstErr = $_.Exception.Message
    }

    if ($null -eq $result) {
        if (-not $firstErr) { $firstErr = "No result object returned." }
        Write-Warning "Test result export failed. Retrying without XML output. Reason: $firstErr"

        $cfgRetry = New-BasePesterConfig
        try {
            $result = Invoke-Pester -Configuration $cfgRetry
        } catch {
            Write-Error "Invoke-Pester failed in both attempts. Last error: $($_.Exception.Message)"
            exit 3
        }
    }
} else {
    $cfg = New-BasePesterConfig
    try {
        $result = Invoke-Pester -Configuration $cfg
    } catch {
        Write-Error "Invoke-Pester failed: $($_.Exception.Message)"
        exit 3
    }
}

if ($null -eq $result) {
    if ($hadResultExportAttempt) {
        Write-Error "No result object returned in both attempts."
    } else {
        Write-Error "No result object returned."
    }
    exit 4
}

if ($result.FailedCount -gt 0 -or $result.Result -ne "Passed") {
    Write-Host "[TEST] FailedCount=$($result.FailedCount) PassedCount=$($result.PassedCount)"
    exit 1
}

Write-Host "[TEST] PassedCount=$($result.PassedCount) FailedCount=$($result.FailedCount)"
exit 0
