Set-StrictMode -Version Latest

Describe 'run_loop.ps1 contract' -Tag 'unit' {
    BeforeAll {
        $thisFile = if ($PSCommandPath) { $PSCommandPath } else { $MyInvocation.MyCommand.Path }
        if (-not $thisFile) {
            $fallback = Resolve-Path '.\tests\run_loop.Tests.ps1' -ErrorAction SilentlyContinue
            if ($fallback) { $thisFile = $fallback.Path }
        }
        if (-not $thisFile) {
            throw "Could not resolve test file path from PSCommandPath/MyInvocation."
        }

        $TestDir = Split-Path -Parent $thisFile
        $RepoRoot = Resolve-Path (Join-Path $TestDir '..')
        if ($RepoRoot -is [System.Management.Automation.PathInfo]) {
            $RepoRoot = $RepoRoot.Path
        }

        $script:RepoRoot = $RepoRoot
        $script:ScriptUnderTest = Join-Path $script:RepoRoot 'sandbox\run_loop.ps1'
        $script:RunDbg = Join-Path $script:RepoRoot 'sandbox\run_dbg.ps1'
        $script:RunProd = Join-Path $script:RepoRoot 'sandbox\run_prod.ps1'
        if (-not $script:ScriptUnderTest) { throw "ScriptUnderTest path is null." }
        $script:Source = Get-Content -LiteralPath $script:ScriptUnderTest -Raw -ErrorAction Stop
    }

    It 'has required scripts present' {
        (Test-Path -LiteralPath $script:ScriptUnderTest) | Should -BeTrue
        (Test-Path -LiteralPath $script:RunDbg) | Should -BeTrue
        (Test-Path -LiteralPath $script:RunProd) | Should -BeTrue
    }

    It 'parses with zero syntax errors' {
        $tokens = $null
        $errors = $null
        [void][System.Management.Automation.Language.Parser]::ParseFile($script:ScriptUnderTest, [ref]$tokens, [ref]$errors)
        @($errors).Count | Should -Be 0
    }

    It 'references dev/prod preset and calls run_dbg/run_prod' {
        $script:Source | Should -Match '\[ValidateSet\("dev",\s*"prod"\)\]'
        $script:Source | Should -Match 'run_dbg\.ps1'
        $script:Source | Should -Match 'run_prod\.ps1'
    }

    It 'mentions summary.txt and stage keys' {
        $script:Source | Should -Match 'summary\.txt'
        $script:Source | Should -Match 'sorted_events_raw'
        $script:Source | Should -Match 'after_resolve_same_frame'
        $script:Source | Should -Match 'after_min_step_interval'
        $script:Source | Should -Match 'after_same_foot_min_step_interval'
        $script:Source | Should -Match 'near_resolve'
        $script:Source | Should -Match 'min_step_dropped_total'
    }

    It 'has semi-auto gate via Read-Host' {
        $script:Source | Should -Match 'Read-Host'
    }

    It 'clamps MaxIters to 1..3' {
        $hasValidateRange = $script:Source -match '\[ValidateRange\(\s*1\s*,\s*3\s*\)\].*\$MaxIters'
        $hasClampMath = ($script:Source -match '\[Math\]::Max\(\s*1') -and ($script:Source -match '\[Math\]::Min\(\s*3')
        ($hasValidateRange -or $hasClampMath) | Should -BeTrue
    }
}

Describe 'run_loop.ps1 mockable behavior' -Tag 'unit' {
    BeforeAll {
        $thisFile = if ($PSCommandPath) { $PSCommandPath } else { $MyInvocation.MyCommand.Path }
        if (-not $thisFile) {
            $fallback = Resolve-Path '.\tests\run_loop.Tests.ps1' -ErrorAction SilentlyContinue
            if ($fallback) { $thisFile = $fallback.Path }
        }
        if (-not $thisFile) {
            throw "Could not resolve test file path from PSCommandPath/MyInvocation."
        }

        $TestDir = Split-Path -Parent $thisFile
        $RepoRoot = Resolve-Path (Join-Path $TestDir '..')
        if ($RepoRoot -is [System.Management.Automation.PathInfo]) {
            $RepoRoot = $RepoRoot.Path
        }
        $script:ScriptUnderTest2 = Join-Path $RepoRoot 'sandbox\run_loop.ps1'

        # dot-source to load functions only (script should not auto-run when dot-sourced)
        . $script:ScriptUnderTest2 -Video 'dummy.mp4'
    }

    It 'exposes Invoke-RunLoop when dot-sourced' {
        (Get-Command Invoke-RunLoop -ErrorAction SilentlyContinue) | Should -Not -BeNullOrEmpty
    }

    It 'dev path calls Invoke-RunDbg once and defaults MaxFrames=200; does not call Invoke-RunProd when Read-Host is n' {
        $fakeRepo = Join-Path $TestDrive 'repo'
        $fakeSandbox = Join-Path $fakeRepo 'sandbox'
        New-Item -ItemType Directory -Path $fakeSandbox -Force | Out-Null
        New-Item -ItemType File -Path (Join-Path $fakeSandbox 'run_dbg.ps1') -Force | Out-Null
        New-Item -ItemType File -Path (Join-Path $fakeSandbox 'run_prod.ps1') -Force | Out-Null

        Mock Get-RepoRoot { $fakeRepo }
        Mock Read-Host { 'n' }

        $script:DbgCalls = 0
        $script:ProdCalls = 0
        $script:DbgSawMax200 = $false
        Set-Item -Path Function:Invoke-RunDbg -Value {
            param($RunnerPath, [Alias('Args')]$RunnerArgs)
            $script:DbgCalls++
            $argText = @($RunnerArgs) -join ' '
            if ($argText -match 'MaxFrames' -and $argText -match '200') {
                $script:DbgSawMax200 = $true
            }
            return 0
        }
        Set-Item -Path Function:Invoke-RunProd -Value {
            param($RunnerPath, [Alias('Args')]$RunnerArgs)
            $script:ProdCalls++
            return 0
        }

        $null = Invoke-RunLoop -Video 'dummy.mp4' -Preset dev -MaxIters 1 -NoWriteSummary

        $script:DbgCalls | Should -Be 1
        $script:DbgSawMax200 | Should -BeTrue
        $script:ProdCalls | Should -Be 0
    }

    It 'semi-auto gate: Read-Host=y triggers Invoke-RunProd once' {
        $fakeRepo = Join-Path $TestDrive 'repo_y'
        $fakeSandbox = Join-Path $fakeRepo 'sandbox'
        New-Item -ItemType Directory -Path $fakeSandbox -Force | Out-Null
        New-Item -ItemType File -Path (Join-Path $fakeSandbox 'run_dbg.ps1') -Force | Out-Null
        New-Item -ItemType File -Path (Join-Path $fakeSandbox 'run_prod.ps1') -Force | Out-Null

        Mock Get-RepoRoot { $fakeRepo }
        Mock Read-Host { 'y' }

        $script:DbgCalls2 = 0
        $script:ProdCalls2 = 0
        Set-Item -Path Function:Invoke-RunDbg -Value {
            param($RunnerPath, [Alias('Args')]$RunnerArgs)
            $script:DbgCalls2++
            return 0
        }
        Set-Item -Path Function:Invoke-RunProd -Value {
            param($RunnerPath, [Alias('Args')]$RunnerArgs)
            $script:ProdCalls2++
            return 0
        }

        $null = Invoke-RunLoop -Video 'dummy.mp4' -Preset dev -MaxIters 1 -NoWriteSummary

        $script:DbgCalls2 | Should -Be 1
        $script:ProdCalls2 | Should -Be 1
    }

    It 'Parse-RunLog and Write-Summary can be unit-tested with sample log text' {
        $sample = @'
[INFO] Resolution=1920x1080, FPS=59.94, Frames=36602
[DBG] stage=sorted_events_raw n=720
[DBG] stage=after_resolve_same_frame n=623
[DBG] stage=after_min_step_interval n=623
[DBG] stage=after_same_foot_min_step_interval n=623
[DBG] near_resolve: merged_count=42
[DBG] min_step_delta_frames_hist_1to10: 1:0 2:1 3:2 4:0 5:0 6:0 7:0 8:0 9:0 10:0
'@
        $parsed = Parse-RunLog -LogText $sample
        $parsed.Keys | Should -Contain 'stage_sorted_events_raw'
        $parsed.Keys | Should -Contain 'stage_after_min_step_interval'
        $parsed.Keys | Should -Contain 'near_resolve_merged_count'
        $parsed.Keys | Should -Contain 'min_step_dropped_total'

        $runDir = Join-Path $TestDrive 'run_u'
        New-Item -ItemType Directory -Path $runDir -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $runDir 'run.log') -Value $sample -Encoding UTF8
        $summary = Write-Summary -RunDir $runDir -Phase 'dev'
        (Test-Path -LiteralPath $summary) | Should -BeTrue
        $summaryText = Get-Content -LiteralPath $summary -Raw
        $summaryText | Should -Match 'stage_sorted_events_raw='
        $summaryText | Should -Match 'stage_after_min_step_interval='
        $summaryText | Should -Match 'near_resolve_merged_count='
    }
}
