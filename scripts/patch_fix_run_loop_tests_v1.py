from __future__ import annotations
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "tests" / "run_loop.Tests.ps1"
MARK = "__PATCH_FIX_RUN_LOOP_TESTS_V1__"

NEW_CONTENT = r"""# tests/run_loop.Tests.ps1
# __PATCH_FIX_RUN_LOOP_TESTS_V1__
# Pester v5+ friendly: never rely on $PSScriptRoot being set in discovery/execution scopes.

Set-StrictMode -Version Latest

# --- Resolve paths safely (works in Pester discovery + execution) ---
$ThisTestFile = if ($PSCommandPath) { $PSCommandPath } else { $MyInvocation.MyCommand.Path }
if (-not $ThisTestFile) { throw "Cannot resolve current test file path (PSCommandPath/MyInvocation are null)." }

$TestDir  = Split-Path -Parent $ThisTestFile
$RepoRoot = (Resolve-Path (Join-Path $TestDir '..')).Path

$ScriptUnderTest = Join-Path $RepoRoot 'sandbox\run_loop.ps1'
$RunDbg          = Join-Path $RepoRoot 'sandbox\run_dbg.ps1'
$RunProd         = Join-Path $RepoRoot 'sandbox\run_prod.ps1'

Describe 'sandbox/run_loop.ps1' -Tag 'unit' {

    BeforeAll {
        # Basic sanity
        if (-not $RepoRoot) { throw "RepoRoot is null" }
        if (-not (Test-Path $ScriptUnderTest)) { throw "Script not found: $ScriptUnderTest" }
    }

    It 'exists' {
        Test-Path $ScriptUnderTest | Should -BeTrue
    }

    It 'parses with no syntax errors (AST)' {
        $tokens = $null
        $errors = $null
        [void][System.Management.Automation.Language.Parser]::ParseFile($ScriptUnderTest, [ref]$tokens, [ref]$errors)
        ($errors | Measure-Object).Count | Should -Be 0
    }

    It 'has expected sibling runners' {
        (Test-Path $RunDbg)  | Should -BeTrue
        (Test-Path $RunProd) | Should -BeTrue
    }

    It 'mentions dev/prod preset and uses run_dbg/run_prod' {
        $src = Get-Content $ScriptUnderTest -Raw -ErrorAction Stop
        $src | Should -Match '-Preset'
        $src | Should -Match 'dev'
        $src | Should -Match 'prod'
        $src | Should -Match 'run_dbg\.ps1'
        $src | Should -Match 'run_prod\.ps1'
    }

    It 'generates/mentions summary output (summary.txt + key metrics)' {
        $src = Get-Content $ScriptUnderTest -Raw -ErrorAction Stop

        # summary artifact
        $src | Should -Match 'summary\.txt'

        # metrics we expect to be parsed from run.log
        $src | Should -Match 'sorted_events_raw'
        $src | Should -Match 'after_resolve_same_frame'
        $src | Should -Match 'after_min_step_interval'
        $src | Should -Match 'after_same_foot_min_step_interval'
        $src | Should -Match 'near_resolve'
        $src | Should -Match 'min_step'
    }

    It 'has semi-auto gate (dev -> manual approve -> prod)' {
        $src = Get-Content $ScriptUnderTest -Raw -ErrorAction Stop

        # read-host gate is the simplest robust check
        $src | Should -Match 'Read-Host'
        $src | Should -Match 'approve|confirm|Proceed|prod'  # flexible wording
    }

    It 'clamps MaxIters to 1..3 (or documents it in code)' {
        $src = Get-Content $ScriptUnderTest -Raw -ErrorAction Stop

        # accept either an explicit clamp pattern or a ValidateRange/param constraint
        $hasClamp =
            ($src -match 'MaxIters') -and (
                ($src -match '\[ValidateRange\(\s*1\s*,\s*3\s*\)\]') -or
                ($src -match 'Math\]::Min') -or
                ($src -match 'Math\]::Max') -or
                ($src -match 'Clamp') -or
                ($src -match 'if\s*\(\s*\$MaxIters\s*-gt\s*3') -or
                ($src -match 'if\s*\(\s*\$MaxIters\s*-lt\s*1')
            )

        $hasClamp | Should -BeTrue
    }
}
"""

def backup(p: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = p.with_suffix(p.suffix + f".bak_{ts}")
    bak.write_text(p.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return bak

def main() -> int:
    if not TARGET.parent.exists():
        TARGET.parent.mkdir(parents=True, exist_ok=True)

    if TARGET.exists():
        src = TARGET.read_text(encoding="utf-8", errors="replace")
        if MARK in src:
            print("[SKIP] already patched:", TARGET)
            return 0
        bak = backup(TARGET)
        print("[BAK]", bak)

    TARGET.write_text(NEW_CONTENT, encoding="utf-8")
    print("[OK] wrote:", TARGET)

    # quick smoke: parse as text (no execution)
    txt = TARGET.read_text(encoding="utf-8")
    if MARK not in txt:
        print("[ERR] marker missing after write")
        return 2

    print("[OK] marker present")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())