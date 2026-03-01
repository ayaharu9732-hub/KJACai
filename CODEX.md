\# KJACai Codex Operating Guide (Canonical)



This repo is \*\*KJACai\*\*, an athletics performance analysis/report system.

We prioritize \*\*minimal safe changes\*\*, keeping legacy/sandbox intact unless explicitly requested.



\## Canonical entrypoints

\- Pipeline: `python scripts/run\_pipeline.py \[mode] --args "..."`

\- Report:   `python scripts/run\_report.py --args "..."`

\- Health check: `powershell -ExecutionPolicy Bypass -File scripts/check.ps1`



\## Rules (must follow)

1\) Before editing: list exact files to change and why.

2\) Keep changes minimal. Do not touch `sandbox/` or legacy experiment scripts unless explicitly asked.

3\) After changes run:

&nbsp;  - `powershell -ExecutionPolicy Bypass -File .\\scripts\\check.ps1`

&nbsp;  - `python -m pytest -q` (if tests exist)

4\) If failing, fix and re-run until passing.

5\) Show:

&nbsp;  - `git diff --stat`

&nbsp;  - suggested commit message



\## Standard task template (paste below)

You are working in repo `C:\\Users\\Futamura\\KJACai`.



Goal:

\- Implement the next small improvement I describe.

\- Keep changes minimal and safe.



Now do this task:

\[TASK HERE]

