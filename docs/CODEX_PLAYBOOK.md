\# Codex Playbook (KJACai)



You are working in repo: C:\\Users\\Futamura\\KJACai



\## Goal

\- Implement the next small improvement I describe.

\- Keep changes minimal and safe.

\- Do not touch legacy/sandbox experiments unless explicitly requested.



\## Rules (non-negotiable)

1\) Before editing, list the exact files you plan to change and why.

2\) Make the smallest change that solves the goal.

3\) After changes, run:

&nbsp;  - powershell -ExecutionPolicy Bypass -File .\\scripts\\check.ps1

4\) If check fails, fix and re-run until it passes.

5\) Then show:

&nbsp;  - git diff --stat

&nbsp;  - a short diff summary (what changed)

&nbsp;  - suggested commit message

6\) Never add secrets or local notes. If anything looks like a key/token, remove it and add to .gitignore.



\## Canonical entrypoints

\- Pipeline:

&nbsp; - python scripts/run\_pipeline.py

\- Report:

&nbsp; - python scripts/run\_report.py

\- Check:

&nbsp; - powershell -ExecutionPolicy Bypass -File scripts/check.ps1



\## Default policy

\- Prefer editing ONLY:

&nbsp; - src/ (core logic)

&nbsp; - scripts/run\_\*.py (entrypoints)

&nbsp; - scripts/check.ps1 (validation)

&nbsp; - docs/ (documentation)

&nbsp; - .github/workflows (CI)

\- Avoid:

&nbsp; - sandbox/ unless explicitly asked



\## Task template (paste a task below)

\[TASK HERE]

