# KJACai

KJACai is an athletics performance analysis and reporting system (sprint / relay).
This repository contains canonical entrypoints, reporting tools, and (optional) GUI utilities.

---

## ✅ Canonical entrypoints

### Pipeline (canonical)
```bash
python scripts/run_pipeline.py

Note: The current legacy pipeline implementation (scripts/kjac_pipeline_v1_3.py) expects MP4 videos under videos/.
If videos/ has no MP4 files, it will stop with: “videos フォルダに MP4 が見つかりません。”

Run with extra args (quoted paths supported):

python scripts/run_pipeline.py all --args "--csv videos/race_PLUS.csv --out output/pipeline.pdf"
Report (canonical)
python scripts/run_report.py --args "--help"
🧪 Local checks

Run compile checks:

powershell -ExecutionPolicy Bypass -File scripts/check.ps1
📦 Requirements (split “extras”)

Install minimal (CI / light):

pip install -r requirements/base.txt

AI features:

pip install -r requirements/ai.txt

GUI features:

pip install -r requirements/gui.txt

Full install:

pip install -r requirements/full.txt
🤖 Codex workflow

See CODEX.md for the canonical “paste-and-loop” workflow.


## 3) commit & push
README保存したら：

```powershell
cd C:\Users\Futamura\KJACai
git add README.md
git commit -m "docs: add README with canonical usage"
git push