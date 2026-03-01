# KJACai

KJACai is an athletics performance analysis and reporting system for sprint and relay athletes.

It provides video analysis pipelines, metric extraction, AI-assisted reporting, and GUI tools used in the KJAC AI project.

---

## 🚀 Canonical Entry Points

### Pipeline (MAIN ENTRYPOINT)

Run the full analysis pipeline:

```bash
python scripts/run_pipeline.py

This is the official execution path for analysis.

Report Generation

Generate a report from existing metrics:

python scripts/run_report.py
📁 Project Structure
src/        Core analysis logic
scripts/    CLI entrypoints and utilities
videos/     Input videos
outputs/    Generated analysis results
sandbox/    Experimental / legacy scripts
⚙️ Requirements

Install dependencies:

pip install -r requirements.txt
✅ Development Check

Run repository validation:

powershell -ExecutionPolicy Bypass -File scripts/check.ps1
🤖 CI

GitHub Actions automatically runs checks on push and pull requests.

📌 Status

Active development — canonical pipeline structure established.


---

# ✅ これを書く理由（超重要）

実は今あなたの repo はもう：

✅ Codex運用可能  
✅ CI動作  
✅ canonical entrypoint確立  
✅ runtime_paths統一  

＝ **個人スクリプト → 開発プロジェクト化 完了**

README はその「宣言書」です。

---

# ✅ 次にやる（30秒）

```powershell
notepad README.md