import sys
import os
import json
import traceback
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QCheckBox,
    QPlainTextEdit,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QFrame,
    QStatusBar,
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QThread, Signal


CONFIG_PATH = "pose_gui_config_video.json"

DEFAULT_CONFIG = {
    "video_path": "",
    "overlay": True,
    "pdf": True,
    "ai": True,
}


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = DEFAULT_CONFIG.copy()
            cfg.update(data)
            return cfg
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ================= バックエンド =================

class AnalyzerBackend:
    """動画1本 → metrics → PDF まで一括実行"""

    METRICS_SCRIPT = r"C:\Users\Futamura\KJACai\scripts\pose_metrics_analyzer_v3_4.py"
    METRICS_CSV = r"C:\Users\Futamura\KJACai\outputs\koike_pose_metrics_v3.csv"

    def run_analysis(self, video_path, options, progress_callback=None):
        import subprocess

        def p(msg: str):
            if progress_callback:
                progress_callback(msg)

        if not video_path or not os.path.isfile(video_path):
            return False, "有効な動画ファイルが指定されていません。"

        if not video_path.lower().endswith(".mp4"):
            return False, "mp4 形式の動画ファイルを選択してください。"

        video_name = os.path.basename(video_path)
        athlete = os.path.splitext(video_name)[0]
        p(f"動画ファイル: {video_path}")
        p(f"選手名(仮): {athlete}")

        # ---- メトリクス CSV 自動生成 ----
        p("pose_metrics_analyzer_v3_4.py によるメトリクス解析を開始します。")
        if not os.path.isfile(self.METRICS_SCRIPT):
            return False, f"メトリクス解析スクリプトが見つかりません: {self.METRICS_SCRIPT}"

        cmd = ["python", self.METRICS_SCRIPT, "--video", video_path]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            return False, f"pose_metrics_analyzer_v3_4.py の実行に失敗しました: {e}"

        if not os.path.exists(self.METRICS_CSV):
            return False, f"メトリクスCSVが見つかりませんでした: {self.METRICS_CSV}"

        p(f"メトリクスCSVを使用: {self.METRICS_CSV}")

        # ---- フォームレポート生成 ----
        try:
            import pose_reporter_pdf_ai_v5_5_3 as core
        except Exception as e:
            return False, f"pose_reporter_pdf_ai_v5_5_3 の読み込みに失敗しました: {e}"

        video_id = core.get_video_id(video_path)
        out_dirs = core.get_root_output_dirs(athlete, video_id)

        log_name = f"{core.VERSION_STR}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        core.LOG_FILE = os.path.join(out_dirs["logs"], log_name)
        p(f"ログファイル: {core.LOG_FILE}")

        try:
            p("PDF レポート生成を開始します。")
            core.build_pdf(self.METRICS_CSV, video_path, athlete, out_dirs)
        except Exception as e:
            return False, f"build_pdf 実行中にエラーが発生しました: {e}"

        pdf_path = os.path.join(
            out_dirs["pdf"],
            f"{athlete}_form_report_{core.VERSION_STR}.pdf"
        )
        p(f"PDF レポート生成が完了しました: {pdf_path}")
        return True, pdf_path


# ================== Worker ==================

class Worker(QThread):
    progress = Signal(str)
    finished = Signal(bool, str)
    error = Signal(str)

    def __init__(self, backend: AnalyzerBackend, video_path, options, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.video_path = video_path
        self.options = options

    def run(self):
        try:
            self.progress.emit("解析を開始します...")
            ok, msg = self.backend.run_analysis(
                self.video_path,
                self.options,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(ok, msg)
        except Exception:
            tb = traceback.format_exc()
            self.error.emit(tb)


# ================== GUI ==================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("KJAC フォーム解析 GUI (動画選択版)")
        self.resize(1100, 700)

        self.backend = AnalyzerBackend()
        self.current_worker: Worker | None = None

        self._setup_ui()
        self._load_initial_config()

    def _setup_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)

        top_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        left_layout.addWidget(self._create_io_group())
        left_layout.addWidget(self._create_option_group())
        left_layout.addLayout(self._create_buttons_row())

        right_layout = QVBoxLayout()
        right_layout.addWidget(self._create_preview_group())

        top_layout.addLayout(left_layout, 2)
        top_layout.addLayout(right_layout, 3)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(self._create_log_group(), 3)

        status = QStatusBar()
        self.setStatusBar(status)

        self.setCentralWidget(central)

    def _create_io_group(self) -> QGroupBox:
        group = QGroupBox("入出力設定")

        form = QFormLayout()

        self.le_video = QLineEdit()
        btn_video = QPushButton("参照...")
        btn_video.clicked.connect(self._select_video)
        h_video = QHBoxLayout()
        h_video.addWidget(self.le_video)
        h_video.addWidget(btn_video)
        form.addRow("動画ファイル (mp4)", h_video)

        group.setLayout(form)
        return group

    def _create_option_group(self) -> QGroupBox:
        group = QGroupBox("オプション")

        v = QVBoxLayout()

        self.cb_overlay = QCheckBox("overlay 画像を出力（pose_reporter 標準）")
        self.cb_overlay.setChecked(True)

        self.cb_pdf = QCheckBox("フォームレポート PDF を出力")
        self.cb_pdf.setChecked(True)

        self.cb_ai = QCheckBox("AI コメントを生成")
        self.cb_ai.setChecked(True)

        v.addWidget(self.cb_overlay)
        v.addWidget(self.cb_pdf)
        v.addWidget(self.cb_ai)

        group.setLayout(v)
        return group

    def _create_buttons_row(self) -> QHBoxLayout:
        h = QHBoxLayout()

        self.btn_run = QPushButton("解析開始")
        self.btn_run.clicked.connect(self._on_run_clicked)

        self.btn_preview = QPushButton("プレビュー更新")
        self.btn_preview.clicked.connect(self._on_preview_clicked)

        self.btn_save = QPushButton("設定保存")
        self.btn_save.clicked.connect(self._on_save_clicked)

        h.addWidget(self.btn_run)
        h.addWidget(self.btn_preview)
        h.addWidget(self.btn_save)
        return h

    def _create_preview_group(self) -> QGroupBox:
        group = QGroupBox("プレビュー（outputs 以下の最新 PNG）")

        v = QVBoxLayout()

        self.lbl_preview = QLabel("プレビュー画像はここに表示されます")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setFrameShape(QFrame.Box)
        self.lbl_preview.setMinimumSize(400, 300)

        v.addWidget(self.lbl_preview)
        group.setLayout(v)
        return group

    def _create_log_group(self) -> QGroupBox:
        group = QGroupBox("ログ")

        v = QVBoxLayout()

        self.te_log = QPlainTextEdit()
        self.te_log.setReadOnly(True)

        v.addWidget(self.te_log)
        group.setLayout(v)
        return group

    # ---------- 設定 ----------

    def _load_initial_config(self):
        cfg = load_config()
        self.le_video.setText(cfg.get("video_path", ""))
        self.cb_overlay.setChecked(cfg.get("overlay", True))
        self.cb_pdf.setChecked(cfg.get("pdf", True))
        self.cb_ai.setChecked(cfg.get("ai", True))
        self.log("設定を読み込みました。")

    def _collect_config(self) -> dict:
        return {
            "video_path": self.le_video.text().strip(),
            "overlay": self.cb_overlay.isChecked(),
            "pdf": self.cb_pdf.isChecked(),
            "ai": self.cb_ai.isChecked(),
        }

    def _save_config(self):
        cfg = self._collect_config()
        save_config(cfg)
        self.log("設定を保存しました。")

    # ---------- 共通 ----------

    def log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.te_log.appendPlainText(f"[{ts}] {message}")
        sb = self.te_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _build_options(self) -> dict:
        return {
            "overlay": self.cb_overlay.isChecked(),
            "pdf": self.cb_pdf.isChecked(),
            "ai": self.cb_ai.isChecked(),
        }

    # ---------- ファイル選択 ----------

    def _select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "動画ファイルを選択",
            "",
            "Video Files (*.mp4);;All Files (*)",
        )
        if path:
            self.le_video.setText(path)
            self.log(f"動画ファイルを選択: {path}")

    # ---------- ボタン ----------

    def _on_run_clicked(self):
        if not self._validate_inputs():
            return
        video_path = self.le_video.text().strip()
        options = self._build_options()
        started = self._start_worker(video_path, options, show_busy_dialog=True)
        if started:
            self.log("解析ジョブを開始しました。")

    def _on_preview_clicked(self):
        self._update_preview()

    def _on_save_clicked(self):
        self._save_config()
        QMessageBox.information(self, "情報", "設定を保存しました。")

    # ---------- Worker 起動 ----------

    def _start_worker(self, video_path, options, show_busy_dialog: bool) -> bool:
        if self.current_worker is not None and self.current_worker.isRunning():
            if show_busy_dialog:
                QMessageBox.information(self, "処理中", "現在解析中です。完了をお待ちください。")
            return False

        self.current_worker = Worker(self.backend, video_path, options)
        self.current_worker.progress.connect(self.log)
        self.current_worker.finished.connect(self._on_worker_finished)
        self.current_worker.error.connect(self._on_worker_error)
        self.current_worker.start()
        return True

    def _on_worker_finished(self, ok: bool, message: str):
        if ok:
            self.log(f"解析完了: {message}")
            self.statusBar().showMessage("解析が完了しました。", 5000)
        else:
            self.log(f"解析失敗: {message}")
            QMessageBox.warning(self, "エラー", message)
        self._update_preview(auto=True)

    def _on_worker_error(self, traceback_str: str):
        self.log("解析中に例外が発生しました。詳細はコンソールを確認してください。")
        print(traceback_str, file=sys.stderr)
        QMessageBox.critical(self, "致命的エラー", "解析中に例外が発生しました。")

    # ---------- プレビュー ----------

    def _update_preview(self, auto: bool = False):
        outputs_root = "outputs"
        if not os.path.isdir(outputs_root):
            if not auto:
                QMessageBox.information(self, "情報", "outputs フォルダが見つかりません。")
            return

        latest_image = self._find_latest_image(outputs_root)
        if not latest_image:
            if not auto:
                QMessageBox.information(self, "情報", "outputs 内に画像ファイルが見つかりませんでした。")
            return

        pixmap = QPixmap(latest_image)
        if pixmap.isNull():
            if not auto:
                QMessageBox.warning(self, "エラー", "画像の読み込みに失敗しました。")
            return

        self.lbl_preview.setPixmap(
            pixmap.scaled(
                self.lbl_preview.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        self.lbl_preview.setToolTip(latest_image)
        self.log(f"プレビュー更新: {latest_image}")

    @staticmethod
    def _find_latest_image(root_dir: str) -> str | None:
        exts = (".png", ".jpg", ".jpeg")
        latest_path = None
        latest_mtime = None
        for current_root, _, files in os.walk(root_dir):
            for name in files:
                if not name.lower().endswith(exts):
                    continue
                path = os.path.join(current_root, name)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                if latest_path is None or mtime > latest_mtime:
                    latest_path = path
                    latest_mtime = mtime
        return latest_path

    # ---------- 入力チェック ----------

    def _validate_inputs(self) -> bool:
        video_path = self.le_video.text().strip()
        if not video_path or not os.path.isfile(video_path):
            QMessageBox.warning(self, "エラー", "有効な動画ファイルを選択してください。")
            return False
        if not video_path.lower().endswith(".mp4"):
            QMessageBox.warning(self, "エラー", "mp4 形式の動画ファイルを選択してください。")
            return False
        return True

    # ---------- closeEvent ----------

    def closeEvent(self, event):
        self._save_config()
        if self.current_worker is not None and self.current_worker.isRunning():
            self.current_worker.quit()
            self.current_worker.wait(1000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()






