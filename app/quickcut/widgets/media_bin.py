from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal as Signal
from PyQt5.QtWidgets import QFileDialog, QFrame, QLabel, QVBoxLayout


class MediaBinWidget(QFrame):
    """Drop zone for quick media import."""

    mediaSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("mediaBin")
        self.setStyleSheet(
            """
            QFrame#mediaBin {
                border: 2px dashed #4a4a4a;
                border-radius: 12px;
                background-color: #1f1f1f;
                color: #f5f5f5;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label = QLabel("拖曳媒體到這裡\n或點擊選擇檔案", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.exists():
                self._handle_new_media(path)
                break

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "選擇影片或音訊",
                "",
                "Media Files (*.mp4 *.mov *.mkv *.mp3 *.wav *.m4a);;All Files (*.*)",
            )
            if file_path:
                self._handle_new_media(Path(file_path))

    def _handle_new_media(self, path: Path):
        self.label.setText(f"已載入：{path.name}")
        self.mediaSelected.emit(str(path))
