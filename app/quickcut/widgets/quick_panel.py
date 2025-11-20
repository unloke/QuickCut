from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal as Signal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..models import QuickEditSettings


class QuickEditPanel(QFrame):
    settingsChanged = Signal(object)
    analyzeRequested = Signal(object)
    exportRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("quickPanel")
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        grid = QGridLayout()
        row = 0

        self.pre_roll = QSpinBox()
        self.pre_roll.setRange(0, 2000)
        self.pre_roll.setSuffix(" ms")
        self.pre_roll.setValue(120)

        self.post_roll = QSpinBox()
        self.post_roll.setRange(0, 2000)
        self.post_roll.setSuffix(" ms")
        self.post_roll.setValue(150)

        self.silence_db = QDoubleSpinBox()
        self.silence_db.setRange(-90.0, 0.0)
        self.silence_db.setSuffix(" dB")
        self.silence_db.setValue(-42.0)

        self.low_db = QDoubleSpinBox()
        self.low_db.setRange(-90.0, 0.0)
        self.low_db.setSuffix(" dB")
        self.low_db.setValue(-34.0)

        self.min_silence = QDoubleSpinBox()
        self.min_silence.setRange(0.05, 2.0)
        self.min_silence.setSingleStep(0.05)
        self.min_silence.setSuffix(" s")
        self.min_silence.setValue(0.30)

        self.min_speech = QDoubleSpinBox()
        self.min_speech.setRange(0.05, 2.0)
        self.min_speech.setSingleStep(0.05)
        self.min_speech.setSuffix(" s")
        self.min_speech.setValue(0.15)

        self.remove_fillers = QCheckBox("遮罩口頭禪")
        self.remove_fillers.setChecked(True)

        self.remove_repeats = QCheckBox("遮罩重複語意")
        self.remove_repeats.setChecked(True)

        for label, widget in [
            ("語音前預留", self.pre_roll),
            ("語音後預留", self.post_roll),
            ("靜音門檻", self.silence_db),
            ("低能量門檻", self.low_db),
            ("最小靜音", self.min_silence),
            ("最小語音", self.min_speech),
        ]:
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
            row += 1

        layout.addLayout(grid)
        layout.addWidget(self.remove_fillers)
        layout.addWidget(self.remove_repeats)

        self.analyze_button = QPushButton("分析")
        self.export_button = QPushButton("輸出結果")
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.export_button)
        layout.addStretch(1)

        for widget in [
            self.pre_roll,
            self.post_roll,
            self.silence_db,
            self.low_db,
            self.min_silence,
            self.min_speech,
        ]:
            widget.valueChanged.connect(self._emit_settings)
        self.remove_fillers.stateChanged.connect(self._emit_settings)
        self.remove_repeats.stateChanged.connect(self._emit_settings)
        self.analyze_button.clicked.connect(self._handle_analyze_clicked)
        self.export_button.clicked.connect(self.exportRequested.emit)

        self._emit_settings()

    def settings(self) -> QuickEditSettings:
        return QuickEditSettings(
            pre_roll_ms=self.pre_roll.value(),
            post_roll_ms=self.post_roll.value(),
            silence_threshold_db=self.silence_db.value(),
            low_energy_threshold_db=self.low_db.value(),
            min_silence=self.min_silence.value(),
            min_speech=self.min_speech.value(),
            remove_fillers=self.remove_fillers.isChecked(),
            remove_repeated=self.remove_repeats.isChecked(),
        )

    def _emit_settings(self):
        self.settingsChanged.emit(self.settings())

    def _handle_analyze_clicked(self):
        self.analyzeRequested.emit(self.settings())

    def set_analyze_enabled(self, enabled: bool):
        self.analyze_button.setEnabled(enabled)
