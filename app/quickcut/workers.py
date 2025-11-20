from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal as Signal

from .models import AnalysisResult, QuickEditSettings, TranscriptionResult
from .pipeline import AnalyzerPipeline
from .transcription import TranscriptionEngine


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)


class AnalyzerTask(QRunnable):
    """Background task that performs analysis and transcription sequentially."""

    def __init__(
        self,
        pipeline: AnalyzerPipeline,
        transcription: TranscriptionEngine,
        media_path: Path,
        settings: QuickEditSettings,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.transcription = transcription
        self.media_path = Path(media_path)
        self.settings = settings
        self.signals = WorkerSignals()

    def run(self):
        try:
            analysis: AnalysisResult = self.pipeline.analyze(
                self.media_path,
                self.settings,
                progress_cb=self.signals.progress.emit,
            )
            transcription_result: TranscriptionResult = self.transcription.transcribe(
                analysis.speech_wav_path,
                analysis.speech_time_map,
                analysis.envelope,
                analysis.low_segments,
                self.settings.low_energy_threshold_db,
            )
            payload = {
                "analysis": analysis,
                "transcription": transcription_result,
                "settings": self.settings,
            }
            self.signals.finished.emit(payload)
        except Exception as exc:
            self.signals.error.emit(str(exc))
