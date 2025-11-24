from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal as Signal

from .models import AnalysisResult, QuickEditSettings, TranscriptionResult
from .pipeline import AnalyzerPipeline
from .transcription import VoskTranscriber


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)


class AnalyzerTask(QRunnable):
    """Background task that runs analysis plus verbatim transcription."""

    def __init__(
        self,
        pipeline: AnalyzerPipeline,
        transcriber: VoskTranscriber,
        media_path: Path,
        settings: QuickEditSettings,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.transcriber = transcriber
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
            transcription: TranscriptionResult = self.transcriber.transcribe(
                analysis.wav_path,
                analysis.speech_segments,
            )
            payload = {
                "analysis": analysis,
                "transcription": transcription,
                "settings": self.settings,
            }
            self.signals.finished.emit(payload)
        except Exception as exc:
            self.signals.error.emit(str(exc))
