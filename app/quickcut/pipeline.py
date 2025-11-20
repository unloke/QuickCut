from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .models import AnalysisResult, QuickEditSettings, Segment


class AnalyzerPipeline:
    """Coordinates FFmpeg extraction, C++ analyzer execution, and speech-only export."""

    def __init__(self, analyzer_executable: Path):
        self.analyzer_executable = Path(analyzer_executable)
        if not self.analyzer_executable.exists():
            raise FileNotFoundError(f"Audio analyzer binary not found: {self.analyzer_executable}")

    def analyze(
        self,
        media_path: Path,
        settings: QuickEditSettings,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> AnalysisResult:
        media_path = Path(media_path)
        if progress_cb:
            progress_cb("抽取聲音中…")

        cache_dir = Path(tempfile.mkdtemp(prefix="quickcut_"))
        wav_path = cache_dir / "source.wav"
        speech_wav_path = cache_dir / "speech_only.wav"

        try:
            self._extract_audio(media_path, wav_path)
            if progress_cb:
                progress_cb("執行 C++ 分析中…")

            payload = self._run_analyzer(wav_path, settings)
            result = self._build_analysis_result(
                payload,
                media_path=media_path,
                cache_dir=cache_dir,
                wav_path=wav_path,
                speech_wav_path=speech_wav_path,
            )

            if progress_cb:
                progress_cb("產出僅保留語音的音檔…")

            result.speech_time_map = self._render_speech_only_audio(result, speech_wav_path)
            return result
        except Exception:
            shutil.rmtree(cache_dir, ignore_errors=True)
            raise

    def _extract_audio(self, media_path: Path, wav_path: Path) -> None:
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(wav_path),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"FFmpeg 無法處理檔案:\n{completed.stderr.strip()}")

    def _run_analyzer(self, wav_path: Path, settings: QuickEditSettings) -> dict:
        cmd = [
            str(self.analyzer_executable),
            "--input",
            str(wav_path),
            "--threshold-db",
            f"{settings.silence_threshold_db}",
            "--low-db",
            f"{settings.low_energy_threshold_db}",
            "--min-silence",
            f"{settings.min_silence}",
            "--min-speech",
            f"{settings.min_speech}",
            "--window-ms",
            f"{settings.window_ms}",
            "--hop-ms",
            f"{settings.hop_ms}",
            "--pre-roll",
            f"{settings.pre_roll_seconds()}",
            "--post-roll",
            f"{settings.post_roll_seconds()}",
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        stdout = completed.stdout.strip()
        if completed.returncode != 0:
            raise RuntimeError(f"分析程式回報錯誤:\n{stdout or completed.stderr}")
        data = json.loads(stdout or "{}")
        if "error" in data:
            raise RuntimeError(data["error"])
        return data

    def _build_analysis_result(
        self,
        payload: dict,
        *,
        media_path: Path,
        cache_dir: Path,
        wav_path: Path,
        speech_wav_path: Path,
    ) -> AnalysisResult:
        sample_rate = int(payload.get("sample_rate", 16000))
        duration = float(payload.get("duration", 0))
        envelope = [
            (float(item.get("time", 0.0)), float(item.get("rms", 0.0)))
            for item in payload.get("envelope", [])
        ]

        def parse_segments(key: str, kind: str) -> List[Segment]:
            values = []
            for entry in payload.get("segments", {}).get(key, []):
                values.append(
                    Segment(
                        start=float(entry.get("start", 0.0)),
                        end=float(entry.get("end", 0.0)),
                        kind=kind,
                        label=entry.get("label", ""),
                        meta={k: entry.get(k) for k in ("peak", "avg")},
                    )
                )
            return values

        speech_segments = self._merge_segments(parse_segments("speech", "speech"))
        silence_segments = parse_segments("silence", "silence")
        low_segments = parse_segments("low_energy", "low_energy")

        mask_suggestions: List[Segment] = []
        for entry in payload.get("mask_suggestions", []):
            mask_suggestions.append(
                Segment(
                    start=float(entry.get("start", 0.0)),
                    end=float(entry.get("end", 0.0)),
                    kind=str(entry.get("type", "silence")),
                )
            )

        if not speech_segments and duration > 0:
            speech_segments = [Segment(start=0.0, end=duration, kind="speech")]

        return AnalysisResult(
            media_path=media_path,
            cache_dir=cache_dir,
            wav_path=wav_path,
            speech_wav_path=speech_wav_path,
            sample_rate=sample_rate,
            duration=duration,
            envelope=envelope,
            speech_segments=speech_segments,
            silence_segments=silence_segments,
            low_segments=low_segments,
            mask_suggestions=mask_suggestions,
            speech_time_map=[],
        )

    def _render_speech_only_audio(self, analysis: AnalysisResult, output_path: Path) -> List[tuple[float, float, float, float]]:
        with wave.open(str(analysis.wav_path), "rb") as reader:
            channels = reader.getnchannels()
            if channels != 1:
                raise RuntimeError("僅支援單聲道音訊。")
            sample_width = reader.getsampwidth()
            if sample_width != 2:
                raise RuntimeError("音訊格式錯誤，需為 16bit PCM。")
            audio_data = reader.readframes(reader.getnframes())

        samples = np.frombuffer(audio_data, dtype=np.int16).copy()
        total = len(samples)
        segments = analysis.speech_segments
        chunks: List[np.ndarray] = []
        mapping: List[tuple[float, float, float, float]] = []
        trimmed_cursor = 0.0

        for seg in segments:
            start = max(0, int(seg.start * analysis.sample_rate))
            end = min(total, int(seg.end * analysis.sample_rate))
            if end > start:
                chunks.append(samples[start:end])
                chunk_duration = (end - start) / float(analysis.sample_rate)
                mapping.append((seg.start, seg.end, trimmed_cursor, trimmed_cursor + chunk_duration))
                trimmed_cursor += chunk_duration

        if not chunks:
            chunks.append(samples)
            full_duration = total / float(analysis.sample_rate)
            mapping.append((0.0, full_duration, 0.0, full_duration))

        merged = np.concatenate(chunks)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(analysis.sample_rate)
            writer.writeframes(merged.astype(np.int16).tobytes())
        return mapping

    def _merge_segments(self, segments: List[Segment]) -> List[Segment]:
        if not segments:
            return []
        normalized: List[Segment] = []
        for segment in sorted(segments, key=lambda seg: seg.start):
            start = float(segment.start)
            end = float(segment.end)
            if end <= start:
                continue
            payload = dict(segment.meta or {})
            if not normalized:
                normalized.append(Segment(start=start, end=end, kind=segment.kind, label=segment.label, meta=payload))
                continue
            last = normalized[-1]
            if start <= last.end + 1e-3:
                last.end = max(last.end, end)
            else:
                normalized.append(Segment(start=start, end=end, kind=segment.kind, label=segment.label, meta=payload))
        return normalized
