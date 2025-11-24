from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from .models import AnalysisResult, QuickEditSettings, Segment


class AnalyzerPipeline:
    """Coordinates FFmpeg extraction and C++ analyzer execution."""

    def __init__(self, analyzer_executable: Path):
        self.analyzer_executable = Path(analyzer_executable)
        if not self.analyzer_executable.exists():
            raise FileNotFoundError(f"Audio analyzer binary not found: {self.analyzer_executable}")
        self._audio_cache: dict[Path, tuple[int, int, Path]] = {}

    def analyze(
        self,
        media_path: Path,
        settings: QuickEditSettings,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> AnalysisResult:
        media_path = Path(media_path)
        media_stat = media_path.stat()
        cached_wav = self._get_cached_wav(media_path, media_stat)
        cache_dir: Path | None = None
        wav_path: Path
        try:
            if cached_wav and cached_wav.exists():
                wav_path = cached_wav
                cache_dir = wav_path.parent
                if progress_cb:
                    progress_cb("使用快取的音訊檔…")
            else:
                if progress_cb:
                    progress_cb("抽取聲音中…")
                cache_dir = Path(tempfile.mkdtemp(prefix="quickcut_"))
                wav_path = cache_dir / "source.wav"
                self._extract_audio(media_path, wav_path)
                self._audio_cache[media_path.resolve()] = (
                    media_stat.st_size,
                    int(media_stat.st_mtime),
                    wav_path,
                )
            if progress_cb:
                progress_cb("執行 C++ 分析中…")

            payload = self._run_analyzer(wav_path, settings)
            result = self._build_analysis_result(
                payload,
                media_path=media_path,
                cache_dir=cache_dir or wav_path.parent,
                wav_path=wav_path,
            )
            return result
        except Exception:
            if cache_dir and cache_dir.exists() and not cached_wav:
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
            "0",
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

    def _get_cached_wav(self, media_path: Path, media_stat: Optional[os.stat_result]) -> Path | None:
        key = media_path.resolve()
        cached = self._audio_cache.get(key)
        if not cached:
            return None
        size, mtime, wav_path = cached
        if not wav_path.exists():
            return None
        if media_stat is None:
            try:
                media_stat = media_path.stat()
            except OSError:
                return None
        if media_stat.st_size == size and int(media_stat.st_mtime) == mtime:
            return wav_path
        return None

    def _build_analysis_result(
        self,
        payload: dict,
        *,
        media_path: Path,
        cache_dir: Path,
        wav_path: Path,
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
            speech_wav_path=wav_path,
            sample_rate=sample_rate,
            duration=duration,
            envelope=envelope,
            speech_segments=speech_segments,
            silence_segments=silence_segments,
            low_segments=low_segments,
            mask_suggestions=mask_suggestions,
            speech_time_map=[],
        )

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
