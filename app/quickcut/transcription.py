from __future__ import annotations

import re
import string
import threading
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Set

from .models import (
    CharacterTiming,
    Segment,
    TranscriptSegment,
    TranscriptionResult,
    WordTiming,
    render_text_from_words,
)

LABEL_PRIORITY = {"normal": 0, "repeat": 1, "stutter": 2, "filler": 3}
FILLER_MERGE_GAP = 0.08
DETECTION_MIN_DURATION = 0.02
AUTO_HIDE_MIN_DURATION = 0.08
AUTO_HIDE_MAX_DURATION = 0.5
PADDING_BEFORE = 0.04
PADDING_AFTER = 0.05
STUTTER_MAX_GAP = 0.35
MAX_BAD_RUN = 3
BAD_RATIO_THRESHOLD = 0.85
LOW_ENERGY_MIN_OVERLAP_RATIO = 0.8
HIGH_ENERGY_VETO_RATIO = 0.25
CONFIDENCE_THRESHOLD = 0.4
PUNCTUATION_CHARS = string.punctuation + "，。！？；：「」『』（）《》〈〉、……—～·【】『』"

EN_FILLER_SINGLE = {
    "um",
    "uh",
    "erm",
    "hmm",
    "like",
    "actually",
    "basically",
    "literally",
    "kind",
    "sort",
}
ZH_FILLER_SINGLE = {"欸", "嗯", "呃", "啊", "啦", "然後", "那個", "這個", "其實", "就是說", "這樣子", "基本上"}
FILLER_MULTI_TOKENS = [
    ("you", "know"),
    ("i", "mean"),
    ("you", "see"),
    ("sort", "of"),
    ("kind", "of"),
    ("in", "fact"),
    ("你", "知道"),
    ("那個",),
    ("這個",),
    ("然後",),
    ("其實",),
    ("就是說",),
    ("基本上",),
    ("這樣子",),
]


@dataclass
class WordToken:
    text: str
    start: float
    end: float
    normalized: str
    label: str = "normal"
    confidence: float = 1.0

    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def set_label(self, new_label: str) -> None:
        if LABEL_PRIORITY[new_label] > LABEL_PRIORITY[self.label]:
            self.label = new_label


@dataclass
class SuppressedInterval:
    start: float
    end: float
    label: str
    content: str
    token_indices: List[int] = field(default_factory=list)
    label_counts: Dict[str, int] = field(default_factory=dict)
    detection_reasons: Set[str] = field(default_factory=set)
    decision_reasons: Set[str] = field(default_factory=set)
    auto_hide: bool = False
    token_count: int = 0
    bad_token_count: int = 0
    bad_ratio: float = 0.0
    energy_overlap: float = 0.0
    avg_confidence: float = 0.0
    core_start: float = 0.0
    core_end: float = 0.0
    longest_bad_run: int = 0
    loud_ratio: float = 0.0


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = text.strip()
    if not normalized:
        return ""
    normalized = normalized.strip(PUNCTUATION_CHARS).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.lower()


def _build_word_stream(segments: List[dict]) -> tuple[List[WordToken], List[tuple[int, int]]]:
    tokens: List[WordToken] = []
    spans: List[tuple[int, int]] = []
    for entry in segments:
        seg_start_idx = len(tokens)
        words = entry.get("words") or []
        if not words:
            token = _build_token(entry.get("text", ""), entry.get("start"), entry.get("end"), None)
            if token:
                tokens.append(token)
        else:
            for word in words:
                token = _build_token(word.get("word"), word.get("start"), word.get("end"), word.get("probability"))
                if token:
                    tokens.append(token)
        spans.append((seg_start_idx, len(tokens)))
    return tokens, spans


def _build_token(
    word: str | None,
    start: float | None,
    end: float | None,
    confidence: float | None,
) -> WordToken | None:
    raw_text = word or ""
    trimmed = raw_text.strip()
    if not trimmed:
        return None
    start = float(start or 0.0)
    end = float(end or start)
    if end <= start:
        end = start + 1e-3
    conf = float(confidence) if confidence is not None else 1.0
    if not (0.0 <= conf <= 1.0):
        conf = 1.0
    return WordToken(text=raw_text, start=start, end=end, normalized=_normalize_text(raw_text), confidence=conf)


def _estimate_speech_duration(
    segments: List[dict],
    tokens: Sequence[WordToken],
    time_map: List[tuple[float, float, float, float]] | None,
) -> float:
    if time_map:
        return max((cut_end for *_, cut_end in time_map), default=0.0)
    if tokens:
        return max(token.end for token in tokens)
    return max((float(entry.get("end", 0.0)) for entry in segments), default=0.0)


def _label_single_word_fillers(tokens: Sequence[WordToken]) -> None:
    for token in tokens:
        norm = token.normalized
        if not norm:
            continue
        if norm in EN_FILLER_SINGLE or norm in ZH_FILLER_SINGLE:
            token.set_label("filler")


def _label_multi_word_fillers(tokens: Sequence[WordToken]) -> None:
    if not tokens:
        return
    patterns = [_normalize_pattern(pattern) for pattern in FILLER_MULTI_TOKENS]
    if not patterns:
        return
    max_len = max(len(pattern) for pattern in patterns)
    lookup = {pattern: pattern for pattern in patterns}
    for idx in range(len(tokens)):
        for length in range(max_len, 1, -1):
            if idx + length > len(tokens):
                continue
            window = tuple(token.normalized for token in tokens[idx : idx + length])
            if window in lookup and all(window):
                for token in tokens[idx : idx + length]:
                    token.set_label("filler")
                break


def _normalize_pattern(pattern: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in pattern if part.strip())


def _label_stutters(tokens: Sequence[WordToken]) -> None:
    for idx in range(len(tokens) - 1):
        current = tokens[idx]
        nxt = tokens[idx + 1]
        if (
            current.normalized
            and current.normalized == nxt.normalized
            and nxt.label == "normal"
            and nxt.start - current.end <= STUTTER_MAX_GAP
        ):
            nxt.set_label("stutter")


def _collect_hidden_intervals(tokens: Sequence[WordToken]) -> List[SuppressedInterval]:
    intervals: List[SuppressedInterval] = []
    idx = 0
    total = len(tokens)
    while idx < total:
        token = tokens[idx]
        if token.label == "filler":
            start_idx = idx
            end_idx = idx
            content = token.text.strip()
            current_end = token.end
            while end_idx + 1 < total:
                nxt = tokens[end_idx + 1]
                if nxt.label == "filler" and nxt.start - current_end <= FILLER_MERGE_GAP:
                    end_idx += 1
                    current_end = max(current_end, nxt.end)
                    if nxt.text.strip():
                        content = f"{content} {nxt.text.strip()}".strip()
                    continue
                break
            indices = list(range(start_idx, end_idx + 1))
            interval = SuppressedInterval(
                start=tokens[start_idx].start,
                end=max(tokens[end_idx].end, tokens[start_idx].end),
                label="filler",
                content=content or "filler",
                token_indices=indices,
                label_counts={"filler": len(indices)},
                detection_reasons={"filler"},
                core_start=tokens[start_idx].start,
                core_end=max(tokens[end_idx].end, tokens[start_idx].end),
                longest_bad_run=len(indices),
            )
            intervals.append(interval)
            idx = end_idx + 1
            continue
        if token.label == "stutter":
            prev_norm = tokens[idx - 1].normalized if idx > 0 else ""
            if prev_norm and prev_norm == token.normalized:
                interval = SuppressedInterval(
                    start=token.start,
                    end=token.end,
                    label="stutter",
                    content=token.text.strip() or token.normalized or "stutter",
                    token_indices=[idx],
                    label_counts={"stutter": 1},
                    detection_reasons={"stutter"},
                    core_start=token.start,
                    core_end=token.end,
                    longest_bad_run=1,
                )
                intervals.append(interval)
        idx += 1

    cleaned: List[SuppressedInterval] = []
    for interval in intervals:
        duration = interval.end - interval.start
        if duration < DETECTION_MIN_DURATION:
            continue
        interval.bad_token_count = len(interval.token_indices)
        if not interval.content:
            interval.content = interval.label
        cleaned.append(interval)
    return cleaned


def _decide_auto_hide(
    intervals: Sequence[SuppressedInterval],
    tokens: Sequence[WordToken],
    low_energy_spans: Sequence[tuple[float, float]],
) -> None:
    for interval in intervals:
        interval.token_count = _count_tokens_in_span(tokens, interval.core_start, interval.core_end)
        interval.bad_token_count = len(interval.token_indices)
        if interval.bad_token_count:
            conf_sum = sum(max(0.0, min(1.0, tokens[idx].confidence)) for idx in interval.token_indices)
            interval.avg_confidence = conf_sum / interval.bad_token_count
        else:
            interval.avg_confidence = 0.0
        interval.bad_ratio = (interval.bad_token_count / interval.token_count) if interval.token_count > 0 else 0.0
        interval.energy_overlap = _overlap_ratio(interval.core_start, interval.core_end, low_energy_spans)
        interval.auto_hide = _interval_passes_gates(interval)


def _count_tokens_in_span(tokens: Sequence[WordToken], start: float, end: float) -> int:
    total = 0
    for token in tokens:
        if token.end <= start:
            continue
        if token.start >= end:
            break
        total += 1
    return total


def _interval_passes_gates(interval: SuppressedInterval) -> bool:
    duration = max(0.0, interval.core_end - interval.core_start)
    ok = True
    if duration < AUTO_HIDE_MIN_DURATION:
        interval.decision_reasons.add("duration_short")
        ok = False
    if duration > AUTO_HIDE_MAX_DURATION:
        interval.decision_reasons.add("duration_long")
        ok = False
    if interval.token_count == 0:
        interval.decision_reasons.add("no_tokens")
        ok = False
    elif interval.bad_ratio < BAD_RATIO_THRESHOLD:
        interval.decision_reasons.add("density_low")
        ok = False
    if interval.energy_overlap < LOW_ENERGY_MIN_OVERLAP_RATIO:
        interval.decision_reasons.add("energy_high")
        ok = False
    if interval.longest_bad_run > MAX_BAD_RUN:
        interval.decision_reasons.add("run_limit")
        ok = False
    if interval.avg_confidence < CONFIDENCE_THRESHOLD:
        interval.decision_reasons.add("low_confidence")
        ok = False
    return ok


def _apply_interval_padding(
    intervals: Sequence[SuppressedInterval],
    speech_duration: float,
) -> List[SuppressedInterval]:
    padded: List[SuppressedInterval] = []
    for interval in intervals:
        start = max(0.0, interval.core_start - PADDING_BEFORE)
        end = interval.core_end + PADDING_AFTER
        if speech_duration > 0:
            end = min(speech_duration, end)
        if end - start <= 1e-3:
            continue
        interval.start = start
        interval.end = end
        padded.append(interval)
    return padded


def _project_low_energy_segments(
    low_segments: Sequence[Segment] | None,
    time_map: List[tuple[float, float, float, float]] | None,
) -> List[tuple[float, float]]:
    if not low_segments:
        return []
    if not time_map:
        spans = []
        for seg in low_segments:
            start = float(seg.start)
            end = float(seg.end)
            if end > start:
                spans.append((start, end))
        return spans

    spans: List[tuple[float, float]] = []
    for seg in low_segments:
        low_start = float(seg.start)
        low_end = float(seg.end)
        if low_end <= low_start:
            continue
        for original_start, original_end, cut_start, cut_end in time_map:
            if low_end <= original_start or low_start >= original_end:
                continue
            span_start = max(low_start, original_start)
            span_end = min(low_end, original_end)
            if span_end <= span_start:
                continue
            original_span = max(original_end - original_start, 1e-6)
            cut_span = max(cut_end - cut_start, 1e-6)
            scale = cut_span / original_span
            cut_span_start = cut_start + (span_start - original_start) * scale
            cut_span_end = cut_start + (span_end - original_start) * scale
            if cut_span_end > cut_span_start:
                spans.append((cut_span_start, cut_span_end))
    spans.sort(key=lambda item: item[0])
    return spans


def _map_intervals_to_original(
    intervals: Sequence[SuppressedInterval],
    mapper: Callable[[float], float],
) -> List[Segment]:
    mapped: List[Segment] = []
    for interval in intervals:
        start = mapper(interval.start)
        end = mapper(interval.end)
        if end - start <= 1e-4:
            continue
        meta: Dict[str, object] = {
            "auto_hide": interval.auto_hide,
            "token_count": interval.token_count,
            "bad_token_count": interval.bad_token_count,
            "bad_ratio": interval.bad_ratio,
            "energy_overlap": interval.energy_overlap,
            "avg_confidence": interval.avg_confidence,
            "core_start": interval.core_start,
            "core_end": interval.core_end,
            "label_counts": dict(interval.label_counts),
            "longest_bad_run": interval.longest_bad_run,
            "loud_ratio": interval.loud_ratio,
        }
        if interval.detection_reasons:
            meta["detection_reasons"] = sorted(interval.detection_reasons)
        if interval.decision_reasons:
            meta["decision_reasons"] = sorted(interval.decision_reasons)
        mapped.append(Segment(start=start, end=end, kind=interval.label, label=interval.content, meta=meta))
    mapped.sort(key=lambda seg: seg.start)
    return mapped


def _overlap_ratio(start: float, end: float, spans: Sequence[tuple[float, float]]) -> float:
    duration = max(end - start, 1e-6)
    if not spans:
        return 0.0
    overlap = 0.0
    for span_start, span_end in spans:
        if span_end <= start:
            continue
        if span_start >= end:
            break
        overlap += max(0.0, min(end, span_end) - max(start, span_start))
    return max(0.0, min(1.0, overlap / duration))


def _map_words_to_original(tokens: Sequence[WordToken], mapper: Callable[[float], float]) -> List[WordTiming]:
    mapped: List[WordTiming] = []
    for token in tokens:
        start = mapper(token.start)
        end = mapper(token.end)
        if end - start <= 1e-4:
            end = start + 1e-4
        mapped.append(WordTiming(word=token.text, start=start, end=end, normalized=token.normalized, label=token.label))
    return mapped


def _slice_intervals_for_segment(ranges: Sequence[Segment], start: float, end: float) -> List[Segment]:
    sliced: List[Segment] = []
    for rng in ranges:
        overlap_start = max(start, rng.start)
        overlap_end = min(end, rng.end)
        if overlap_end - overlap_start <= 1e-3:
            continue
        sliced.append(
            Segment(
                start=overlap_start,
                end=overlap_end,
                kind=rng.kind,
                label=rng.label,
                meta=dict(rng.meta or {}),
            )
        )
    return sliced


class TranscriptionEngine:
    """Wraps the Whisper tiny model with lightweight caching."""

    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            import whisper

            self._model = whisper.load_model(self.model_size)

    def transcribe(
        self,
        wav_path: Path,
        time_map: List[tuple[float, float, float, float]] | None = None,
        envelope: List[tuple[float, float]] | None = None,
        low_energy_segments: Sequence[Segment] | None = None,
        low_energy_threshold_db: float | None = None,
    ) -> TranscriptionResult:
        wav_path = Path(wav_path)
        with self._lock:
            self._ensure_model()
            result = self._model.transcribe(
                str(wav_path),
                fp16=False,
                verbose=False,
                word_timestamps=True,
            )

        segments_data = sorted(result.get("segments", []), key=lambda seg: float(seg.get("start", 0.0)))
        if not segments_data:
            return TranscriptionResult(segments=[], full_text="")

        word_stream, segment_word_spans = _build_word_stream(segments_data)
        speech_duration = _estimate_speech_duration(segments_data, word_stream, time_map)
        mapper = lambda value: self._map_to_original_time(value, time_map)
        _label_single_word_fillers(word_stream)
        _label_multi_word_fillers(word_stream)
        _label_stutters(word_stream)
        hidden_intervals = _collect_hidden_intervals(word_stream)
        energy_spans = _project_low_energy_segments(low_energy_segments, time_map)
        _decide_auto_hide(hidden_intervals, word_stream, energy_spans)
        self._apply_envelope_veto(hidden_intervals, mapper, envelope, low_energy_threshold_db)
        hidden_intervals = _apply_interval_padding(hidden_intervals, speech_duration)

        mapped_words = _map_words_to_original(word_stream, mapper)
        mapped_hidden = _map_intervals_to_original(hidden_intervals, mapper)

        transcripts: List[TranscriptSegment] = []
        for idx, entry in enumerate(segments_data):
            local_start = float(entry.get("start", 0.0))
            local_end = float(entry.get("end", local_start))
            start = mapper(local_start)
            end = mapper(local_end)
            span_start, span_end = segment_word_spans[idx]
            word_slice = mapped_words[span_start:span_end]
            display_text = render_text_from_words(word_slice) or entry.get("text", "").strip()
            characters = self._build_character_map(display_text, start, end, envelope, word_slice)
            suppressed_ranges = _slice_intervals_for_segment(mapped_hidden, start, end)
            transcripts.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=display_text,
                    is_filler=any(rng.kind == "filler" for rng in suppressed_ranges),
                    is_repeat=any(rng.kind in {"stutter", "repeat"} for rng in suppressed_ranges),
                    characters=characters,
                    words=word_slice,
                    suppressed_ranges=suppressed_ranges,
                )
            )

        full_text = " ".join(seg.text for seg in transcripts).strip()
        return TranscriptionResult(segments=transcripts, full_text=full_text)

    def _map_to_original_time(
        self, local_time: float, time_map: List[tuple[float, float, float, float]] | None
    ) -> float:
        if not time_map:
            return local_time
        for original_start, original_end, cut_start, cut_end in time_map:
            if cut_start <= local_time <= cut_end:
                span = max(1e-6, cut_end - cut_start)
                ratio = (local_time - cut_start) / span
                return original_start + ratio * (original_end - original_start)
        if local_time < time_map[0][2]:
            return time_map[0][0]
        return time_map[-1][1]

    def _apply_envelope_veto(
        self,
        intervals: Sequence[SuppressedInterval],
        mapper: Callable[[float], float],
        envelope: Sequence[tuple[float, float]] | None,
        threshold_db: float | None,
    ) -> None:
        if not intervals or not envelope or threshold_db is None:
            return
        envelope_points = list(envelope)
        if not envelope_points:
            return
        envelope_times = [point[0] for point in envelope_points]
        sample_threshold = 10 ** (float(threshold_db) / 20.0)
        for interval in intervals:
            original_start = mapper(interval.core_start)
            original_end = mapper(interval.core_end)
            if original_end - original_start <= 1e-4:
                interval.loud_ratio = 0.0
                continue
            span = original_end - original_start
            steps = max(3, min(25, int(span / 0.04)))
            values: List[float] = []
            for step in range(steps):
                if steps == 1:
                    point = (original_start + original_end) / 2.0
                else:
                    point = original_start + span * (step / (steps - 1))
                values.append(self._sample_envelope(envelope_points, envelope_times, point))
            if not values:
                interval.loud_ratio = 0.0
                continue
            loud_ratio = sum(1 for value in values if value > sample_threshold) / len(values)
            interval.loud_ratio = loud_ratio
            if interval.auto_hide and loud_ratio > HIGH_ENERGY_VETO_RATIO:
                interval.auto_hide = False
                interval.decision_reasons.add("energy_veto")

    def _build_character_map(
        self,
        text: str,
        start: float,
        end: float,
        envelope: List[tuple[float, float]] | None,
        word_timings: Sequence[WordTiming] | None,
    ) -> List[CharacterTiming]:
        cleaned = [ch for ch in text if not ch.isspace()]
        if not cleaned or end <= start:
            return []
        total = len(cleaned)
        step = (end - start) / total
        envelope_points: Sequence[tuple[float, float]] = envelope or []
        envelope_times = [point[0] for point in envelope_points]
        timeline: List[CharacterTiming] = []
        if word_timings:
            for word in word_timings:
                chars = [ch for ch in word.word if not ch.isspace()]
                if not chars:
                    continue
                word_start = max(start, word.start)
                word_end = min(end, word.end)
                if word_end <= word_start:
                    continue
                word_step = (word_end - word_start) / len(chars)
                cursor = word_start
                for idx, ch in enumerate(chars):
                    char_start = cursor
                    char_end = word_start + word_step * (idx + 1) if idx < len(chars) - 1 else word_end
                    center = (char_start + char_end) / 2.0
                    pitch = self._sample_envelope(envelope_points, envelope_times, center)
                    timeline.append(CharacterTiming(char=ch, start=char_start, end=char_end, pitch=pitch))
                    cursor = char_end
            if len(timeline) == total:
                return timeline

        cursor = start
        for idx, ch in enumerate(cleaned):
            char_start = cursor
            char_end = start + step * (idx + 1) if idx < total - 1 else end
            center = (char_start + char_end) / 2.0
            pitch = self._sample_envelope(envelope_points, envelope_times, center)
            timeline.append(CharacterTiming(char=ch, start=char_start, end=char_end, pitch=pitch))
            cursor = char_end
        return timeline

    def _sample_envelope(
        self,
        envelope: Sequence[tuple[float, float]],
        times: Sequence[float],
        time_point: float,
    ) -> float:
        if not envelope:
            return 0.0
        idx = bisect_left(times, time_point)
        if idx <= 0:
            return envelope[0][1]
        if idx >= len(envelope):
            return envelope[-1][1]
        t0, v0 = envelope[idx - 1]
        t1, v1 = envelope[idx]
        if t1 == t0:
            return v0
        ratio = (time_point - t0) / (t1 - t0)
        return v0 + ratio * (v1 - v0)
