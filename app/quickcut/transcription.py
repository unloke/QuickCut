from __future__ import annotations

import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import json
from vosk import KaldiRecognizer, Model

from .models import (
    CharacterTiming,
    Segment,
    TranscriptSegment,
    TranscriptionResult,
    WordTiming,
    render_text_from_words,
)

# 擴充贅字庫 (中英混合)
SAFE_FILLERS = {
    "um", "uh", "erm", "hmm", "uhh", "er", "ah",
    "呃", "嗯", "啊"}
AMBIGUOUS_FILLERS = {"like", "actually", "basically", "literally", "sorta", "kinda", "right", "you know"}

PUNCTUATION = set(" !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~。，、！？：；「」『』（）—…")
VOWELS = set("aeiou")
STANDARD_ABBREVIATIONS = {"tv", "fm", "am", "ok"}
STUTTER_MAX_GAP = 0.35
PHRASE_REPEAT_MAX = 4
PHRASE_REPEAT_THRESHOLD = 0.82
INTERVAL_GAP = 0.25
SEGMENT_GAP = 1.0

# SenseVoice 建議的最小處理單元

def _is_junk_token(word: str, confidence: float) -> bool:
    text = (word or "").strip().lower()
    if not text:
        return True
    if confidence < 0.4:
        return True
    if len(text) == 1 and text not in {"a", "i"}:
        return True
    if len(text) < 3:
        if text in STANDARD_ABBREVIATIONS:
            return False
        if all(ch.isalpha() and ch not in VOWELS for ch in text):
            return True
    if not any(ch.isalpha() for ch in text):
        return True
    return False


@dataclass
class WordToken:
    text: str
    start: float
    end: float
    confidence: float
    label: str = "normal"

    @property
    def normalized(self) -> str:
        # 移除標點進行比較
        return "".join(ch for ch in self.text.lower().strip() if ch not in PUNCTUATION)


@dataclass
class SuppressedInterval:
    start: float
    end: float
    label: str
    tokens: List[int]


@dataclass
class MergedChunk:
    start: float
    end: float
    segments: List[Segment]
    index: int


class VoskTranscriber:
    """Performs verbatim ASR with Vosk and marks fillers / repeated phrases."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"找不到 Vosk 模型：{self.model_dir}")
        self._model: Model | None = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> Model:
        if self._model is None:
            self._model = Model(str(self.model_dir))
        return self._model

    def transcribe(
        self,
        wav_path: Path,
        speech_segments: List[Segment],
    ) -> TranscriptionResult:
        wav_path = Path(wav_path)
        if not speech_segments:
            return TranscriptionResult(segments=[], full_text="")

        words = self._run_vosk(wav_path, speech_segments)
        if not words:
            raise RuntimeError("轉錄失敗：未取得任何語音辨識結果，請確認音訊內容與格式。")
        words.sort(key=lambda item: float(item.get("start", 0.0)))
        ordered_segments = sorted(speech_segments, key=lambda seg: seg.start)

        transcripts: List[TranscriptSegment] = []
        for segment in ordered_segments:
            words_for_segment = _slice_words_for_segment(
                words,
                segment.start,
                segment.end,
            )

            segment_data = _build_segment_from_words(segment, words_for_segment)
            tokens, spans = _build_word_stream([segment_data])

            if not tokens:
                transcripts.append(
                    TranscriptSegment(
                        start=segment.start,
                        end=segment.end,
                        text="",
                        characters=[],
                        words=[],
                        suppressed_ranges=[],
                    )
                )
                continue

            _label_fillers(tokens)
            _label_repetitions(tokens)

            intervals = _collect_intervals(tokens)
            mapped_words = _tokens_to_word_timings(tokens)
            mapped_intervals = _intervals_to_segments(intervals)

            if spans:
                start_idx, end_idx = spans[0]
                word_slice = mapped_words[start_idx:end_idx]
            else:
                word_slice = []

            # 智慧渲染：SenseVoice 對中英文的處理
            display_text = _smart_render_text(word_slice)

            characters = _build_character_map(display_text, segment.start, segment.end)
            suppressed = _slice_intervals(mapped_intervals, segment.start, segment.end)

            transcripts.append(
                TranscriptSegment(
                    start=segment.start,
                    end=segment.end,
                    text=display_text,
                    is_filler=any(rng.kind == "filler" for rng in suppressed),
                    is_repeat=any(rng.kind == "repeat" for rng in suppressed),
                    characters=characters,
                    words=word_slice,
                    suppressed_ranges=suppressed,
                )
            )

        full_text = " ".join(seg.text for seg in transcripts if seg.text).strip()
        return TranscriptionResult(segments=transcripts, full_text=full_text)

    def _run_vosk(self, wav_path: Path, speech_segments: Sequence[Segment]) -> List[Dict[str, float | str]]:
        model = self._ensure_model()
        if not wav_path.exists():
            raise FileNotFoundError(f"音訊檔不存在：{wav_path}")

        with self._lock, wave.open(str(wav_path), "rb") as reader:
            channels = reader.getnchannels()
            sample_rate = reader.getframerate()
            sample_width = reader.getsampwidth()
            if channels != 1:
                raise RuntimeError("Vosk 僅支援單聲道 16kHz 音訊。")
            if sample_rate != 16000:
                raise RuntimeError(f"音訊取樣率需為 16kHz，目前為 {sample_rate}Hz。")
            if sample_width != 2:
                raise RuntimeError("音訊必須為 16-bit PCM 格式。")

            duration = reader.getnframes() / float(sample_rate or 1)
            chunks = _prepare_recognition_windows(speech_segments, duration)
            if not chunks:
                chunks = [MergedChunk(start=0.0, end=duration, segments=[], index=0)]

            bytes_per_frame = sample_width * channels
            words: List[Dict[str, float | str]] = []
            for chunk in chunks:
                start_frame = max(0, int(round(chunk.start * sample_rate)))
                end_frame = max(start_frame, int(round(chunk.end * sample_rate)))
                if end_frame <= start_frame:
                    continue
                reader.setpos(start_frame)

                rec = KaldiRecognizer(model, sample_rate)
                rec.SetWords(True)

                remaining_frames = end_frame - start_frame
                while remaining_frames > 0:
                    frame_batch = min(remaining_frames, 4000)
                    data = reader.readframes(frame_batch)
                    if not data:
                        break
                    frames_read = len(data) // bytes_per_frame
                    if frames_read <= 0:
                        break
                    remaining_frames -= frames_read
                    if rec.AcceptWaveform(data):
                        chunk_result = json.loads(rec.Result() or "{}")
                        words.extend(_extract_words_from_result(chunk_result, offset_seconds=start_frame / sample_rate))

                final_result = json.loads(rec.FinalResult() or "{}")
                words.extend(_extract_words_from_result(final_result, offset_seconds=start_frame / sample_rate))
        return words



def _merge_close_segments(segments: Sequence[Segment], gap_threshold: float = 1.0) -> List[MergedChunk]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda seg: seg.start)
    groups: List[List[Segment]] = []
    current: List[Segment] = [ordered[0]]

    for seg in ordered[1:]:
        prev = current[-1]
        gap = seg.start - prev.end
        if gap < gap_threshold:
            current.append(seg)
        else:
            groups.append(current)
            current = [seg]
    groups.append(current)

    merged: List[MergedChunk] = []
    for idx, group in enumerate(groups):
        start = group[0].start
        end = max(seg.end for seg in group)
        merged.append(MergedChunk(start=start, end=end, segments=list(group), index=idx))
    return merged


def _prepare_recognition_windows(segments: Sequence[Segment], duration: float) -> List[MergedChunk]:
    """Clamp segments to the audio duration and merge gaps to reduce Vosk passes."""
    if not segments or duration <= 0:
        return []
    valid: List[Segment] = []
    for seg in segments:
        start = max(0.0, min(float(seg.start), duration))
        end = max(0.0, min(float(seg.end), duration))
        if end - start <= 1e-3:
            continue
        valid.append(Segment(start=start, end=end, kind=seg.kind, label=seg.label, meta=dict(seg.meta or {})))
    if not valid:
        return []
    return _merge_close_segments(valid, gap_threshold=SEGMENT_GAP)


def _extract_words_from_result(payload: dict, offset_seconds: float = 0.0) -> List[Dict[str, float | str]]:
    words: List[Dict[str, float | str]] = []
    for word in payload.get("result", []) or []:
        start = offset_seconds + float(word.get("start", 0.0))
        end = offset_seconds + float(word.get("end", word.get("start", 0.0)))
        confidence = float(word.get("confidence", word.get("conf", word.get("probability", 1.0))))
        words.append(
            {
                "word": word.get("word", ""),
                "start": start,
                "end": end,
                "confidence": confidence,
            }
        )
    return words


def _slice_words_for_segment(
    words: Sequence[Dict[str, float | str]],
    start: float,
    end: float,
    tolerance: float = 0.15,
) -> List[Dict[str, float | str]]:
    sliced: List[Dict[str, float | str]] = []
    for word in words:
        w_start = float(word.get("start", 0.0))
        w_end = float(word.get("end", w_start))
        if w_end < start - tolerance or w_start > end + tolerance:
            continue
        sliced.append(
            {
                "word": word.get("word", ""),
                "start": max(start, w_start),
                "end": min(end, w_end),
                "confidence": float(word.get("confidence", 1.0)),
            }
        )
    return sliced


def _build_segment_from_words(segment: Segment, words: List[Dict[str, float | str]]) -> dict:
    text = "".join(str(word.get("word", "")) for word in words).strip()
    return {
        "text": text,
        "start": float(segment.start),
        "end": float(segment.end),
        "words": [
            {
                "word": str(word.get("word", "")),
                "start": float(word.get("start", 0.0)),
                "end": float(word.get("end", word.get("start", 0.0))),
                "confidence": float(word.get("confidence", 1.0)),
            }
            for word in words
        ],
    }


def _tokens_to_word_timings(tokens: Sequence[WordToken]) -> List[WordTiming]:
    mapped: List[WordTiming] = []
    for token in tokens:
        mapped.append(
            WordTiming(
                word=token.text,
                start=float(token.start),
                end=float(token.end),
                confidence=token.confidence,
                normalized=token.normalized,
                label=token.label,
            )
        )
    return mapped


def _intervals_to_segments(intervals: Sequence[SuppressedInterval]) -> List[Segment]:
    mapped: List[Segment] = []
    for interval in intervals:
        mapped.append(
            Segment(
                start=float(interval.start),
                end=float(interval.end),
                kind=interval.label,
                label=interval.label,
            )
        )
    return mapped


def _build_word_stream(segments: Iterable[dict]) -> tuple[List[WordToken], List[tuple[int, int]]]:
    tokens: List[WordToken] = []
    spans: List[tuple[int, int]] = []
    for segment in segments:
        start_idx = len(tokens)
        for word in segment.get("words", []) or []:
            w_text = str(word.get("word", ""))
            confidence = float(word.get("confidence", 1.0))
            if _is_junk_token(w_text, confidence):
                continue
            tokens.append(
                WordToken(
                    text=w_text,
                    start=float(word.get("start", 0.0)),
                    end=float(word.get("end", word.get("start", 0.0))),
                    confidence=confidence,
                )
            )
        spans.append((start_idx, len(tokens)))
    return tokens, spans


def _label_fillers(tokens: Sequence[WordToken]) -> None:
    for token in tokens:
        if token.normalized in SAFE_FILLERS:
            token.label = "filler"


def _label_repetitions(tokens: Sequence[WordToken]) -> None:
    for idx in range(1, len(tokens)):
        prev = tokens[idx - 1]
        curr = tokens[idx]
        if curr.normalized and prev.normalized:
            if curr.normalized == prev.normalized and curr.start - prev.end <= STUTTER_MAX_GAP:
                curr.label = "repeat"

    for idx in range(len(tokens)):
        for span in range(2, PHRASE_REPEAT_MAX + 1):
            if idx + 2 * span > len(tokens):
                continue
            first = [tok.normalized for tok in tokens[idx : idx + span]]
            second = [tok.normalized for tok in tokens[idx + span : idx + 2 * span]]
            if not all(first) or not all(second):
                continue
            if first == second:
                for tok in tokens[idx + span : idx + 2 * span]:
                    tok.label = "repeat"


def _collect_intervals(tokens: Sequence[WordToken]) -> List[SuppressedInterval]:
    intervals: List[SuppressedInterval] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.label == "normal":
            idx += 1
            continue
        start = token.start
        end = token.end
        label = token.label
        indices = [idx]
        j = idx + 1
        while j < len(tokens):
            nxt = tokens[j]
            if nxt.label == label and nxt.start - end <= INTERVAL_GAP:
                end = max(end, nxt.end)
                indices.append(j)
                j += 1
            else:
                break
        intervals.append(SuppressedInterval(start=start, end=end, label=label, tokens=indices))
        idx = j
    return intervals


def _slice_intervals(ranges: Sequence[Segment], start: float, end: float) -> List[Segment]:
    sliced: List[Segment] = []
    for rng in ranges:
        overlap_start = max(start, rng.start)
        overlap_end = min(end, rng.end)
        if overlap_end - overlap_start <= 1e-4:
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


def _smart_render_text(words: List[WordTiming]) -> str:
    """Intelligently join words with spaces based on character type (CJK vs Latin)."""
    if not words:
        return ""
    
    parts = []
    for i, w in enumerate(words):
        text = w.word
        # 如果是第一個詞，直接加入
        if i == 0:
            parts.append(text)
            continue
            
        prev_text = words[i-1].word
        
        # 檢查當前詞和前一個詞是否包含 CJK 字符
        is_curr_cjk = any('\u4e00' <= char <= '\u9fff' for char in text)
        is_prev_cjk = any('\u4e00' <= char <= '\u9fff' for char in prev_text)
        
        # 如果兩個都是中文，不加空格；否則加空格
        if is_curr_cjk and is_prev_cjk:
            parts.append(text)
        else:
            parts.append(" " + text)
            
    return "".join(parts)


def _build_character_map(text: str, start: float, end: float) -> List[CharacterTiming]:
    cleaned = [ch for ch in text if not ch.isspace()]
    if not cleaned or end <= start:
        return []
    total = len(cleaned)
    duration = end - start
    step = duration / total
    timeline: List[CharacterTiming] = []
    cursor = start
    for idx, ch in enumerate(cleaned):
        char_start = cursor
        char_end = start + step * (idx + 1) if idx < total - 1 else end
        timeline.append(CharacterTiming(char=ch, start=char_start, end=char_end, pitch=0.0))
        cursor = char_end
    return timeline
