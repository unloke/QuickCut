from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set
import uuid


def _region_id() -> str:
    return uuid.uuid4().hex


@dataclass
class QuickEditSettings:
    pre_roll_ms: int = 120
    post_roll_ms: int = 150
    silence_threshold_db: float = -42.0
    low_energy_threshold_db: float = -34.0
    min_silence: float = 0.30
    min_speech: float = 0.15
    window_ms: float = 25.0
    hop_ms: float = 10.0
    remove_fillers: bool = True
    remove_repeated: bool = True
    add_audio_crossfades: bool = False

    def pre_roll_seconds(self) -> float:
        return self.pre_roll_ms / 1000.0

    def post_roll_seconds(self) -> float:
        return self.post_roll_ms / 1000.0


@dataclass
class Segment:
    start: float
    end: float
    kind: str = "speech"
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class ClipRegion:
    region_id: str = field(default_factory=_region_id)
    start: float = 0.0
    end: float = 0.0
    kind: str = "speech"
    text: str = ""
    visible: bool = True
    tags: Set[str] = field(default_factory=set)
    auto_hide_reasons: Set[str] = field(default_factory=set)
    user_override: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "ClipRegion":
        return ClipRegion(
            region_id=_region_id(),
            start=self.start,
            end=self.end,
            kind=self.kind,
            text=self.text,
            visible=self.visible,
            tags=set(self.tags),
            auto_hide_reasons=set(self.auto_hide_reasons),
            user_override=self.user_override,
            metadata=dict(self.metadata),
        )

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def apply_auto_reason(self, reason: str, hide: bool) -> None:
        if hide:
            self.auto_hide_reasons.add(reason)
        else:
            self.auto_hide_reasons.discard(reason)
        if not self.user_override:
            self.visible = not self.auto_hide_reasons

    def clear_auto_reason(self, reason: str) -> None:
        self.auto_hide_reasons.discard(reason)
        if not self.auto_hide_reasons and not self.user_override:
            self.visible = True

    def toggle_user_visibility(self) -> None:
        self.user_override = True
        self.visible = not self.visible


@dataclass
class AnalysisResult:
    media_path: Path
    cache_dir: Path
    wav_path: Path
    speech_wav_path: Path
    sample_rate: int
    duration: float
    envelope: List[tuple[float, float]]
    speech_segments: List[Segment]
    silence_segments: List[Segment]
    low_segments: List[Segment]
    mask_suggestions: List[Segment]
    speech_time_map: List[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class CharacterTiming:
    char: str
    start: float
    end: float
    pitch: float = 0.0


@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    confidence: float = 1.0
    normalized: str = ""
    label: str = "normal"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def render_text_from_words(words: Sequence[WordTiming]) -> str:
    display: List[str] = []
    for word in words:
        chunk = (word.word or "").strip()
        if chunk:
            display.append(chunk)
    return " ".join(display)


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    is_filler: bool = False
    is_repeat: bool = False
    characters: List[CharacterTiming] = field(default_factory=list)
    words: List[WordTiming] = field(default_factory=list)
    suppressed_ranges: List[Segment] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def slice(self, start: float, end: float) -> "TranscriptSegment | None":
        start = max(self.start, start)
        end = min(self.end, end)
        if end <= start:
            return None

        sliced_chars: List[CharacterTiming] = []
        for char in self.characters:
            if char.end <= start or char.start >= end:
                continue
            sliced_chars.append(
                CharacterTiming(
                    char=char.char,
                    start=max(char.start, start),
                    end=min(char.end, end),
                    pitch=char.pitch,
                )
            )

        sliced_words: List[WordTiming] = []
        for word in self.words:
            if word.end <= start or word.start >= end:
                continue
            sliced_words.append(
                WordTiming(
                    word=word.word,
                    start=max(word.start, start),
                    end=min(word.end, end),
                    confidence=word.confidence,
                    normalized=word.normalized,
                    label=word.label,
                )
            )

        word_text = render_text_from_words(sliced_words)
        text = word_text

        sliced_ranges: List[Segment] = []
        for rng in self.suppressed_ranges:
            overlap_start = max(rng.start, start)
            overlap_end = min(rng.end, end)
            if overlap_end - overlap_start <= 1e-3:
                continue
            sliced_ranges.append(
                Segment(
                    start=overlap_start,
                    end=overlap_end,
                    kind=rng.kind,
                    label=rng.label,
                    meta=dict(rng.meta or {}),
                )
            )

        return TranscriptSegment(
            start=start,
            end=end,
            text=text,
            is_filler=self.is_filler,
            is_repeat=self.is_repeat,
            characters=sliced_chars,
            words=sliced_words,
            suppressed_ranges=sliced_ranges,
        )


@dataclass
class TranscriptionResult:
    segments: List[TranscriptSegment]
    full_text: str


@dataclass
class ProjectState:
    analysis: AnalysisResult
    transcription: Optional[TranscriptionResult] = None
    regions: List[ClipRegion] = field(default_factory=list)
    settings: QuickEditSettings = field(default_factory=QuickEditSettings)

    def save_project(self, json_path: str):
        def serialize_region(r: ClipRegion) -> dict:
            d = dataclasses.asdict(r)
            d['tags'] = list(d['tags'])
            d['auto_hide_reasons'] = list(d['auto_hide_reasons'])
            return d
        data = {
            "regions": [serialize_region(r) for r in self.regions],
            "settings": dataclasses.asdict(self.settings),
            "media_path": str(self.analysis.media_path),
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_project(cls, json_path: str) -> tuple[List[ClipRegion], QuickEditSettings, Path]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        regions = []
        for r in data['regions']:
            r['tags'] = set(r['tags'])
            r['auto_hide_reasons'] = set(r['auto_hide_reasons'])
            regions.append(ClipRegion(**r))
        settings = QuickEditSettings(**data['settings'])
        media_path = Path(data['media_path'])
        return regions, settings, media_path
