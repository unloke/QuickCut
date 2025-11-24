from __future__ import annotations

from typing import Callable, Iterable, List

from .models import AnalysisResult, ClipRegion, QuickEditSettings, Segment, TranscriptSegment, TranscriptionResult

HALLUCINATION_STOPWORDS = {"a", "the", "it", "is", "to", "of", "that", "and", "but", "or"}


def build_regions(
    analysis: AnalysisResult,
    transcription: TranscriptionResult | None,
    settings: QuickEditSettings,
) -> List[ClipRegion]:
    regions = _build_base_regions(analysis)
    regions = _apply_mask_suggestions(regions, analysis.mask_suggestions)

    if transcription:
        regions = _apply_transcripts(regions, transcription)
        regions = _hide_empty_speech(regions)
        regions = _apply_auto_tags(regions, "filler", settings.remove_fillers)
        regions = _apply_auto_tags(regions, "repeat", settings.remove_repeated)

    regions = _prune_hallucinations(regions)
    return _merge_regions(regions)


def _build_base_regions(analysis: AnalysisResult) -> List[ClipRegion]:
    regions: List[ClipRegion] = []
    cursor = 0.0
    tolerance = 1e-4
    for seg in sorted(analysis.speech_segments, key=lambda s: s.start):
        if seg.start - cursor > tolerance:
            regions.append(
                ClipRegion(
                    start=cursor,
                    end=seg.start,
                    kind="silence",
                    visible=False,
                    tags={"silence"},
                    auto_hide_reasons={"silence"},
                )
            )
        seg_start = max(cursor, seg.start)
        seg_end = max(seg_start, seg.end)
        if seg_end - seg_start <= tolerance:
            cursor = max(cursor, seg.end)
            continue
        regions.append(
            ClipRegion(
                start=seg_start,
                end=seg_end,
                kind="speech",
                visible=True,
                tags={"speech"},
            )
        )
        cursor = seg_end

    if cursor < analysis.duration:
        regions.append(
            ClipRegion(
                start=cursor,
                end=analysis.duration,
                kind="silence",
                visible=False,
                tags={"silence"},
                auto_hide_reasons={"silence"},
            )
        )
    return regions


def _apply_mask_suggestions(regions: List[ClipRegion], suggestions: Iterable[Segment]) -> List[ClipRegion]:
    updated = regions

    def hide_region(region: ClipRegion, reason: str) -> ClipRegion:
        region.tags.add(reason)
        region.apply_auto_reason(reason, hide=True)
        return region

    for suggestion in suggestions:
        reason = suggestion.kind or "silence"

        def mutate(target: ClipRegion) -> ClipRegion:
            if reason == "silence" and target.kind == "speech":
                return target
            return hide_region(target, reason)

        updated = _split_and_mutate(updated, suggestion.start, suggestion.end, mutate)
    return updated


def _apply_transcripts(regions: List[ClipRegion], transcription: TranscriptionResult) -> List[ClipRegion]:
    transcript_segments = transcription.segments
    if not transcript_segments:
        return regions

    result: List[ClipRegion] = []

    for region in regions:
        if region.kind != "speech":
            result.append(region)
            continue

        overlaps = [
            seg for seg in transcript_segments if _overlaps(region.start, region.end, seg.start, seg.end)
        ]
        if not overlaps:
            result.append(region)
            continue

        cursor = region.start
        for seg in overlaps:
            overlap_start = max(region.start, seg.start)
            overlap_end = min(region.end, seg.end)
            if overlap_end <= overlap_start:
                continue
            if overlap_start > cursor:
                filler = region.clone()
                filler.start = cursor
                filler.end = overlap_start
                filler.text = ""
                result.append(filler)
            chunk = region.clone()
            chunk.start = overlap_start
            chunk.end = overlap_end
            chunk.tags.add("speech")
            transcript_slice = seg.slice(chunk.start, chunk.end)
            if transcript_slice:
                chunk.metadata["transcript"] = transcript_slice
                avg_conf = _average_confidence(transcript_slice)
                if avg_conf is not None:
                    chunk.metadata["avg_confidence"] = avg_conf
                else:
                    chunk.metadata.pop("avg_confidence", None)
            else:
                chunk.metadata.pop("transcript", None)
                chunk.metadata.pop("avg_confidence", None)
                chunk.text = ""
            result.extend(_split_chunk_by_transcript(chunk, transcript_slice))
            cursor = overlap_end
        if cursor < region.end:
            tail = region.clone()
            tail.start = cursor
            tail.end = region.end
            tail.text = ""
            tail.metadata.pop("avg_confidence", None)
            result.append(tail)

    return result


def _split_chunk_by_transcript(chunk: ClipRegion, transcript: TranscriptSegment | None) -> List[ClipRegion]:
    if not transcript:
        chunk.metadata.pop("transcript", None)
        chunk.metadata.pop("avg_confidence", None)
        chunk.text = ""
        return [chunk]
    chunk.metadata["transcript"] = transcript
    avg_conf = _average_confidence(transcript)
    if avg_conf is not None:
        chunk.metadata["avg_confidence"] = avg_conf
    else:
        chunk.metadata.pop("avg_confidence", None)
    chunk.text = transcript.text
    pieces: List[ClipRegion] = []
    cursor = chunk.start
    for rng in transcript.suppressed_ranges:
        if rng.kind not in {"filler", "stutter", "repeat"}:
            continue
        sup_start = max(cursor, max(chunk.start, rng.start))
        sup_end = min(chunk.end, rng.end)
        if sup_end <= sup_start:
            continue
        if sup_start > cursor:
            keep = _clone_with_transcript_slice(chunk, transcript, cursor, sup_start)
            if keep:
                pieces.append(keep)
        hidden = _clone_with_transcript_slice(chunk, transcript, sup_start, sup_end, reason=rng.kind)
        if hidden:
            pieces.append(hidden)
        cursor = sup_end

    if cursor < chunk.end:
        tail = _clone_with_transcript_slice(chunk, transcript, cursor, chunk.end)
        if tail:
            pieces.append(tail)

    return pieces


def _clone_with_transcript_slice(
    template: ClipRegion,
    transcript: TranscriptSegment,
    start: float,
    end: float,
    reason: str | None = None,
) -> ClipRegion | None:
    if end - start <= 1e-4:
        return None
    clone = template.clone()
    clone.start = start
    clone.end = end
    sub_transcript = transcript.slice(start, end)
    if sub_transcript:
        clone.metadata["transcript"] = sub_transcript
        clone.text = sub_transcript.text
        avg_conf = _average_confidence(sub_transcript)
        if avg_conf is not None:
            clone.metadata["avg_confidence"] = avg_conf
        else:
            clone.metadata.pop("avg_confidence", None)
    else:
        clone.metadata.pop("transcript", None)
        clone.metadata.pop("avg_confidence", None)
        clone.text = ""
    if reason:
        clone.tags.add(reason)
        auto_tag = _auto_reason_from_kind(reason)
        clone.tags.add(auto_tag)
        clone.apply_auto_reason(auto_tag, hide=True)
    return clone


def _auto_reason_from_kind(kind: str) -> str:
    if kind == "filler":
        return "filler"
    if kind in {"stutter", "repeat"}:
        return "repeat"
    return kind or "mask"


def _apply_auto_tags(regions: List[ClipRegion], tag: str, enabled: bool) -> List[ClipRegion]:
    for region in regions:
        if tag in region.tags and region.kind == "speech":
            if enabled:
                region.apply_auto_reason(tag, hide=True)
            else:
                region.clear_auto_reason(tag)
    return regions


def _split_and_mutate(
    regions: List[ClipRegion],
    start: float,
    end: float,
    mutate: Callable[[ClipRegion], ClipRegion],
) -> List[ClipRegion]:
    if start >= end:
        return regions
    result: List[ClipRegion] = []
    for region in regions:
        if region.end <= start or region.start >= end:
            result.append(region)
            continue

        if region.start < start:
            left = region.clone()
            left.end = start
            if left.duration > 1e-6:
                result.append(left)
        middle = region.clone()
        middle.start = max(region.start, start)
        middle.end = min(region.end, end)
        if middle.end > middle.start:
            result.append(mutate(middle))
        if region.end > end:
            right = region.clone()
            right.start = end
            if right.duration > 1e-6:
                result.append(right)
        else:
            continue

    return _merge_regions(result)


def _hide_empty_speech(regions: List[ClipRegion]) -> List[ClipRegion]:
    result: List[ClipRegion] = []
    for region in regions:
        if region.kind == "speech" and region.duration < 2.0 and not region.text.strip():
            region.tags.add("empty_speech")
            region.apply_auto_reason("empty_speech", hide=True)
        result.append(region)
    return result


def _average_confidence(transcript: TranscriptSegment | None) -> float | None:
    if not transcript or not transcript.words:
        return None
    confidences = [max(0.0, word.confidence) for word in transcript.words]
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def _prune_hallucinations(regions: List[ClipRegion]) -> List[ClipRegion]:
    pruned: List[ClipRegion] = []
    for region in regions:
        if region.kind != "speech" or not region.text.strip():
            pruned.append(region)
            continue
        duration = region.duration
        normalized = region.text.strip().lower()
        words = [w for w in normalized.split() if w]
        avg_conf = region.metadata.get("avg_confidence")
        reason: str | None = None
        if duration < 0.4 and avg_conf is not None and avg_conf < 0.7:
            reason = "low_confidence"
        elif duration < 0.3 and len(words) == 1 and words[0] in HALLUCINATION_STOPWORDS:
            reason = "noise"
        elif not any(ch.isalnum() for ch in region.text):
            reason = "noise"
        if reason:
            region.tags.add(reason)
            region.apply_auto_reason(reason, hide=True)
        pruned.append(region)
    return pruned


def _merge_regions(regions: List[ClipRegion]) -> List[ClipRegion]:
    if not regions:
        return []
    merged: List[ClipRegion] = [regions[0]]
    for region in regions[1:]:
        last = merged[-1]
        if (
            abs(last.end - region.start) < 1e-4
            and last.kind == region.kind
            and last.visible == region.visible
            and last.tags == region.tags
            and last.auto_hide_reasons == region.auto_hide_reasons
            and last.user_override == region.user_override
            and last.text == region.text
        ):
            last.end = region.end
        else:
            merged.append(region)
    return merged


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)
