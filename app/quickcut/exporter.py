from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from .models import ClipRegion


def export_visible_segments(
    media_path: Path,
    regions: Iterable[ClipRegion],
    output_path: Path,
    preset: str = "veryfast",
    crf: int = 18,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    """Invokes FFmpeg to export only the visible segments."""

    ranges: List[Tuple[float, float]] = []
    for region in regions:
        if region.visible and region.duration > 0:
            ranges.append((region.start, region.end))

    if not ranges:
        raise RuntimeError("沒有任何可輸出的片段。")

    merged = _merge_ranges(sorted(ranges, key=lambda pair: pair[0]))
    select_expr = "+".join(f"between(t,{start:.3f},{end:.3f})" for start, end in merged)
    filter_complex = (
        f"[0:v]select='{select_expr}',setpts=N/FRAME_RATE/TB[v];"
        f"[0:a]aselect='{select_expr}',asetpts=N/SR/TB[a]"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(media_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_path),
    ]

    if progress_cb:
        progress_cb("FFmpeg 正在輸出影片…")
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"輸出失敗:\n{completed.stderr.strip()}")
    return output_path


def _merge_ranges(ranges: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not ranges:
        return []
    merged = [list(ranges[0])]
    for start, end in ranges[1:]:
        last = merged[-1]
        if start <= last[1] + 1e-3:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]
