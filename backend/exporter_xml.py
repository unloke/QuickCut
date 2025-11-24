from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree as ET
from xml.dom import minidom

from app.quickcut.models import ClipRegion


def export_to_xml(
    regions: List[ClipRegion],
    output_path: Path,
    media_path: Optional[Path] = None,
    frame_rate: int = 30,
    audio_crossfade_duration: float = 0.5,
) -> Path:
    """
    將 ClipRegion 列表導出為 XML 格式 (FCPXML-like)。

    Args:
        regions: 要導出的片段列表
        output_path: 輸出 XML 檔案路徑
        media_path: 原始媒體檔案路徑 (可選)
        frame_rate: 幀率，用於時間對齊
        audio_crossfade_duration: 音訊交叉淡入淡出持續時間 (秒)

    Returns:
        輸出檔案路徑
    """
    # 過濾可見片段
    visible_regions = [r for r in regions if r.visible and r.duration > 0]

    if not visible_regions:
        raise ValueError("沒有可見的片段可導出")

    # 建立 XML 結構
    root = ET.Element("fcpxml", version="1.9")

    # 資源區段
    resources = ET.SubElement(root, "resources")

    # 添加媒體資源
    if media_path:
        media_id = "r1"
        ET.SubElement(resources, "asset", {
            "id": media_id,
            "name": media_path.name,
            "src": str(media_path),
            "hasVideo": "1",
            "hasAudio": "1",
            "format": "r2"
        })

        # 格式資源
        ET.SubElement(resources, "format", {
            "id": "r2",
            "name": "FFVideoFormat1080p30",
            "frameDuration": f"100/3000s",  # 30 fps
            "width": "1920",
            "height": "1080"
        })

    # 程式庫區段
    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", {"name": "QuickCut Export"})
    project = ET.SubElement(event, "project", {"name": "Timeline"})

    # 序列
    sequence = ET.SubElement(project, "sequence", {
        "format": "r2",
        "tcStart": "0s",
        "tcFormat": "NDF",
        "audioLayout": "stereo",
        "audioRate": "48k"
    })

    # 脊柱 (主時間軸)
    spine = ET.SubElement(sequence, "spine")

    # 將片段添加到脊柱
    timeline_offset = 0.0
    previous_clip = None

    for i, region in enumerate(visible_regions):
        clip = _create_clip_element(region, timeline_offset, frame_rate)

        spine.append(clip)

        # 在相鄰片段之間添加音訊交叉淡入淡出
        if previous_clip is not None and audio_crossfade_duration > 0:
            transition = _create_audio_transition(
                timeline_offset - region.duration / 2, audio_crossfade_duration, frame_rate
            )
            # 插入轉場到前一個片段之後
            spine.insert(-1, transition)

        previous_clip = clip
        timeline_offset += region.duration

    # 寫入檔案
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 美化 XML 輸出
    rough_string = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ", encoding="utf-8")

    # 移除空行
    lines = pretty_xml.decode("utf-8").split("\n")
    non_empty_lines = [line for line in lines if line.strip()]
    final_xml = "\n".join(non_empty_lines).encode("utf-8")

    with open(output_path, "wb") as f:
        f.write(final_xml)

    return output_path


def _create_clip_element(region: ClipRegion, timeline_offset: float, frame_rate: int) -> ET.Element:
    """建立片段 XML 元素"""
    # 將時間轉換為幀數並對齊幀率
    start_frame = _time_to_frame(region.start, frame_rate)
    end_frame = _time_to_frame(region.end, frame_rate)
    duration_frames = end_frame - start_frame

    # 時間軸偏移
    offset_frame = _time_to_frame(timeline_offset, frame_rate)

    clip = ET.Element("asset-clip", {
        "name": region.text[:50] if region.text else f"Clip_{region.region_id[:8]}",
        "offset": _frames_to_fcp_timecode(offset_frame, frame_rate),
        "duration": _frames_to_fcp_timecode(duration_frames, frame_rate),
        "tcFormat": "NDF",
        "audioRole": "dialogue",
        "videoRole": "video"
    })

    # 來源範圍
    start_timecode = _frames_to_fcp_timecode(start_frame, frame_rate)
    duration_timecode = _frames_to_fcp_timecode(duration_frames, frame_rate)

    video = ET.SubElement(clip, "video")
    ET.SubElement(video, "video", {
        "offset": "0s",
        "duration": duration_timecode,
        "srcIn": start_timecode,
        "srcOut": _frames_to_fcp_timecode(end_frame, frame_rate)
    })

    audio = ET.SubElement(clip, "audio")
    ET.SubElement(audio, "audio", {
        "offset": "0s",
        "duration": duration_timecode,
        "srcIn": start_timecode,
        "srcOut": _frames_to_fcp_timecode(end_frame, frame_rate),
        "role": "dialogue"
    })

    return clip


def _create_audio_transition(offset: float, duration: float, frame_rate: int) -> ET.Element:
    """建立音訊交叉淡入淡出轉場"""
    # 轉場從前一個片段結束前開始
    offset_frame = _time_to_frame(offset - duration, frame_rate)
    duration_frames = math.ceil(duration * frame_rate)

    transition = ET.Element("transition", {
        "name": "Audio Crossfade",
        "offset": _frames_to_fcp_timecode(offset_frame, frame_rate),
        "duration": _frames_to_fcp_timecode(duration_frames, frame_rate)
    })

    return transition


def _time_to_frame(time_seconds: float, frame_rate: int) -> int:
    """將秒數轉換為幀數，進行幀率對齊"""
    return round(time_seconds * frame_rate)


def _frames_to_timecode(frames: int, frame_rate: int) -> str:
    """將幀數轉換為時間碼格式"""
    total_seconds = frames / frame_rate
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    frames_remainder = frames % frame_rate

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames_remainder:02d}"


def _frames_to_fcp_timecode(frames: int, frame_rate: int) -> str:
    """將幀數轉換為 FCP 時間格式 (如 100/3000s)"""
def export_to_edl(
    regions: List[ClipRegion],
    output_path: Path,
    media_path: Optional[Path] = None,
    frame_rate: int = 30,
) -> Path:
    """
    將 ClipRegion 列表導出為 EDL 格式。

    Args:
        regions: 要導出的片段列表
        output_path: 輸出 EDL 檔案路徑
        media_path: 原始媒體檔案路徑 (可選)
        frame_rate: 幀率，用於時間對齊

    Returns:
        輸出檔案路徑
    """
    # 過濾可見片段
    visible_regions = [r for r in regions if r.visible and r.duration > 0]

    if not visible_regions:
        raise ValueError("沒有可見的片段可導出")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # EDL 標題
        f.write("TITLE: QuickCut Export\n")
        f.write("FCM: NON-DROP FRAME\n\n")

        # 為每個片段寫入 EDL 條目
        timeline_offset = 0.0
        for i, region in enumerate(visible_regions):
            edit_number = f"{i+1:03d}"
            source_name = media_path.name if media_path else "Source"

            # 將時間轉換為時間碼
            src_in = _frames_to_timecode(_time_to_frame(region.start, frame_rate), frame_rate)
            src_out = _frames_to_timecode(_time_to_frame(region.end, frame_rate), frame_rate)
            rec_in = _frames_to_timecode(_time_to_frame(timeline_offset, frame_rate), frame_rate)
            rec_out = _frames_to_timecode(_time_to_frame(timeline_offset + region.duration, frame_rate), frame_rate)

            # EDL 行格式: 編輯號 轉場 軌道 來源 入點 出點 記錄入點 出點
            f.write(f"{edit_number}  AX       V     C        {src_in} {src_out} {rec_in} {rec_out}\n")

            # 添加來源檔案註釋
            f.write(f"* FROM CLIP NAME: {region.text[:50] if region.text else f'Clip_{region.region_id[:8]}'}\n")
            f.write(f"* SOURCE FILE: {source_name}\n\n")

            timeline_offset += region.duration

    return output_path
    return f"{frames * 100}/{frame_rate * 100}s"