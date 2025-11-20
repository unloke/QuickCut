from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path

# Ensure Qt uses the Media Foundation backend before importing PyQt modules.
os.environ.setdefault("QT_MULTIMEDIA_PREFERRED_PLUGINS", "windowsmediafoundation")

from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

from .exporter import export_visible_segments
from .models import (
    AnalysisResult,
    ClipRegion,
    ProjectState,
    QuickEditSettings,
    TranscriptSegment,
    TranscriptionResult,
)
from .pipeline import AnalyzerPipeline
from .timeline_builder import build_regions
from .transcription import TranscriptionEngine
from .widgets.media_bin import MediaBinWidget
from .widgets.preview import PreviewPanel
from .widgets.quick_panel import QuickEditPanel
from .widgets.timeline import TimelineWidget
from .workers import AnalyzerTask


APP_STYLESHEET = """
QWidget {
    background-color: #050505;
    color: #f5f5f5;
    font-family: "Segoe UI", "Microsoft JhengHei", sans-serif;
}
QFrame#quickPanel,
QFrame#previewPanel,
QFrame#mediaBin {
    background-color: #1b1b1b;
    border: 1px solid #2f2f2f;
    border-radius: 10px;
}
QSplitter::handle {
    background-color: #1c1c1c;
}
QPushButton {
    background-color: #2b2b2b;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    padding: 6px 12px;
    color: #f5f5f5;
}
QPushButton:hover {
    background-color: #3c3c3c;
}
QPushButton:pressed {
    background-color: #1f1f1f;
}
QSlider::groove:horizontal {
    border: 1px solid #333333;
    height: 6px;
    background: #2a2a2a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #f5f5f5;
    border: 1px solid #050505;
    width: 12px;
    margin: -5px 0;
    border-radius: 6px;
}
QStatusBar {
    background-color: #080808;
    border-top: 1px solid #1f1f1f;
}
"""


def _discover_analyzer_binary() -> Path:
    base_dir = Path(__file__).resolve().parents[2]
    candidates = [
        base_dir / "backend" / "audio_analyzer" / "build" / "Release" / "audio_analyzer.exe",
        base_dir / "backend" / "audio_analyzer" / "build" / "audio_analyzer.exe",
        base_dir / "backend" / "audio_analyzer" / "build" / "audio_analyzer",
        base_dir / "backend" / "audio_analyzer" / "audio_analyzer.exe",
        base_dir / "backend" / "audio_analyzer" / "audio_analyzer",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("找不到 audio_analyzer，可先執行 CMake 建置。")


class QuickCutWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuickCut Whisper")
        self.resize(1280, 780)

        analyzer_path = _discover_analyzer_binary()
        self.pipeline = AnalyzerPipeline(analyzer_path)
        self.transcriber = TranscriptionEngine("tiny")
        self.project: ProjectState | None = None
        self.current_media_path: Path | None = None
        self.current_settings = QuickEditSettings()
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(1)
        self._active_tasks: list[AnalyzerTask] = []
        self._has_manual_edits = False
        self._using_placeholder_project = False
        self._visible_ranges: list[tuple[float, float]] = []
        self._skip_hidden_seek = False
        self._syncing_timeline_scroll = False

        self.media_bin = MediaBinWidget()
        self.quick_panel = QuickEditPanel()
        self.quick_panel.set_analyze_enabled(False)
        self.preview = PreviewPanel()
        self.timeline = TimelineWidget()
        self.preview.positionUpdated.connect(self._handle_preview_position)
        self.preview.durationUpdated.connect(self._handle_preview_duration)
        self.preview.errorOccurred.connect(self._handle_preview_error)
        self.timeline.viewChanged.connect(self._sync_timeline_scrollbar)
        self.status_bar = QStatusBar()
        self.timeline_controls = self._build_timeline_controls()

        container = QWidget()
        main_layout = QVBoxLayout(container)
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        left_layout.addWidget(self.media_bin)
        left_layout.addWidget(self.quick_panel)
        left_layout.addStretch(1)
        top_splitter.addWidget(left_panel)
        top_splitter.addWidget(self.preview)
        top_splitter.setStretchFactor(0, 0)
        top_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(top_splitter, stretch=1)
        main_layout.addWidget(self.timeline_controls, stretch=0)
        main_layout.addWidget(self.timeline, stretch=0)
        self.setCentralWidget(container)
        self.setStatusBar(self.status_bar)

        self.media_bin.mediaSelected.connect(self._handle_media_selected)
        self.quick_panel.settingsChanged.connect(self._handle_settings_changed)
        self.quick_panel.analyzeRequested.connect(self._handle_manual_analyze)
        self.quick_panel.exportRequested.connect(self._handle_export)
        self.timeline.timeSelected.connect(self._seek_preview)
        self.timeline.regionToggled.connect(self._toggle_region)
        self.timeline.regionSelected.connect(self._handle_region_selected_from_timeline)

        self._set_status("尚未載入媒體。")
        self._update_editing_actions()

    def _set_status(self, message: str):
        self.status_bar.showMessage(message)

    def _handle_media_selected(self, path: str):
        self.current_media_path = Path(path)
        self.preview.load_media(self.current_media_path)
        self.project = None
        self._has_manual_edits = False
        self._using_placeholder_project = False
        self._skip_hidden_seek = False
        self._visible_ranges = []
        self.timeline.set_regions([], 0.0)
        self.timeline.set_envelope([])
        self.quick_panel.set_analyze_enabled(True)
        self._update_editing_actions()
        self._set_status(f"載入 {self.current_media_path.name}，請調整設定後按「分析」。")

    def _handle_settings_changed(self, settings: QuickEditSettings):
        self.current_settings = settings
        if self._has_manual_edits and self.project:
            self._set_status("設定已變更，請按「分析」重新套用至手動編輯後的片段。")
            return
        if self.project and self.project.analysis and self.project.transcription:
            self.project.regions = build_regions(
                self.project.analysis,
                self.project.transcription,
                self.current_settings,
            )
            self.timeline.set_regions(self.project.regions, self.project.analysis.duration)
            self._refresh_preview_ranges()
            self._update_editing_actions()

    def _handle_manual_analyze(self, settings: QuickEditSettings):
        if not self.current_media_path:
            QMessageBox.warning(self, "尚未載入媒體", "請先選擇或拖曳影片。")
            return
        self.current_settings = settings
        self._run_analysis(settings)

    def _run_analysis(self, settings: QuickEditSettings):
        if not self.current_media_path:
            return
        self._has_manual_edits = False
        self._using_placeholder_project = False
        self.quick_panel.set_analyze_enabled(False)
        task = AnalyzerTask(
            self.pipeline,
            self.transcriber,
            self.current_media_path,
            settings,
        )
        task.signals.progress.connect(self._set_status)
        task.signals.error.connect(partial(self._handle_task_error, task))
        task.signals.finished.connect(partial(self._handle_task_finished, task))
        self._set_status("開始分析…")
        self._active_tasks.append(task)
        self.thread_pool.start(task)

    def _finalize_task(self, task: AnalyzerTask):
        if task in self._active_tasks:
            self._active_tasks.remove(task)

    def _handle_task_finished(self, task: AnalyzerTask, payload: dict):
        self._finalize_task(task)
        analysis: AnalysisResult = payload["analysis"]
        transcription: TranscriptionResult = payload["transcription"]
        settings: QuickEditSettings = payload["settings"]
        self.current_settings = settings
        regions = build_regions(analysis, transcription, settings)
        self.project = ProjectState(analysis=analysis, transcription=transcription, regions=regions)
        self.timeline.set_regions(regions, analysis.duration)
        self.timeline.set_envelope(analysis.envelope)
        self._has_manual_edits = False
        self._using_placeholder_project = False
        self._refresh_preview_ranges()
        self._update_editing_actions()
        self.quick_panel.set_analyze_enabled(True)
        self._set_status("分析完成，可雙擊遮罩/顯示片段。")

    def _handle_task_error(self, task: AnalyzerTask, message: str):
        self._finalize_task(task)
        QMessageBox.critical(self, "分析錯誤", message)
        self.quick_panel.set_analyze_enabled(True)
        self._set_status("分析失敗。")

    def _toggle_region(self, region_id: str):
        if not self.project:
            return
        updated = False
        for region in self.project.regions:
            if region.region_id == region_id:
                # 防止隱藏預留片段
                if "placeholder" in region.tags and region.visible:
                    continue  # 不允許隱藏預留片段
                region.toggle_user_visibility()
                updated = True
                break
        if updated:
            self._mark_manual_edit("已切換片段顯示狀態。")
            self.timeline.set_regions(
                self.project.regions,
                self.project.analysis.duration,
                preserve_view=True,
            )
            self.timeline.set_selected_region(region.region_id)
            self._handle_region_selected(region.region_id, announce=False)
            self._refresh_preview_ranges()
            self._update_editing_actions()

    def _seek_preview(self, seconds: float):
        self.preview.set_playhead(seconds)

    def _handle_export(self):
        if not self.project or not self.project.regions:
            QMessageBox.information(self, "無內容", "請先完成分析。")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "輸出影片",
            str(self.project.analysis.media_path.with_stem(self.project.analysis.media_path.stem + "_quickcut")),
            "MP4 (*.mp4)",
        )
        if not save_path:
            return
        try:
            export_visible_segments(
                self.project.analysis.media_path,
                self.project.regions,
                Path(save_path),
                progress_cb=self._set_status,
            )
            self._set_status("輸出完成。")
            QMessageBox.information(self, "完成", f"已輸出到 {save_path}")
        except Exception as exc:
            QMessageBox.critical(self, "輸出失敗", str(exc))
            self._set_status("輸出失敗。")

    def _handle_preview_duration(self, duration: float):
        if not self.current_media_path or duration <= 0:
            return
        if self.project and not self._using_placeholder_project:
            return
        if self.project and self._using_placeholder_project:
            if abs(self.project.analysis.duration - duration) < 1e-3:
                # 確保預留片段始終顯示
                for region in self.project.regions:
                    if "placeholder" in region.tags:
                        region.visible = True
                        region.user_override = True
                        region.auto_hide_reasons.clear()
                self.timeline.set_regions(self.project.regions, duration)
                self._refresh_preview_ranges()
                self._update_editing_actions()
                return
            if self._has_manual_edits:
                return
        self._seed_basic_timeline(duration)

    def _seed_basic_timeline(self, duration: float):
        if duration <= 0 or not self.current_media_path:
            return
        placeholder_analysis = AnalysisResult(
            media_path=self.current_media_path,
            cache_dir=self.current_media_path.parent,
            wav_path=self.current_media_path,
            speech_wav_path=self.current_media_path,
            sample_rate=0,
            duration=duration,
            envelope=[],
            speech_segments=[],
            silence_segments=[],
            low_segments=[],
            mask_suggestions=[],
            speech_time_map=[],
        )
        placeholder_region = ClipRegion(
            start=0.0,
            end=duration,
            kind="speech",
            text="",
            visible=True,
            tags={"speech", "placeholder"},
            user_override=True,
        )
        self.project = ProjectState(
            analysis=placeholder_analysis,
            transcription=None,
            regions=[placeholder_region],
        )
        self._using_placeholder_project = True
        self.timeline.set_regions(self.project.regions, duration)
        self.timeline.set_envelope([])
        self._refresh_preview_ranges()
        self._update_editing_actions()

    def _refresh_preview_ranges(self):
        if not self.project:
            self._visible_ranges = []
            return
        ranges: list[tuple[float, float]] = []
        for region in self.project.regions:
            if region.visible and region.end - region.start > 1e-3:
                ranges.append((region.start, region.end))
        ranges.sort(key=lambda pair: pair[0])
        self._visible_ranges = self._merge_ranges(ranges)

    def _handle_preview_position(self, seconds: float):
        self.timeline.set_playhead(seconds)
        if self._skip_hidden_seek:
            self._skip_hidden_seek = False
            return
        if not self.project or not self.preview.is_playing():
            return
        if not self._visible_ranges:
            return
        if self._is_time_visible(seconds):
            return
        next_visible = self._next_visible_time(seconds)
        if next_visible is None:
            self.preview.pause()
            self._set_status("已播放到最後的顯示片段。")
        else:
            self._skip_hidden_seek = True
            self.preview.set_playhead(next_visible)

    def _handle_preview_error(self, message: str):
        feedback = message or "預覽播放發生未知錯誤。"
        self._set_status(f"預覽錯誤：{feedback}")

    def _is_time_visible(self, seconds: float) -> bool:
        epsilon = 1e-3
        for start, end in self._visible_ranges:
            if start - epsilon <= seconds <= end + epsilon:
                return True
        return False

    def _next_visible_time(self, seconds: float) -> float | None:
        epsilon = 1e-3
        for start, end in self._visible_ranges:
            if start - epsilon <= seconds < end - epsilon:
                return seconds
            if seconds < start - epsilon:
                return start
        return None

    @staticmethod
    def _merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not ranges:
            return []
        merged: list[list[float]] = [[ranges[0][0], ranges[0][1]]]
        for start, end in ranges[1:]:
            if end <= start:
                continue
            last = merged[-1]
            if start <= last[1] + 1e-3:
                last[1] = max(last[1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    def _sync_timeline_scrollbar(self, start: float, span: float, total: float):
        if not hasattr(self, "timeline_scroll"):
            return
        self._syncing_timeline_scroll = True
        try:
            if total <= 0 or span >= total:
                self.timeline_scroll.setEnabled(False)
                self.timeline_scroll.setValue(0)
            else:
                max_start = total - span
                ratio = 0.0 if max_start <= 0 else max(0.0, min(1.0, start / max_start))
                self.timeline_scroll.setEnabled(True)
                self.timeline_scroll.setValue(int(ratio * 1000))
        finally:
            self._syncing_timeline_scroll = False

    def _handle_timeline_scroll_changed(self, value: int):
        if self._syncing_timeline_scroll:
            return
        self.timeline.set_view_start_ratio(value / 1000.0)

    def _build_timeline_controls(self) -> QWidget:
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(QLabel("片段編輯："))
        self.cut_button = QPushButton("切割 (Ctrl+K)")
        self.toggle_visibility_button = QPushButton("切換顯示 (Ctrl+Shift+T)")

        layout.addWidget(self.cut_button)
        layout.addWidget(self.toggle_visibility_button)
        layout.addStretch(1)

        layout.addWidget(QLabel("時間軸滑動："))
        self.timeline_scroll = QSlider(Qt.Orientation.Horizontal)
        self.timeline_scroll.setRange(0, 1000)
        self.timeline_scroll.setEnabled(False)
        self.timeline_scroll.valueChanged.connect(self._handle_timeline_scroll_changed)
        layout.addWidget(self.timeline_scroll, stretch=1)
        self._sync_timeline_scrollbar(0.0, 0.0, 0.0)

        self.cut_shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
        self.toggle_visibility_shortcut = QShortcut(QKeySequence("Ctrl+Shift+T"), self)

        self.cut_button.clicked.connect(self._cut_at_playhead)
        self.toggle_visibility_button.clicked.connect(self._toggle_selected_visibility)
        self.cut_shortcut.activated.connect(self._cut_at_playhead)
        self.toggle_visibility_shortcut.activated.connect(self._toggle_selected_visibility)
        return toolbar

    def _cut_at_playhead(self):
        if not self.project:
            return
        region = self._get_selected_region()
        if not region:
            self._set_status("請先在時間軸上選擇要切割的片段。")
            return
        time = self.timeline.playhead()
        if not (region.start < time < region.end):
            self._set_status("播放頭必須位於片段範圍內才能切割。")
            return
        epsilon = max(0.02, region.duration * 0.02)
        if (time - region.start) < epsilon or (region.end - time) < epsilon:
            self._set_status("切割點過於靠近邊界，請往內移動一點。")
            return

        original_end = region.end
        original_duration = region.duration
        new_region = region.clone()
        region.end = time
        new_region.start = time
        new_region.end = original_end

        if not self._apply_transcript_split(region, new_region, time):
            left_text, right_text = self._split_text_by_ratio(
                region.text,
                original_duration,
                time - region.start,
            )
            region.text = left_text
            new_region.text = right_text
            region.metadata.pop("transcript", None)
            new_region.metadata.pop("transcript", None)

        idx = self.project.regions.index(region)
        self.project.regions.insert(idx + 1, new_region)

        self._mark_manual_edit("已切割片段。")
        self.timeline.set_regions(
            self.project.regions,
            self.project.analysis.duration,
            preserve_view=True,
        )
        self.timeline.set_selected_region(new_region.region_id)
        self._handle_region_selected(new_region.region_id, sync_preview=True)
        self._refresh_preview_ranges()

    def _set_selection_visibility(self, visible: bool):
        if not self.project:
            return
        region = self._get_selected_region()
        if not region:
            self._set_status("請先選擇要切換顯示的片段。")
            return
        # 防止隱藏預留片段
        if "placeholder" in region.tags and not visible:
            self._set_status("無法隱藏預留片段。")
            return
        region.user_override = True
        region.visible = visible
        if visible:
            region.auto_hide_reasons.discard("manual_hide")
        else:
            region.auto_hide_reasons.add("manual_hide")

        self._mark_manual_edit("已更新片段顯示狀態。")
        self.timeline.set_regions(
            self.project.regions,
            self.project.analysis.duration,
            preserve_view=True,
        )
        self.timeline.set_selected_region(region.region_id)
        self._handle_region_selected(region.region_id)
        self._refresh_preview_ranges()

    def _toggle_selected_visibility(self):
        if not self.project:
            return
        region = self._get_selected_region()
        if not region:
            self._set_status("請先選擇要切換顯示的片段。")
            return
        self._set_selection_visibility(not region.visible)

    def _handle_region_selected_from_timeline(self, region_id: str | None):
        self._handle_region_selected(region_id, announce=True, sync_preview=True)

    def _handle_region_selected(self, region_id: str | None, *, announce: bool = False, sync_preview: bool = False):
        region = None
        if region_id and self.project:
            region = self._get_region_by_id(region_id)
            if region and announce:
                state = "輸出中" if region.visible else "已遮罩"
                label = region.text or state
                self._set_status(
                    f"選擇 {self._format_seconds(region.start)} ~ {self._format_seconds(region.end)} ｜ {label}"
                )
        if region and sync_preview:
            self.preview.set_playhead(region.start)
        self._update_editing_actions()

    def _update_editing_actions(self):
        if not hasattr(self, "cut_button"):
            return
        region = self._get_selected_region()
        can_cut = bool(region and region.duration > 0.05)
        can_toggle = bool(region)
        self.cut_button.setEnabled(can_cut)
        self.toggle_visibility_button.setEnabled(can_toggle)
        if region:
            if region.visible:
                self.toggle_visibility_button.setText("隱藏 (Ctrl+Shift+T)")
            else:
                self.toggle_visibility_button.setText("顯示 (Ctrl+Shift+T)")
        else:
            self.toggle_visibility_button.setText("切換顯示 (Ctrl+Shift+T)")

    def _get_selected_region(self) -> ClipRegion | None:
        if not self.project:
            return None
        region_id = self.timeline.selected_region_id()
        if not region_id:
            return None
        return self._get_region_by_id(region_id)

    def _get_region_by_id(self, region_id: str) -> ClipRegion | None:
        if not self.project:
            return None
        for region in self.project.regions:
            if region.region_id == region_id:
                return region
        return None

    def _apply_transcript_split(self, region: ClipRegion, new_region: ClipRegion, split_time: float) -> bool:
        transcript: TranscriptSegment | None = region.metadata.get("transcript")
        if not transcript or not transcript.characters:
            return False
        left_seg = transcript.slice(transcript.start, split_time)
        right_seg = transcript.slice(split_time, transcript.end)
        if not left_seg or not right_seg or not left_seg.characters or not right_seg.characters:
            return False

        region.metadata["transcript"] = left_seg
        new_region.metadata["transcript"] = right_seg
        region.text = left_seg.text
        new_region.text = right_seg.text
        return True

    def _split_text_by_ratio(self, text: str, duration: float, offset: float) -> tuple[str, str]:
        text = text or ""
        if not text or duration <= 0 or len(text) < 2:
            return text.strip(), ""
        ratio = max(0.0, min(1.0, offset / max(duration, 1e-6)))
        pivot = max(1, min(len(text) - 1, int(round(len(text) * ratio))))
        left = text[:pivot].strip()
        right = text[pivot:].strip()
        return (left, right)

    def _mark_manual_edit(self, message: str | None = None):
        self._has_manual_edits = True
        if message:
            self._set_status(message)

    def _format_seconds(self, value: float) -> str:
        total = max(0, int(value))
        minutes, seconds = divmod(total, 60)
        return f"{minutes:02d}:{seconds:02d}"


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = QuickCutWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
