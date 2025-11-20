from __future__ import annotations

from typing import Iterable, List, Optional

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal as Signal
from PyQt5.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QWidget

from ..models import ClipRegion


class TimelineWidget(QWidget):
    regionToggled = Signal(str)
    timeSelected = Signal(float)
    regionSelected = Signal(object)
    viewChanged = Signal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._regions: List[ClipRegion] = []
        self._duration = 0.0
        self._playhead = 0.0
        self._hover_region: Optional[str] = None
        self._selected_region: Optional[str] = None
        self._envelope: List[tuple[float, float]] = []
        self._view_start = 0.0
        self._view_duration = 0.0
        self.setMouseTracking(True)
        self.setMinimumHeight(180)

    def set_regions(self, regions: Iterable[ClipRegion], duration: float, preserve_view: bool = False):
        old_view_start = self._view_start
        old_view_duration = self._view_duration
        self._regions = list(regions)
        # 確保預留片段始終 visible
        for region in self._regions:
            if "placeholder" in region.tags:
                region.visible = True
        self._duration = duration
        self._hover_region = None
        self._selected_region = None
        if preserve_view and self._duration > 0 and old_view_duration > 0:
            self._view_duration = min(self._duration, max(old_view_duration, 1e-3))
            max_start = max(0.0, self._duration - self._view_duration)
            self._view_start = max(0.0, min(old_view_start, max_start))
        else:
            self._reset_view_window()
        self.update()
        self._emit_view_changed()

    def set_envelope(self, envelope: List[tuple[float, float]]):
        self._envelope = envelope
        self.update()

    def set_playhead(self, seconds: float):
        seconds = max(0.0, seconds)
        previous = self._playhead
        if abs(seconds - previous) < 1e-4 and self._view_duration > 0:
            return
        self._playhead = seconds
        if self._ensure_playhead_visible():
            self.update()
        else:
            self._update_playhead_line(previous, self._playhead)

    def playhead(self) -> float:
        return self._playhead

    def selected_region_id(self) -> Optional[str]:
        return self._selected_region

    def set_selected_region(self, region_id: Optional[str]):
        if region_id == self._selected_region:
            return
        self._selected_region = region_id
        self.update()

    def focus_view_on_region(self, region_id: str):
        region = next((r for r in self._regions if r.region_id == region_id), None)
        if not region:
            return
        self._zoom_to_window(region.start, region.end)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101010"))
        painter.setRenderHint(QPainter.Antialiasing, True)

        track_rect = self._track_rect()
        painter.setPen(QColor("#303033"))
        for i in range(11):
            x = track_rect.left() + (track_rect.width() * i / 10.0)
            painter.drawLine(int(x), track_rect.top(), int(x), track_rect.bottom())

        self._draw_envelope(painter, track_rect)
        self._draw_regions(painter, track_rect)
        self._draw_playhead(painter, track_rect)

    def _draw_envelope(self, painter: QPainter, rect):
        if not self._envelope or self._duration <= 0:
            return
        max_value = max((value for _, value in self._envelope), default=1e-6)
        if max_value <= 0:
            return
        painter.save()
        view_start = self._view_start
        view_end = self._view_start + max(self._view_duration, 1e-6)
        time_span = max(self._view_duration, 1e-6)
        samples: list[QPointF] = []
        for time, value in self._envelope:
            if time < view_start or time > view_end:
                continue
            local_ratio = (time - view_start) / time_span
            x = rect.left() + local_ratio * rect.width()
            gain = min(1.0, value / max_value)
            y = rect.bottom() - gain * rect.height()
            samples.append(QPointF(x, y))

        if len(samples) < 2:
            painter.restore()
            return

        fill_path = QPainterPath()
        fill_path.moveTo(samples[0].x(), rect.bottom())
        for point in samples:
            fill_path.lineTo(point)
        fill_path.lineTo(samples[-1].x(), rect.bottom())
        fill_path.closeSubpath()
        painter.fillPath(fill_path, QColor(58, 150, 221, 80))

        line_path = QPainterPath()
        line_path.moveTo(samples[0])
        for point in samples[1:]:
            line_path.lineTo(point)
        pen = QPen(QColor("#5ab8ff"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawPath(line_path)
        painter.restore()

    def _draw_regions(self, painter: QPainter, rect):
        if self._duration <= 0:
            return
        view_end = self._view_start + max(self._view_duration, 1e-6)
        time_span = max(self._view_duration, 1e-6)
        for region in self._regions:
            start = max(region.start, self._view_start)
            end = min(region.end, view_end)
            if end <= start:
                continue
            x = rect.left() + ((start - self._view_start) / time_span) * rect.width()
            w = max(2.0, ((end - start) / time_span) * rect.width())
            region_rect = QRectF(rect)
            region_rect.setLeft(x)
            region_rect.setWidth(w)
            # 預留片段始終顯示為藍色
            if "placeholder" in region.tags:
                color = QColor("#3d7bfd")
                color.setAlpha(160)
            else:
                color = QColor("#3d7bfd") if region.visible else QColor("#5a2d2d")
                color.setAlpha(160 if region.visible else 180)
            painter.fillRect(region_rect, color)

            if region.region_id == self._selected_region:
                border = QColor("#ffcc00")
                width = 2.2
            elif region.region_id == self._hover_region:
                border = QColor("#ffffff")
                width = 1.6
            else:
                border = QColor("#111111")
                width = 1.2
            painter.setPen(QPen(border, width))
            painter.drawRect(region_rect)

            if region.text:
                painter.setPen(QColor("#ffffff"))
                painter.drawText(
                    region_rect.adjusted(4, 2, -4, -2),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    region.text,
                )

    def _draw_playhead(self, painter: QPainter, rect):
        if self._duration <= 0:
            return
        time_span = max(self._view_duration, 1e-6)
        ratio = (self._playhead - self._view_start) / time_span
        ratio = max(0.0, min(1.0, ratio))
        x = rect.left() + ratio * rect.width()
        painter.setPen(QPen(QColor("#ffcc00"), 2))
        painter.drawLine(int(x), rect.top(), int(x), rect.bottom())

    def _update_playhead_line(self, old_time: float, new_time: float):
        if self._duration <= 0:
            self.update()
            return
        rect = self._track_rect()
        if rect.width() <= 0:
            self.update()
            return
        old_rect = self._playhead_line_rect(old_time, rect)
        new_rect = self._playhead_line_rect(new_time, rect)
        if not old_rect and not new_rect:
            return
        if old_rect and new_rect:
            update_rect = old_rect.united(new_rect)
        else:
            update_rect = old_rect or new_rect
        if update_rect:
            padded = update_rect.adjusted(-2, -2, 2, 2)
            self.update(padded.toRect())

    def _playhead_line_rect(self, time_value: float, track_rect: QRectF) -> QRectF | None:
        if self._view_duration <= 0:
            return None
        view_start = self._view_start
        view_end = view_start + self._view_duration
        if time_value < view_start or time_value > view_end:
            return None
        ratio = (time_value - view_start) / max(self._view_duration, 1e-6)
        ratio = max(0.0, min(1.0, ratio))
        x = track_rect.left() + ratio * track_rect.width()
        return QRectF(x - 2.5, track_rect.top(), 5.0, track_rect.height())

    def mouseMoveEvent(self, event: QMouseEvent):
        time = self._x_to_time(event.pos().x())
        region = self._find_region(time)
        region_id = region.region_id if region else None
        if region_id != self._hover_region:
            self._hover_region = region_id
            self.update()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if self._duration <= 0:
            return
        time = self._x_to_time(event.pos().x())
        region = self._find_region(time)
        new_selection = region.region_id if region else None
        if new_selection != self._selected_region:
            self._selected_region = new_selection
            self.regionSelected.emit(new_selection)
            self.update()
        elif region is None:
            self.regionSelected.emit(None)
        self.timeSelected.emit(time)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self._duration <= 0:
            return
        time = self._x_to_time(event.pos().x())
        region = self._find_region(time)
        if region:
            self.regionToggled.emit(region.region_id)
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if self._duration <= 0:
            return super().wheelEvent(event)
        delta = event.angleDelta()
        vertical = delta.y()
        horizontal = delta.x()
        anchor_x = event.position().x() if hasattr(event, "position") else event.pos().x()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            zoom_delta = vertical or horizontal
            self._handle_zoom(zoom_delta, anchor_x)
        else:
            scroll_delta = horizontal if abs(horizontal) > abs(vertical) else vertical
            self._handle_scroll(scroll_delta)
        event.accept()

    def _x_to_time(self, x: float) -> float:
        if self._duration <= 0:
            return 0.0
        rect = self._track_rect()
        ratio = max(0.0, min(1.0, (x - rect.left()) / max(1.0, rect.width())))
        span = max(self._view_duration, 1e-6)
        return self._view_start + ratio * span

    def _find_region(self, time_point: float) -> Optional[ClipRegion]:
        for region in self._regions:
            if region.start <= time_point <= region.end:
                return region
        return None

    def _track_rect(self) -> QRectF:
        return self.rect().adjusted(10, 30, -10, -30)

    def _reset_view_window(self):
        if self._duration <= 0:
            self._view_start = 0.0
            self._view_duration = 0.0
        else:
            self._view_start = 0.0
            self._view_duration = max(self._duration, 1e-3)
        self._emit_view_changed()

    def _ensure_playhead_visible(self) -> bool:
        if self._duration <= 0 or self._view_duration <= 0:
            return False
        view_end = self._view_start + self._view_duration
        changed = False
        if self._playhead < self._view_start:
            self._view_start = max(0.0, self._playhead - 0.1 * self._view_duration)
            changed = True
        elif self._playhead > view_end:
            max_start = max(0.0, self._duration - self._view_duration)
            new_start = max(0.0, min(max_start, self._playhead - 0.9 * self._view_duration))
            if new_start != self._view_start:
                self._view_start = new_start
                changed = True
        if changed:
            self._emit_view_changed()
        return changed

    def _handle_zoom(self, angle_delta: float, anchor_x: float):
        if angle_delta == 0 or self._duration <= 0:
            return
        factor = 0.9 if angle_delta > 0 else 1.1
        min_span = min(self._duration, max(0.25, self._duration / 200.0))
        new_span = max(min_span, min(self._duration, self._view_duration * factor if self._view_duration else self._duration * factor))
        anchor_time = self._x_to_time(anchor_x)
        if self._view_duration <= 0:
            self._view_duration = self._duration
        ratio = (anchor_time - self._view_start) / max(1e-6, self._view_duration)
        new_start = anchor_time - ratio * new_span
        max_start = max(0.0, self._duration - new_span)
        self._view_start = max(0.0, min(new_start, max_start))
        self._view_duration = new_span
        self._emit_view_changed()
        self.update()

    def _handle_scroll(self, angle_delta: float):
        if angle_delta == 0 or self._duration <= self._view_duration:
            return
        offset = -angle_delta / 120.0
        delta_time = offset * self._view_duration * 0.1
        max_start = max(0.0, self._duration - self._view_duration)
        new_start = max(0.0, min(self._view_start + delta_time, max_start))
        if new_start != self._view_start:
            self._view_start = new_start
            self._emit_view_changed()
        self.update()

    def _zoom_to_window(self, start: float, end: float):
        if self._duration <= 0:
            return
        start = max(0.0, start)
        end = min(self._duration, max(start + 1e-3, end))
        self._view_start = start
        self._view_duration = end - start
        self._emit_view_changed()
        self.update()

    def set_view_start_ratio(self, ratio: float):
        if self._duration <= 0 or self._view_duration <= 0 or self._view_duration >= self._duration:
            return
        ratio = max(0.0, min(1.0, ratio))
        max_start = self._duration - self._view_duration
        new_start = ratio * max_start
        if abs(new_start - self._view_start) < 1e-6:
            return
        self._view_start = new_start
        self.update()
        self._emit_view_changed()

    def _emit_view_changed(self):
        if self._duration <= 0:
            self.viewChanged.emit(0.0, 0.0, 0.0)
            return
        span = min(self._view_duration or self._duration, self._duration)
        start = min(max(self._view_start, 0.0), max(0.0, self._duration - span))
        self.viewChanged.emit(start, span, self._duration)
