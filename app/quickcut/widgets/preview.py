from __future__ import annotations

from pathlib import Path
import sys
import os

from PyQt5.QtCore import Qt, pyqtSignal as Signal, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QPushButton, QSlider, QVBoxLayout, QWidget


def _detect_vlc_installation():
    """Detect VLC installation and set up environment variables."""
    vlc_paths = [
        r"C:\Program Files\VideoLAN\VLC",
        r"C:\Program Files (x86)\VideoLAN\VLC",
        r"C:\Program Files\vlc",
        r"C:\Program Files (x86)\vlc"
    ]

    for vlc_path in vlc_paths:
        libvlc_path = os.path.join(vlc_path, "libvlc.dll")
        if os.path.exists(libvlc_path):
            # Set VLC plugin path
            plugin_path = os.path.join(vlc_path, "plugins")
            if os.path.exists(plugin_path):
                os.environ['VLC_PLUGIN_PATH'] = plugin_path

            # Add VLC path to PATH for DLL loading
            if vlc_path not in os.environ.get('PATH', ''):
                os.environ['PATH'] = vlc_path + os.pathsep + os.environ.get('PATH', '')

            return vlc_path

    return None


def _get_vlc_installation_instructions():
    """Return installation instructions for VLC."""
    return (
        "VLC 媒體播放器未安裝。請按以下步驟安裝：\n"
        "1. 前往 https://www.videolan.org/vlc/ 下載 VLC 媒體播放器\n"
        "2. 運行安裝程式並完成安裝\n"
        "3. 重新啟動應用程式\n"
        "注意：安裝 64 位元版本以獲得最佳相容性。"
    )


# Detect VLC installation
vlc_install_path = _detect_vlc_installation()
VLC_AVAILABLE = False

if vlc_install_path:
    try:
        import vlc
        VLC_AVAILABLE = True
        print(f"VLC found at: {vlc_install_path}")
    except (ImportError, OSError) as e:
        print(f"VLC Python module failed to load: {e}")
        print("This may indicate a corrupted VLC installation.")
else:
    print("VLC media player not found on system.")
    print(_get_vlc_installation_instructions())


class VLCPreviewWidget(QWidget):
    """VLC-based video preview widget as fallback for Qt multimedia issues."""

    positionUpdated = Signal(float)
    durationUpdated = Signal(float)
    errorOccurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DontCreateNativeAncestors)
        self.setAttribute(Qt.WA_NativeWindow)

        if not VLC_AVAILABLE:
            if vlc_install_path is None:
                raise RuntimeError("VLC 媒體播放器未安裝。\n" + _get_vlc_installation_instructions())
            else:
                raise RuntimeError("VLC Python 模組載入失敗。請檢查 VLC 安裝是否完整。")

        try:
            # Create VLC instance
            self.instance = vlc.Instance()
            self.player = self.instance.media_player_new()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VLC: {e}")

        # Set up event manager for callbacks
        self.event_manager = self.player.event_manager()
        self.event_manager.event_attach(vlc.EventType.MediaPlayerPositionChanged, self._on_position_changed)
        self.event_manager.event_attach(vlc.EventType.MediaPlayerLengthChanged, self._on_length_changed)
        self.event_manager.event_attach(vlc.EventType.MediaPlayerEncounteredError, self._on_error)

        self._duration_ms = 0
        self._current_media: Path | None = None

    def load_media(self, media_path: Path):
        if not media_path.exists():
            self.errorOccurred.emit(f"找不到檔案：{media_path}")
            return

        self._current_media = media_path.resolve()
        media = self.instance.media_new(str(self._current_media))
        self.player.set_media(media)

        # Set video output to this widget
        if sys.platform.startswith('win'):
            self.player.set_hwnd(self.winId())
        elif sys.platform.startswith('linux'):
            self.player.set_xwindow(self.winId())
        elif sys.platform.startswith('darwin'):
            self.player.set_nsobject(self.winId())

        self.pause()

    def play(self):
        self.player.play()

    def pause(self):
        self.player.pause()

    def is_playing(self) -> bool:
        return self.player.is_playing()

    def toggle_playback(self):
        if self.is_playing():
            self.pause()
        else:
            self.play()

    def _seek(self, position_ms: int):
        if self._duration_ms > 0:
            position = position_ms / self._duration_ms
            self.player.set_position(position)

    def set_playhead(self, seconds: float, preroll_ms: int = 0):
        if self._duration_ms > 0:
            target_ms = max(0.0, seconds * 1000.0 - max(0, preroll_ms))
            position = target_ms / self._duration_ms
            self.player.set_position(position)

    def _on_position_changed(self, event):
        position = self.player.get_position()
        if self._duration_ms > 0:
            current_ms = position * self._duration_ms
            self.positionUpdated.emit(current_ms / 1000.0)

    def _on_length_changed(self, event):
        self._duration_ms = self.player.get_length()
        self.durationUpdated.emit(self._duration_ms / 1000.0)

    def _on_error(self, event):
        self.errorOccurred.emit("VLC 播放器錯誤：無法播放媒體檔案")


class PreviewPanel(QFrame):
    positionUpdated = Signal(float)
    durationUpdated = Signal(float)
    errorOccurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("previewPanel")
        self.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(self)

        # Create both video widgets
        self.qt_video_widget = QVideoWidget()
        self.qt_video_widget.setVisible(True)

        self.vlc_video_widget = None
        if VLC_AVAILABLE:
            try:
                self.vlc_video_widget = VLCPreviewWidget()
                self.vlc_video_widget.setVisible(False)
                self.vlc_video_widget.positionUpdated.connect(self._handle_position_changed)
                self.vlc_video_widget.durationUpdated.connect(self._handle_duration_changed)
                self.vlc_video_widget.errorOccurred.connect(self._handle_error)
            except Exception as e:
                print(f"Failed to initialize VLC widget: {e}")
                self.vlc_video_widget = None

        # Add widgets to layout
        layout.addWidget(self.qt_video_widget, stretch=1)
        if self.vlc_video_widget:
            layout.addWidget(self.vlc_video_widget, stretch=1)

        controls = QHBoxLayout()
        self.play_button = QPushButton("播放 / 暫停")
        controls.addWidget(self.play_button)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setSingleStep(1)
        controls.addWidget(self.slider, stretch=1)
        layout.addLayout(controls)

        # Qt player
        self.qt_player = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        self.qt_player.setVideoOutput(self.qt_video_widget)
        self.qt_player.setVolume(50)  # Set volume to 50%
        self.qt_player.setNotifyInterval(15)

        self.play_button.clicked.connect(self.toggle_playback)
        self.slider.sliderMoved.connect(self._seek)

        self.qt_player.positionChanged.connect(lambda pos: self._handle_position_changed(pos / 1000.0))
        self.qt_player.durationChanged.connect(lambda dur: self._handle_duration_changed(dur / 1000.0))
        self.qt_player.error.connect(self._handle_qt_error)

        self._duration_ms = 0
        self._current_media: Path | None = None
        self._last_error: str | None = None
        self._tried_wmf = False
        self._using_vlc = False
        self._vlc_failed = False

    def load_media(self, media_path: Path):
        media_path = Path(media_path)
        print(f"Loading media: {media_path}")  # Debug log
        if not media_path.exists():
            print(f"Media file not found: {media_path}")  # Debug log
            self.errorOccurred.emit(f"找不到檔案：{media_path}")
            return
        self._current_media = media_path.resolve()
        self._last_error = None
        self._tried_wmf = False
        self._using_vlc = False
        self._vlc_failed = False

        # Start with Qt player
        self.qt_player.stop()
        media_url = QUrl.fromLocalFile(str(self._current_media))
        print(f"Setting media URL: {media_url}")  # Debug log
        self.qt_player.setMedia(QMediaContent(media_url))
        self.pause()
        self.slider.setValue(0)

        # Show Qt widget, hide VLC
        self.qt_video_widget.setVisible(True)
        if self.vlc_video_widget:
            self.vlc_video_widget.setVisible(False)

        print("Media loaded with Qt player")  # Debug log

    def toggle_playback(self):
        if self._using_vlc and self.vlc_video_widget:
            self.vlc_video_widget.toggle_playback()
        else:
            if self.is_playing():
                self.pause()
            else:
                self.play()

    def is_playing(self) -> bool:
        if self._using_vlc and self.vlc_video_widget:
            return self.vlc_video_widget.is_playing()
        else:
            return self.qt_player.state() == QMediaPlayer.State.PlayingState

    def play(self):
        if self._using_vlc and self.vlc_video_widget:
            self.vlc_video_widget.play()
        else:
            self.qt_player.play()

    def pause(self):
        if self._using_vlc and self.vlc_video_widget:
            self.vlc_video_widget.pause()
        else:
            self.qt_player.pause()

    def _handle_position_changed(self, position_s: float):
        position_ms = int(round(position_s * 1000))
        if not self.slider.isSliderDown():
            self.slider.setValue(position_ms)
        self.positionUpdated.emit(position_s)

    def _handle_duration_changed(self, duration_s: float):
        self._duration_ms = int(round(duration_s * 1000))
        self.slider.setRange(0, max(self._duration_ms, 1))
        self.slider.setPageStep(max(1, self._duration_ms // 20))
        self.durationUpdated.emit(duration_s)

    def _handle_qt_error(self, error_code):
        if error_code == QMediaPlayer.Error.NoError:
            return
        message = self.qt_player.errorString().strip()
        if not message:
            message = f"媒體無法播放 (錯誤碼 {error_code})."
        print(f"Qt Preview error: {message}")  # Debug log

        # Try fallback to VLC if available and not already tried
        if not self._using_vlc and self.vlc_video_widget and not self._vlc_failed:
            print("Qt multimedia failed, trying VLC fallback")  # Debug log
            self._switch_to_vlc()
            return

        if "0x80040266" in message:
            print("Detected DirectShow error 0x80040266, trying WMF backend")  # Debug log
            if not self._tried_wmf:
                self._tried_wmf = True
                # 重試使用 WMF 後端
                if self._current_media:
                    self.qt_player.stop()
                    media_url = QUrl.fromLocalFile(str(self._current_media))
                    self.qt_player.setMedia(QMediaContent(media_url))
                return
            else:
                message = "媒體播放失敗：DirectShow 圖形未連接。已嘗試使用 WMF 後端但仍失敗。請檢查媒體檔案或系統配置。"

        if message != self._last_error:
            self._last_error = message
            print(f"Emitting error: {message}")  # Debug log
            self.errorOccurred.emit(message)

    def _switch_to_vlc(self):
        """Switch from Qt player to VLC player."""
        if not self.vlc_video_widget or self._using_vlc:
            return

        print("Switching to VLC player")  # Debug log
        self.qt_player.stop()
        self.qt_video_widget.setVisible(False)
        self.vlc_video_widget.setVisible(True)

        if self._current_media:
            try:
                self.vlc_video_widget.load_media(self._current_media)
                self._using_vlc = True
                print("Successfully switched to VLC")  # Debug log
            except Exception as e:
                print(f"Failed to switch to VLC: {e}")  # Debug log
                self._vlc_failed = True
                self.vlc_video_widget.setVisible(False)
                self.qt_video_widget.setVisible(True)
                self.errorOccurred.emit("VLC 備用播放器也失敗了")

    def _handle_error(self, message: str):
        """Handle errors from VLC widget."""
        print(f"VLC error: {message}")  # Debug log
        self._vlc_failed = True
        if message != self._last_error:
            self._last_error = message
            self.errorOccurred.emit(message)

    def _seek(self, position_ms: int):
        if self._using_vlc and self.vlc_video_widget:
            self.vlc_video_widget._seek(position_ms)
        else:
            self.qt_player.setPosition(position_ms)

    def set_playhead(self, seconds: float, preroll_ms: int = 0):
        position_ms = max(0, int(round(seconds * 1000)) - max(0, preroll_ms))
        if self._using_vlc and self.vlc_video_widget:
            self.vlc_video_widget.set_playhead(seconds, preroll_ms=preroll_ms)
        else:
            self.qt_player.setPosition(position_ms)
