from __future__ import annotations

import os

# 強制 Qt 優先使用 Windows Media Foundation 後端，避免 DirectShow 無法解碼 mp4。
os.environ.setdefault("QT_MULTIMEDIA_PREFERRED_PLUGINS", "windowsmediafoundation")

__all__ = ["main"]
__version__ = "0.1.0"

