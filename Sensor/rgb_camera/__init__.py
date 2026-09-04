import logging as _logging

_logger = _logging.getLogger(__name__)

from .base import BaseRGBCamera
from .mock_camera import MockRGBCamera

__all__ = [
    "BaseRGBCamera",
    "MockRGBCamera",
]

# RealSense 驱动按需导入 (需要 pyrealsense2)
try:
    from .realsense_camera import RealSenseCamera

    __all__ += ["RealSenseCamera"]
except ImportError as _e:
    _logger.debug("RealSenseCamera 不可用 (缺少 pyrealsense2): %s", _e)
