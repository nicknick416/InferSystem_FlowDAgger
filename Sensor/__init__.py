import logging as _logging

_logger = _logging.getLogger(__name__)

from .base import BaseSensor
from .manager import SensorManager
from .rgb_camera import BaseRGBCamera, MockRGBCamera
from .tactile import BaseTactileSensor

MockCamera = MockRGBCamera

__all__ = [
    "BaseSensor",
    "SensorManager",
    "BaseRGBCamera",
    "BaseTactileSensor",
    "MockRGBCamera",
    "MockCamera",
]

# 按需导入硬件驱动
try:
    from .tactile import OpenCVTactileSensor
    __all__ += ["OpenCVTactileSensor"]
except ImportError as _e:
    _logger.debug("OpenCVTactileSensor 不可用: %s", _e)
