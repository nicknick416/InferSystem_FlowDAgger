import logging as _logging

_logger = _logging.getLogger(__name__)

from .base import BaseTactileSensor

__all__ = [
    "BaseTactileSensor",
]

# OpenCV 触觉驱动按需导入
try:
    from .opencv_tactile import OpenCVTactileSensor

    __all__ += ["OpenCVTactileSensor"]
except ImportError as _e:
    _logger.debug("OpenCVTactileSensor 不可用: %s", _e)
