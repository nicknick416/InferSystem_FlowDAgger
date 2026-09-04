import logging as _logging

_logger = _logging.getLogger(__name__)

from .base import BaseRobot
from .gripper import BaseGripper, GripperState
from .flexiv import FlexivGripper, FlexivRobot

try:
    from .arx5 import Arx5Robot
except Exception as _e:  # pragma: no cover - optional dependency may be absent
    Arx5Robot = None  # type: ignore[assignment]
    _logger.debug("Arx5Robot 不可用 (可选依赖缺失): %s", _e)

try:
    from .arx5_bimanual import Arx5BimanualRobot
except Exception as _e:  # pragma: no cover - optional dependency may be absent
    Arx5BimanualRobot = None  # type: ignore[assignment]
    _logger.debug("Arx5BimanualRobot 不可用 (可选依赖缺失): %s", _e)

try:
    from .aloha import AlohaRobot
except Exception as _e:  # pragma: no cover - optional dependency may be absent
    AlohaRobot = None  # type: ignore[assignment]
    _logger.debug("AlohaRobot 不可用 (可选依赖缺失): %s", _e)

try:
    from .ur import URRobot
except Exception as _e:  # pragma: no cover - optional dependency may be absent
    URRobot = None  # type: ignore[assignment]
    _logger.debug("URRobot 不可用 (可选依赖缺失): %s", _e)

__all__ = [
    "AlohaRobot",
    "Arx5Robot",
    "Arx5BimanualRobot",
    "BaseGripper",
    "BaseRobot",
    "FlexivGripper",
    "FlexivRobot",
    "GripperState",
    "URRobot",
]
