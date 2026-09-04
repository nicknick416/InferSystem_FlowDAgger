import logging as _logging

_logger = _logging.getLogger(__name__)

from .action_processing import canonicalize_action_chunk, canonicalize_action_values, format_policy_state_vector
from .action_smoothing import TemporalActionSmoother
from .async_worker import AsyncInferenceWorker, InferenceObservationSnapshot
from .dispatch import ActionDispatcher, build_eef_state_vector, build_policy_state_vector, build_state_vector
from .obs_mapping import build_camera_key_map, map_image_keys

__all__ = [
    "ActionDispatcher",
    "AsyncInferenceWorker",
    "InferenceObservationSnapshot",
    "TemporalActionSmoother",
    "build_camera_key_map",
    "build_eef_state_vector",
    "build_policy_state_vector",
    "build_state_vector",
    "canonicalize_action_chunk",
    "canonicalize_action_values",
    "format_policy_state_vector",
    "map_image_keys",
]

# client/server 需要 zmq + msgpack，按需导入
try:
    from .client import InferenceClient
    from .server import InferenceServer

    __all__ += ["InferenceClient", "InferenceServer"]
except ImportError as _e:
    _logger.debug("InferenceClient/InferenceServer 不可用 (缺少 zmq/msgpack): %s", _e)
