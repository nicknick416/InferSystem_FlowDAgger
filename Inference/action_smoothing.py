"""Temporal smoothing for model action chunks."""
from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from Core import ActionSpace
from Inference.action_processing import (
    blend_action_values,
    canonicalize_action_values,
    normalize_action_space,
)


class TemporalActionSmoother:
    """Maintain and blend the action sequence currently being executed."""

    def __init__(self, *, action_space: Any = ActionSpace.JOINT_POSITION) -> None:
        self._action_space = normalize_action_space(action_space)
        self._cur_chunk: deque[list[float]] = deque()
        self._last_action: list[float] | None = None
        self._published_steps = 0
        self._lock = threading.Lock()

    @property
    def remaining(self) -> int:
        with self._lock:
            return len(self._cur_chunk)

    def clear(self) -> None:
        with self._lock:
            self._cur_chunk.clear()
            self._last_action = None
            self._published_steps = 0

    def integrate_chunk(
        self,
        actions_chunk: Sequence[Sequence[float]],
        *,
        latency_steps: int | None = 0,
        max_latency_steps: int | None = None,
        overlap_steps: int = 8,
        min_smooth_steps: int | None = None,
    ) -> None:
        """Blend a fresh model chunk into the remaining execution sequence."""
        if actions_chunk is None or len(actions_chunk) == 0:
            return
        if min_smooth_steps is not None:
            overlap_steps = min_smooth_steps
        overlap_steps = max(1, int(overlap_steps))

        with self._lock:
            if latency_steps is None:
                drop_n = self._published_steps
                if max_latency_steps is not None:
                    drop_n = min(drop_n, max(0, int(max_latency_steps)))
            else:
                drop_n = max(0, int(latency_steps))
            if drop_n >= len(actions_chunk):
                return
            new_list = [
                canonicalize_action_values(action, self._action_space)
                for action in actions_chunk[drop_n:]
            ]

            if len(self._cur_chunk) == 0 and self._last_action is not None:
                old_list = [list(self._last_action) for _ in range(overlap_steps)]
                self._last_action = None
            else:
                old_list = list(self._cur_chunk)
                if 0 < len(old_list) < overlap_steps:
                    tail = list(old_list[-1])
                    old_list.extend([list(tail) for _ in range(overlap_steps - len(old_list))])
                elif len(old_list) == 0:
                    self._cur_chunk = deque([list(a) for a in new_list])
                    self._published_steps = 0
                    return
            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                self._cur_chunk = deque([list(a) for a in new_list])
                self._published_steps = 0
                return
            if overlap_len == 1:
                weights_old = np.array([1.0], dtype=np.float64)
            else:
                weights_old = np.linspace(1.0, 0.0, overlap_len, dtype=np.float64)

            smoothed = [
                blend_action_values(
                    old_list[i],
                    new_list[i],
                    weight_old=float(weights_old[i]),
                    action_space=self._action_space,
                )
                for i in range(overlap_len)
            ]
            combined = smoothed + [list(a) for a in new_list[overlap_len:]]
            self._cur_chunk = deque(combined)
            self._published_steps = 0

    def pop_next(self) -> list[float] | None:
        with self._lock:
            if len(self._cur_chunk) == 0:
                return None
            if len(self._cur_chunk) == 1:
                self._last_action = list(self._cur_chunk[0])
            action = list(self._cur_chunk.popleft())
            self._published_steps += 1
            return action

    def peek_next(self) -> list[float] | None:
        with self._lock:
            if len(self._cur_chunk) == 0:
                return None
            return list(self._cur_chunk[0])
