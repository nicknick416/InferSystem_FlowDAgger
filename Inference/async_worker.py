"""Asynchronous inference worker for streaming action chunks into a smoother."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from Core import ActionSpace
from Inference.action_processing import canonicalize_action_chunk
from Inference.action_smoothing import TemporalActionSmoother

logger = logging.getLogger(__name__)


def schedule_next_loop(*, next_t: float, now: float, period: float) -> tuple[float, float]:
    """Return the next loop target and sleep.

    When inference already overran the observation period, sleep is zero so
    the next request starts immediately instead of padding another full period.
    """
    next_t = next_t + period
    sleep_s = next_t - now
    if sleep_s <= 0:
        return now, 0.0
    return next_t, sleep_s


@dataclass(slots=True)
class InferenceObservationSnapshot:
    """Observation snapshot sent to the inference client."""

    images: dict[str, np.ndarray]
    state: list[float]
    prompt: str = ""
    extra: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


ObservationFn = Callable[[], InferenceObservationSnapshot | None]
ActionChunkTransform = Callable[
    [list[list[float]], InferenceObservationSnapshot], list[list[float]]
]


class AsyncInferenceWorker:
    """Background loop that requests chunks and integrates them into a smoother."""

    def __init__(
        self,
        *,
        client: Any,
        observation_fn: ObservationFn,
        smoother: TemporalActionSmoother,
        action_space: Any = ActionSpace.JOINT_POSITION,
        policy_format: Any = "normal",
        canonical_dim: int = 32,
        effective_action_dim: int | None = None,
        overlap_steps: int = 8,
        max_latency_steps: int = 8,
        obs_fps: float = 30.0,
        action_fps: float | None = None,
        raw_action_callback: Callable[[int, Any], None] | None = None,
        action_chunk_transform: ActionChunkTransform | None = None,
        max_chunk_age_s: float = 1.5,
    ) -> None:
        self._client = client
        self._observation_fn = observation_fn
        self._smoother = smoother
        self._action_space = action_space
        self._policy_format = policy_format
        self._canonical_dim = int(canonical_dim)
        self._effective_action_dim = effective_action_dim
        self._overlap_steps = max(1, int(overlap_steps))
        self._max_latency_steps = max(0, int(max_latency_steps))
        self._obs_fps = max(float(obs_fps), 1e-6)
        self._action_fps = max(float(action_fps if action_fps is not None else obs_fps), 1e-6)
        self._raw_action_callback = raw_action_callback
        self._action_chunk_transform = action_chunk_transform
        self._max_chunk_age_s = max(float(max_chunk_age_s), 0.01)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._generation_lock = threading.Lock()
        self._action_generation = 0
        self._inflight = False
        self.request_count = 0
        self.last_infer_ms: float | None = None
        self.last_error: Exception | None = None
        self.last_chunk_monotonic: float | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 3.0, *, raise_on_timeout: bool = False) -> bool:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                logger.critical(
                    "async inference worker did not stop within %.1fs", timeout_s
                )
                if raise_on_timeout:
                    raise TimeoutError("async inference worker stop timed out")
                return False
            self._thread = None
        return True

    def invalidate_pending_actions(self) -> None:
        """清空待执行动作，并让当前仍在推理中的旧请求结果失效。"""
        with self._generation_lock:
            self._action_generation += 1
            self._smoother.clear()
            self.last_chunk_monotonic = None

    def chunk_is_fresh(self, now: float | None = None) -> bool:
        if self.last_chunk_monotonic is None:
            return False
        current = time.monotonic() if now is None else float(now)
        return current - self.last_chunk_monotonic <= self._max_chunk_age_s

    def can_execute_queued_action(self, now: float | None = None) -> bool:
        """Whether a popped leftover action may still be sent.

        A slow in-flight request may make the last integrated chunk older than
        ``max_chunk_age_s`` while the smoother still holds valid steps. Abort only
        when nothing is coming back.
        """
        with self._generation_lock:
            if self.last_chunk_monotonic is None:
                return False
            current = time.monotonic() if now is None else float(now)
            if current - self.last_chunk_monotonic <= self._max_chunk_age_s:
                return True
            return self._inflight

    def run_once(self) -> bool:
        """Request one chunk and integrate it. Returns True if a chunk was added."""
        with self._generation_lock:
            request_generation = self._action_generation
            self._inflight = True
        try:
            return self._run_once_request(request_generation)
        finally:
            with self._generation_lock:
                self._inflight = False

    def _run_once_request(self, request_generation: int) -> bool:
        obs = self._observation_fn()
        if obs is None:
            return False

        t0 = time.perf_counter()
        request_kwargs = {
            "prompt": obs.prompt,
            "extra": obs.extra,
            "episode_id": obs.meta.get("episode_id"),
            "step_id": obs.meta.get("step_id"),
            "policy_version": obs.meta.get("policy_version"),
            "request_generation": request_generation,
        }
        if hasattr(self._client, "predict_chunk_response"):
            response = self._client.predict_chunk_response(
                obs.images, obs.state, **request_kwargs
            )
            raw_actions = response.get("actions")
        else:
            raw_actions = self._client.predict_chunk(
                obs.images, obs.state, **request_kwargs
            )
            response = {"actions": raw_actions, **{
                key: value for key, value in (
                    ("episode_id", obs.meta.get("episode_id")),
                    ("step_id", obs.meta.get("step_id")),
                    ("policy_version", obs.meta.get("policy_version")),
                    ("request_generation", request_generation),
                ) if value is not None
            }}
        self.last_infer_ms = (time.perf_counter() - t0) * 1000
        self.request_count += 1

        expected = {
            "episode_id": obs.meta.get("episode_id"),
            "step_id": obs.meta.get("step_id"),
            "request_generation": request_generation,
            "policy_version": obs.meta.get("policy_version"),
        }
        for key, value in expected.items():
            if value is not None and response.get(key) != value:
                raise RuntimeError(
                    f"stale FlowDAgger response: {key}={response.get(key)!r}, "
                    f"expected {value!r}"
                )

        if raw_actions is None or len(raw_actions) == 0:
            logger.warning("async inference returned empty action chunk")
            return False

        if self._raw_action_callback is not None:
            try:
                self._raw_action_callback(self.request_count, raw_actions)
            except ValueError as exc:
                logger.debug("async raw action callback skipped: %s", exc)
            except Exception:
                logger.warning("async raw action callback failed", exc_info=True)

        actions = canonicalize_action_chunk(
            raw_actions,
            self._action_space,
            policy_format=self._policy_format,
            canonical_dim=self._canonical_dim,
            effective_action_dim=self._effective_action_dim,
        )
        latency_steps = min(
            int((self.last_infer_ms or 0.0) / 1000.0 * self._action_fps),
            self._max_latency_steps,
        )
        with self._generation_lock:
            if request_generation != self._action_generation:
                logger.warning(
                    "discarding stale async chunk #%d after action invalidation",
                    self.request_count,
                )
                return False
            if self._action_chunk_transform is not None:
                actions = self._action_chunk_transform(actions, obs)
            self._smoother.integrate_chunk(
                actions,
                latency_steps=latency_steps,
                overlap_steps=self._overlap_steps,
            )
            self.last_chunk_monotonic = time.monotonic()
        logger.debug(
            "async chunk #%d: actions=%d infer=%.1fms latency_steps=%d remaining=%d",
            self.request_count,
            len(actions),
            self.last_infer_ms,
            latency_steps,
            self._smoother.remaining,
        )
        return True

    def _run_loop(self) -> None:
        period = 1.0 / self._obs_fps
        next_t = time.perf_counter()
        while not self._stop_event.is_set():
            try:
                self.run_once()
                self.last_error = None
            except Exception as e:
                self.last_error = e
                logger.warning("async inference worker error: %s", e, exc_info=True)

            next_t, sleep_s = schedule_next_loop(
                next_t=next_t, now=time.perf_counter(), period=period
            )
            if sleep_s > 0:
                self._stop_event.wait(sleep_s)
