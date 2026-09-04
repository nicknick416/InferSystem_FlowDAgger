"""FlowDAgger episode state and keyboard-event handling.

This module is deliberately hardware independent so the transition rules can be
unit tested without CAN devices or cameras.
"""
from __future__ import annotations

import enum
import threading
import time
import uuid
from dataclasses import dataclass


class FlowDaggerPhase(str, enum.Enum):
    WAIT_START = "wait_start"
    POLICY = "policy"
    EXPERT = "expert"
    ENDING = "ending"
    SAFE_HOLD = "safe_hold"
    HOME = "home"
    UPDATING = "updating"
    READY = "ready"
    STOPPED = "stopped"


class EpisodeLabel(str, enum.Enum):
    ASSISTED_SUCCESS = "assisted_success"
    AUTONOMOUS_SUCCESS = "autonomous_success"
    FAILURE = "failure"
    ABORT = "abort"


@dataclass(frozen=True, slots=True)
class FlowDaggerEvent:
    name: str
    timestamp_s: float


class FlowDaggerKeyEvents:
    """Thread-safe, edge-triggered key-event queue."""

    KEY_TO_EVENT = {
        " ": "toggle_intervention",
        "1": "episode_assisted_success",
        "3": "episode_failure",
        "r": "episode_abort",
        "q": "quit",
        "\n": "start",
        "\r": "start",
    }

    def __init__(self) -> None:
        self._events: list[FlowDaggerEvent] = []
        self._lock = threading.Lock()

    def push_key(self, key: str) -> bool:
        name = self.KEY_TO_EVENT.get(key.lower())
        if name is None:
            return False
        with self._lock:
            self._events.append(FlowDaggerEvent(name, time.time()))
        return True

    def drain(self) -> list[FlowDaggerEvent]:
        with self._lock:
            events, self._events = self._events, []
        return events


class FlowDaggerSessionState:
    """Validated episode state machine used by the hardware loop."""

    def __init__(self) -> None:
        self.phase = FlowDaggerPhase.WAIT_START
        self.episode_id: str | None = None
        self.step_id = 0
        self.label: EpisodeLabel | None = None

    def start(self, episode_id: str | None = None) -> str:
        if self.phase is not FlowDaggerPhase.WAIT_START:
            raise RuntimeError(f"cannot start from {self.phase.value}")
        self.episode_id = episode_id or uuid.uuid4().hex
        self.step_id = 0
        self.label = None
        self.phase = FlowDaggerPhase.POLICY
        return self.episode_id

    def toggle_intervention(self) -> FlowDaggerPhase:
        if self.phase is FlowDaggerPhase.POLICY:
            self.phase = FlowDaggerPhase.EXPERT
        elif self.phase is FlowDaggerPhase.EXPERT:
            self.phase = FlowDaggerPhase.POLICY
        else:
            raise RuntimeError(f"cannot toggle intervention from {self.phase.value}")
        return self.phase

    def advance(self) -> int:
        if self.phase not in (FlowDaggerPhase.POLICY, FlowDaggerPhase.EXPERT):
            raise RuntimeError(f"cannot advance from {self.phase.value}")
        self.step_id += 1
        return self.step_id

    def hold_policy(self) -> FlowDaggerPhase:
        """Keep the episode in POLICY while execution is safely paused."""
        if self.phase is not FlowDaggerPhase.POLICY:
            raise RuntimeError(f"cannot hold policy from {self.phase.value}")
        return self.phase

    def finish(self, label: EpisodeLabel) -> None:
        if self.phase not in (
            FlowDaggerPhase.POLICY, FlowDaggerPhase.EXPERT,
            FlowDaggerPhase.SAFE_HOLD,
        ):
            raise RuntimeError(f"cannot finish from {self.phase.value}")
        self.label = label
        self.phase = FlowDaggerPhase.ENDING

    def safe_hold(self) -> FlowDaggerPhase:
        if self.phase not in (FlowDaggerPhase.POLICY, FlowDaggerPhase.EXPERT):
            raise RuntimeError(f"cannot enter safe hold from {self.phase.value}")
        self.phase = FlowDaggerPhase.SAFE_HOLD
        return self.phase

    def home_complete(self) -> FlowDaggerPhase:
        if self.phase is not FlowDaggerPhase.ENDING:
            raise RuntimeError(f"cannot complete home from {self.phase.value}")
        self.phase = FlowDaggerPhase.HOME
        return self.phase

    def begin_update(self, *, training_queued: bool = True) -> None:
        if self.phase is not FlowDaggerPhase.HOME:
            raise RuntimeError(f"cannot update from {self.phase.value}")
        self.phase = (
            FlowDaggerPhase.UPDATING
            if self.label is EpisodeLabel.ASSISTED_SUCCESS and training_queued
            else FlowDaggerPhase.WAIT_START
        )

    def update_complete(self) -> None:
        if self.phase is not FlowDaggerPhase.UPDATING:
            raise RuntimeError(f"cannot complete update from {self.phase.value}")
        self.phase = FlowDaggerPhase.WAIT_START

    def stop(self) -> None:
        self.phase = FlowDaggerPhase.STOPPED
