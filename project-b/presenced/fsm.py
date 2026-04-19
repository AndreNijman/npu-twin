from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    PRESENT = "present"
    AWAY_GRACE = "away_grace"
    AWAY = "away"


@dataclass
class Transition:
    frm: State
    to: State
    reason: str


class PresenceFSM:
    def __init__(self, grace_period_s: float, now: float | None = None) -> None:
        self.grace_period_s = grace_period_s
        self.state = State.PRESENT
        self.last_seen = now if now is not None else time.monotonic()
        self._grace_started: float | None = None

    def observe(self, face_present: bool, now: float | None = None) -> Transition | None:
        t = now if now is not None else time.monotonic()
        if face_present:
            self.last_seen = t
            self._grace_started = None
            if self.state != State.PRESENT:
                prev = self.state
                self.state = State.PRESENT
                return Transition(prev, self.state, "face detected")
            return None

        if self.state == State.PRESENT:
            self._grace_started = t
            self.state = State.AWAY_GRACE
            return Transition(State.PRESENT, self.state, "face lost; entering grace")

        if self.state == State.AWAY_GRACE:
            started = self._grace_started or t
            if t - started >= self.grace_period_s:
                self.state = State.AWAY
                return Transition(State.AWAY_GRACE, self.state, "grace elapsed")
        return None
