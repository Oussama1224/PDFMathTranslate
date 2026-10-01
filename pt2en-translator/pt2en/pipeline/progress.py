"""Stage-based progress reporting with completion estimates."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Optional


class Stage(str, Enum):
    UPLOADING = "uploading"
    ANALYZING = "analyzing"
    EXTRACTING = "extracting"
    OCR = "ocr"
    TRANSLATING = "translating"
    REBUILDING = "rebuilding"
    VALIDATING = "validating"
    FINALIZING = "finalizing"
    DONE = "done"


STAGE_LABELS = {
    Stage.UPLOADING: "Uploading",
    Stage.ANALYZING: "Analyzing",
    Stage.EXTRACTING: "Extracting",
    Stage.OCR: "OCR",
    Stage.TRANSLATING: "Translating",
    Stage.REBUILDING: "Rebuilding layout",
    Stage.VALIDATING: "Validating",
    Stage.FINALIZING: "Finalizing",
    Stage.DONE: "Done",
}

PROCESSING_STAGES = [
    Stage.ANALYZING,
    Stage.EXTRACTING,
    Stage.OCR,
    Stage.TRANSLATING,
    Stage.REBUILDING,
    Stage.VALIDATING,
    Stage.FINALIZING,
]

DEFAULT_WEIGHTS = {
    Stage.ANALYZING: 0.08,
    Stage.EXTRACTING: 0.04,
    Stage.OCR: 0.12,
    Stage.TRANSLATING: 0.46,
    Stage.REBUILDING: 0.12,
    Stage.VALIDATING: 0.15,
    Stage.FINALIZING: 0.03,
}


@dataclass
class ProgressEvent:
    stage: str
    stage_label: str
    stage_index: int
    stage_progress: float
    overall: float
    message: str
    elapsed: float
    eta_seconds: Optional[float]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class ProgressReporter:
    def __init__(self, callback: Optional[Callable[[ProgressEvent], None]] = None):
        self.callback = callback
        self.weights = dict(DEFAULT_WEIGHTS)
        self.stage: Stage = Stage.ANALYZING
        self.fraction = 0.0
        self.started = time.time()
        self.stage_started = self.started
        self.timings: dict[str, float] = {}
        self._lock = threading.Lock()
        self._eta: Optional[float] = None
        self._last_emit = 0.0

    def reweight(self, costs: dict[Stage, float]) -> None:
        """Re-balance stage weights from estimated relative costs."""
        total = sum(max(c, 0.0) for c in costs.values()) or 1.0
        with self._lock:
            for stage, cost in costs.items():
                self.weights[stage] = max(cost, 0.0) / total

    def _overall(self) -> float:
        done = 0.0
        for s in PROCESSING_STAGES:
            if s == self.stage:
                return min(1.0, done + self.weights.get(s, 0) * self.fraction)
            done += self.weights.get(s, 0)
        return 1.0 if self.stage == Stage.DONE else min(done, 1.0)

    def start(self, stage: Stage, message: str = "") -> None:
        with self._lock:
            now = time.time()
            self.timings[self.stage.value] = (
                self.timings.get(self.stage.value, 0.0) + now - self.stage_started
            )
            self.stage = stage
            self.stage_started = now
            self.fraction = 0.0
        self._emit(message or f"{STAGE_LABELS[stage]}…", force=True)

    def update(self, fraction: float, message: str = "") -> None:
        with self._lock:
            self.fraction = max(self.fraction, min(max(fraction, 0.0), 1.0))
        self._emit(message)

    def finish(self, message: str = "Done") -> None:
        self.start(Stage.DONE, message)

    def _emit(self, message: str, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            if not force and now - self._last_emit < 0.25:
                return
            self._last_emit = now
            overall = self._overall()
            elapsed = now - self.started
            eta = None
            if 0.03 < overall < 1.0 and elapsed > 2:
                raw = elapsed * (1 - overall) / overall
                self._eta = raw if self._eta is None else 0.7 * self._eta + 0.3 * raw
                eta = round(self._eta, 1)
            event = ProgressEvent(
                stage=self.stage.value,
                stage_label=STAGE_LABELS[self.stage],
                stage_index=(
                    PROCESSING_STAGES.index(self.stage)
                    if self.stage in PROCESSING_STAGES
                    else len(PROCESSING_STAGES)
                ),
                stage_progress=round(self.fraction, 4),
                overall=round(overall, 4),
                message=message,
                elapsed=round(elapsed, 1),
                eta_seconds=eta if self.stage != Stage.DONE else 0.0,
            )
        if self.callback:
            self.callback(event)
