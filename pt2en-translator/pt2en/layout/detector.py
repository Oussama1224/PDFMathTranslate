"""Optional ML layout detection.

The default pipeline is fully heuristic. When ``PT2EN_LAYOUT_DETECTOR=doclayout``
is set and ``onnxruntime`` is installed, a DocLayout-YOLO ONNX model is used to
provide region hints (formula, figure, table, title...) that refine the
heuristic classification. Any other detector can be plugged in by implementing
:class:`LayoutDetector`.
"""

from __future__ import annotations

import abc
import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from pt2en.model import Rect, overlap_ratio

log = logging.getLogger(__name__)


@dataclass
class LayoutHints:
    regions: list[tuple[str, Rect, float]] = field(default_factory=list)

    def label_for(self, bbox: Rect, min_overlap: float = 0.6) -> Optional[str]:
        best, best_score = None, 0.0
        for label, rect, conf in self.regions:
            score = overlap_ratio(bbox, rect) * conf
            if overlap_ratio(bbox, rect) >= min_overlap and score > best_score:
                best, best_score = label, score
        return best


class LayoutDetector(abc.ABC):
    name = "base"

    @abc.abstractmethod
    def detect(
        self, image: np.ndarray, page_width: float, page_height: float
    ) -> LayoutHints:
        """Detect regions on a rendered page (RGB array) in PDF coordinates."""


class NullLayoutDetector(LayoutDetector):
    name = "heuristic"

    def detect(self, image, page_width, page_height) -> LayoutHints:
        return LayoutHints()


class DocLayoutYoloDetector(LayoutDetector):
    """DocLayout-YOLO (DocStructBench) exported to ONNX."""

    name = "doclayout"
    HF_REPO = "wybxc/DocLayout-YOLO-DocStructBench-onnx"
    HF_FILE = "doclayout_yolo_docstructbench_imgsz1024.onnx"

    def __init__(self, model_path: Optional[Path] = None):
        import onnxruntime  # noqa: F401 - optional dependency

        if model_path is None or not Path(model_path).exists():
            from huggingface_hub import hf_hub_download

            model_path = Path(hf_hub_download(self.HF_REPO, self.HF_FILE))
        import onnx

        model = onnx.load(str(model_path))
        metadata = {d.key: d.value for d in model.metadata_props}
        self.stride = ast.literal_eval(metadata.get("stride", "32"))
        if isinstance(self.stride, (list, tuple)):
            self.stride = int(max(self.stride))
        self.names = ast.literal_eval(metadata["names"])
        self.session = onnxruntime.InferenceSession(model.SerializeToString())

    def _letterbox(
        self, image: np.ndarray, size: int
    ) -> tuple[np.ndarray, float, int, int]:
        import cv2

        h, w = image.shape[:2]
        r = min(size / h, size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_h, pad_w = (size - nh) % self.stride, (size - nw) % self.stride
        top, left = pad_h // 2, pad_w // 2
        out = cv2.copyMakeBorder(
            resized,
            top,
            pad_h - top,
            left,
            pad_w - left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return out, r, left, top

    def detect(self, image, page_width, page_height) -> LayoutHints:
        size = max(self.stride, int(image.shape[0] / self.stride) * self.stride)
        pix, ratio, pad_x, pad_y = self._letterbox(image[:, :, ::-1], size)
        tensor = np.transpose(pix, (2, 0, 1))[None].astype(np.float32) / 255.0
        preds = self.session.run(None, {"images": tensor})[0]
        preds = preds[preds[..., 4] > 0.25]
        sx = page_width / image.shape[1]
        sy = page_height / image.shape[0]
        hints = LayoutHints()
        for p in preds:
            x0, y0, x1, y1 = (p[:4] - [pad_x, pad_y, pad_x, pad_y]) / ratio
            label = (
                self.names[int(p[-1])]
                if isinstance(self.names, (list, dict))
                else str(int(p[-1]))
            )
            hints.regions.append(
                (str(label), (x0 * sx, y0 * sy, x1 * sx, y1 * sy), float(p[-2]))
            )
        return hints


def create_detector(name: str, model_path: Optional[Path] = None) -> LayoutDetector:
    if name == "doclayout":
        try:
            return DocLayoutYoloDetector(model_path)
        except Exception as exc:  # pragma: no cover - optional dependency
            log.warning(
                "DocLayout detector unavailable (%s); using heuristics only", exc
            )
    return NullLayoutDetector()
