"""Structured document model shared by every pipeline stage.

Coordinates are PyMuPDF page coordinates (origin top-left, y grows downwards),
expressed as ``(x0, y0, x1, y1)`` tuples so the model stays serialisable and
independent from the PDF library.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

Rect = tuple[float, float, float, float]
Point = tuple[float, float]


class ElementKind(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    FOOTNOTE = "footnote"
    TABLE_CELL = "table_cell"
    FORMULA = "formula"
    LABEL = "label"
    OTHER = "other"


class PageStrategy(str, Enum):
    NATIVE = "native"
    SCANNED = "scanned"


class BlockStatus(str, Enum):
    PENDING = "pending"
    TRANSLATED = "translated"
    KEPT = "kept"  # intentionally left unchanged (numbers, English, names...)
    FAILED = "failed"  # could not be translated reliably -> original kept


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class TextStyle:
    font: str = ""
    size: float = 10.0
    bold: bool = False
    italic: bool = False
    serif: bool = True
    mono: bool = False
    color: int = 0
    alpha: float = 1.0
    superscript: bool = False
    subscript: bool = False
    underline: bool = False
    strike: bool = False

    def visual_key(self) -> tuple:
        """Attributes that matter when deciding whether two runs look different."""
        return (
            self.bold,
            self.italic,
            self.mono,
            self.underline,
            self.strike,
            self.superscript,
            self.subscript,
            self.color,
            self.font,
        )

    @property
    def rgb(self) -> tuple[float, float, float]:
        c = self.color
        return (((c >> 16) & 255) / 255.0, ((c >> 8) & 255) / 255.0, (c & 255) / 255.0)

    def copy(self, **changes: Any) -> "TextStyle":
        data = asdict(self)
        data.update(changes)
        return TextStyle(**data)


@dataclass
class Glyph:
    c: str
    bbox: Rect
    origin: Point
    size: float = 10.0

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0


@dataclass
class Run:
    """Contiguous glyphs of one line sharing a style and a role (text or math)."""

    glyphs: list[Glyph]
    style: TextStyle
    is_math: bool = False
    region_id: Optional[str] = None

    @property
    def text(self) -> str:
        return "".join(g.c for g in self.glyphs)

    @property
    def bbox(self) -> Rect:
        return union_rects([g.bbox for g in self.glyphs])


@dataclass
class Line:
    runs: list[Run]
    bbox: Rect
    direction: Point = (1.0, 0.0)
    wmode: int = 0
    baseline: float = 0.0
    size: float = 10.0
    ocr_confidence: Optional[float] = None

    @property
    def text(self) -> str:
        return "".join(r.text for r in self.runs)

    @property
    def glyphs(self) -> list[Glyph]:
        return [g for r in self.runs for g in r.glyphs]

    @property
    def is_horizontal(self) -> bool:
        return abs(self.direction[0] - 1.0) < 1e-3 and abs(self.direction[1]) < 1e-3


@dataclass
class MathRegion:
    """A formula fragment that must be reproduced exactly as in the source."""

    id: str
    page: int
    bbox: Rect
    text: str
    baseline: float
    glyph_count: int = 0


@dataclass
class TextBlock:
    id: str
    page: int
    kind: ElementKind
    bbox: Rect
    lines: list[Line]
    style: TextStyle
    align: str = "left"
    line_pitch: float = 0.0
    first_indent: float = 0.0
    rest_indent: float = 0.0
    prefix_text: str = ""  # bullet / enumerator kept verbatim
    prefix_bbox: Optional[Rect] = None
    text_x0: Optional[float] = None
    translatable: bool = True
    source_text: str = ""  # plain text, for QA and reporting
    source_markup: str = ""  # markup sent to the translator
    styles: dict[str, TextStyle] = field(default_factory=dict)
    math: dict[str, MathRegion] = field(default_factory=dict)
    protected: dict[str, str] = field(default_factory=dict)
    translation: Optional[str] = None
    status: BlockStatus = BlockStatus.PENDING
    region: Optional[Rect] = None
    heading_level: int = 0
    rotation: float = 0.0
    source: str = "native"  # native | ocr
    ocr_confidence: Optional[float] = None
    notes: list[str] = field(default_factory=list)
    placed_bbox: Optional[Rect] = None
    font_scale: float = 1.0
    section: str = ""

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass
class OCRWord:
    text: str
    bbox: Rect  # in the coordinate system of the OCR'd raster (pixels)
    confidence: float
    block: int = 0
    paragraph: int = 0
    line: int = 0


@dataclass
class ImageTextBox:
    """A text label found inside a raster image (pixel coordinates)."""

    id: str
    text: str
    bbox: Rect
    lines: list[list[OCRWord]]
    confidence: float
    translatable: bool = True
    translation: Optional[str] = None
    status: BlockStatus = BlockStatus.PENDING
    notes: list[str] = field(default_factory=list)
    rotation: int = 0  # 0 = horizontal, 90 = reads bottom-to-top, 270 = top-to-bottom


@dataclass
class ImageElement:
    id: str
    page: int
    xref: int
    bbox: Rect
    width: int
    height: int
    smask: int = 0
    transform: tuple = (1, 0, 0, 1, 0, 0)
    has_text: bool = False
    text_boxes: list[ImageTextBox] = field(default_factory=list)
    status: str = "untouched"  # untouched | translated | skipped | failed
    notes: list[str] = field(default_factory=list)


@dataclass
class PageModel:
    number: int
    width: float
    height: float
    strategy: PageStrategy = PageStrategy.NATIVE
    blocks: list[TextBlock] = field(default_factory=list)
    images: list[ImageElement] = field(default_factory=list)
    drawings_count: int = 0
    figure_rects: list[Rect] = field(default_factory=list)
    table_rects: list[Rect] = field(default_factory=list)
    drawing_rects: list[Rect] = field(default_factory=list)
    text_chars: int = 0
    invisible_chars: int = 0
    image_coverage: float = 0.0
    scan_dpi: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def rect(self) -> Rect:
        return (0.0, 0.0, self.width, self.height)


@dataclass
class Issue:
    severity: Severity
    category: str
    message: str
    page: Optional[int] = None
    bbox: Optional[Rect] = None
    element_id: Optional[str] = None
    source_text: Optional[str] = None
    translated_text: Optional[str] = None
    suggestion: Optional[str] = None
    auto_fixed: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["severity"] = self.severity.value
        data["page_number"] = None if self.page is None else self.page + 1
        return data


@dataclass
class DocumentModel:
    page_count: int
    pages: list[PageModel] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    title: str = ""
    body_size: float = 10.0
    issues: list[Issue] = field(default_factory=list)

    def blocks(self):
        for page in self.pages:
            yield from page.blocks

    def block_by_id(self, block_id: str) -> Optional[TextBlock]:
        for block in self.blocks():
            if block.id == block_id:
                return block
        return None

    def images(self):
        for page in self.pages:
            yield from page.images

    def add_issue(self, issue: Issue) -> None:
        self.issues.append(issue)


# ---------------------------------------------------------------- geometry
def union_rects(rects) -> Rect:
    rects = list(rects)
    if not rects:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def rect_area(r: Rect) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def rect_intersection(a: Rect, b: Rect) -> Rect:
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def intersection_area(a: Rect, b: Rect) -> float:
    return rect_area(rect_intersection(a, b))


def rect_contains_point(r: Rect, x: float, y: float, tol: float = 0.0) -> bool:
    return r[0] - tol <= x <= r[2] + tol and r[1] - tol <= y <= r[3] + tol


def rect_center(r: Rect) -> Point:
    return ((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0)


def expand_rect(r: Rect, dx: float, dy: Optional[float] = None) -> Rect:
    dy = dx if dy is None else dy
    return (r[0] - dx, r[1] - dy, r[2] + dx, r[3] + dy)


def overlap_ratio(a: Rect, b: Rect) -> float:
    """Intersection area relative to the smaller rectangle."""
    smaller = min(rect_area(a), rect_area(b))
    if smaller <= 0:
        return 0.0
    return intersection_area(a, b) / smaller
