"""Generates a realistic European-Portuguese course PDF for demos and tests.

The sample exercises every part of the pipeline: headings, styled paragraphs,
lists, inline and display formulas (with a vector fraction bar), a ruled
table, a raster chart with Portuguese labels, a vector flowchart, a vector
chart with a rotated axis title, headers/footers/page numbers, a footnote and
a scanned page.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Optional

import pymupdf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIRS = [
    Path("/usr/share/fonts/truetype/liberation"),
    Path("/usr/share/fonts/truetype/liberation2"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype/libreoffice"),
]


def _find(name: str) -> Optional[str]:
    for d in FONT_DIRS:
        p = d / name
        if p.exists():
            return str(p)
    return None


class _Fonts:
    def __init__(self) -> None:
        def font(file: str, base14: str) -> pymupdf.Font:
            path = _find(file)
            return pymupdf.Font(fontfile=path) if path else pymupdf.Font(base14)

        self.serif = font("LiberationSerif-Regular.ttf", "tiro")
        self.serif_bold = font("LiberationSerif-Bold.ttf", "tibo")
        self.serif_italic = font("LiberationSerif-Italic.ttf", "tiit")
        self.sans = font("LiberationSans-Regular.ttf", "helv")
        self.sans_bold = font("LiberationSans-Bold.ttf", "hebo")
        sym = _find("opens___.ttf")
        self.symbol = pymupdf.Font(fontfile=sym) if sym else pymupdf.Font("symb")
        self.has_open_symbol = bool(sym)


def _pil_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"
    path = _find(name) or _find("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


class _Writer:
    """Tiny typesetting helper on top of TextWriter."""

    def __init__(self, page: pymupdf.Page, fonts: _Fonts):
        self.page = page
        self.f = fonts

    def text(self, x, y, text, font, size, color=(0, 0, 0)):
        tw = pymupdf.TextWriter(self.page.rect)
        tw.append((x, y), text, font=font, fontsize=size)
        tw.write_text(self.page, color=color)
        return x + font.text_length(text, size)

    def runs(self, x, y, runs, size, color=(0, 0, 0)):
        for text, font in runs:
            x = self.text(x, y, text, font, size, color)
        return x

    def paragraph(
        self, x, y, width, runs, size, leading=1.35, justify=True, color=(0, 0, 0)
    ):
        """Greedy line breaking of styled runs; returns the next baseline y."""
        words: list[tuple[str, pymupdf.Font]] = []
        for text, font in runs:
            for i, part in enumerate(text.split(" ")):
                if i > 0:
                    words.append((" ", font))
                if part:
                    words.append((part, font))
        lines: list[list[tuple[str, pymupdf.Font]]] = [[]]
        line_w = 0.0
        for word, font in words:
            w = font.text_length(word, size)
            if word != " " and line_w + w > width and lines[-1]:
                while lines[-1] and lines[-1][-1][0] == " ":
                    lines[-1].pop()
                lines.append([])
                line_w = 0.0
            if word == " " and not lines[-1]:
                continue
            lines[-1].append((word, font))
            line_w += w
        for li, line in enumerate(lines):
            natural = sum(font.text_length(w, size) for w, font in line)
            spaces = sum(1 for w, _ in line if w == " ")
            extra = 0.0
            if justify and li < len(lines) - 1 and spaces:
                extra = (width - natural) / spaces
            cx = x
            for word, font in line:
                if word == " ":
                    cx += font.text_length(" ", size) + extra
                    continue
                cx = self.text(cx, y, word, font, size, color)
            y += size * leading
        return y


def _formula(w: _Writer, x: float, y: float, parts, size: float) -> float:
    """parts: list of (text, kind) where kind in var|op|num|sub|sup."""
    f = w.f
    for text, kind in parts:
        if kind == "var":
            x = w.text(x, y, text, f.serif_italic, size)
        elif kind == "bar":  # variable with an overline (mean), drawn as vector art
            x0 = x
            x = w.text(x, y, text, f.serif_italic, size)
            w.page.draw_line(
                (x0 + size * 0.12, y - size * 0.62),
                (x + size * 0.05, y - size * 0.62),
                width=size * 0.05,
            )
        elif kind == "op":
            x = w.text(x, y, text, f.symbol, size)
        elif kind == "num":
            x = w.text(x, y, text, f.serif, size)
        elif kind == "sub":
            x = w.text(x, y + size * 0.25, text, f.serif_italic, size * 0.7)
        elif kind == "sup":
            x = w.text(x, y - size * 0.38, text, f.serif, size * 0.7)
        elif kind == "space":
            x += size * 0.25
    return x


def _chart_png() -> bytes:
    img = Image.new("RGB", (900, 560), "white")
    d = ImageDraw.Draw(img)
    title = _pil_font(30, bold=True)
    label = _pil_font(22)
    small = _pil_font(20)
    d.text(
        (450, 30),
        "Distribuição das notas por turma",
        fill=(20, 20, 20),
        font=title,
        anchor="mm",
    )
    x0, y0, x1, y1 = 120, 80, 860, 470
    d.line((x0, y1, x1, y1), fill=(0, 0, 0), width=2)
    d.line((x0, y0, x0, y1), fill=(0, 0, 0), width=2)
    for i, v in enumerate(range(0, 31, 5)):
        yy = y1 - (y1 - y0) * v / 30
        d.line((x0 - 6, yy, x0, yy), fill=(0, 0, 0), width=2)
        d.text((x0 - 12, yy), str(v), fill=(0, 0, 0), font=small, anchor="rm")
        if v:
            d.line((x0 + 1, yy, x1, yy), fill=(225, 225, 225), width=1)
    data = [("Turma A", 22, 6), ("Turma B", 17, 9), ("Turma C", 25, 4)]
    for i, (name, ok, ko) in enumerate(data):
        cx = x0 + 120 + i * 240
        for j, (v, col) in enumerate(((ok, (46, 125, 196)), (ko, (230, 120, 40)))):
            bx = cx - 50 + j * 55
            d.rectangle((bx, y1 - (y1 - y0) * v / 30, bx + 50, y1 - 1), fill=col)
        d.text((cx + 3, y1 + 28), name, fill=(0, 0, 0), font=label, anchor="mm")
    # Rotated y-axis title.
    tmp = Image.new("RGB", (300, 40), "white")
    ImageDraw.Draw(tmp).text(
        (150, 20), "Número de alunos", fill=(0, 0, 0), font=label, anchor="mm"
    )
    img.paste(tmp.rotate(90, expand=True), (25, 125))
    d.text((490, 538), "Turma", fill=(0, 0, 0), font=label, anchor="mm")
    # Legend.
    d.rectangle((150, 95, 170, 115), fill=(46, 125, 196))
    d.text((180, 105), "Aprovados", fill=(0, 0, 0), font=small, anchor="lm")
    d.rectangle((330, 95, 350, 115), fill=(230, 120, 40))
    d.text((360, 105), "Reprovados", fill=(0, 0, 0), font=small, anchor="lm")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _header_footer(w: _Writer, page_no: int, total: int) -> None:
    page = w.page
    gray = (0.35, 0.35, 0.35)
    w.text(
        56, 36, "Universidade de Lisboa — Faculdade de Ciências", w.f.sans, 8.5, gray
    )
    w.text(
        page.rect.width - 56 - w.f.sans.text_length("Ano letivo 2025/2026", 8.5),
        36,
        "Ano letivo 2025/2026",
        w.f.sans,
        8.5,
        gray,
    )
    page.draw_line((56, 42), (page.rect.width - 56, 42), color=gray, width=0.5)
    w.text(
        56,
        page.rect.height - 30,
        "Métodos Quantitativos — Capítulo 2",
        w.f.sans,
        8.5,
        gray,
    )
    label = f"Página {page_no} de {total}"
    w.text(
        page.rect.width - 56 - w.f.sans.text_length(label, 8.5),
        page.rect.height - 30,
        label,
        w.f.sans,
        8.5,
        gray,
    )


def _page1(doc: pymupdf.Document, f: _Fonts) -> None:
    page = doc.new_page(width=595, height=842)
    w = _Writer(page, f)
    _header_footer(w, 1, 3)
    x, width = 56.0, 483.0
    w.text(x, 92, "Capítulo 2 — Estatística Descritiva", f.serif_bold, 20)
    w.text(
        x,
        114,
        "Unidade curricular: Métodos Quantitativos para a Gestão",
        f.serif_italic,
        11,
        (0.2, 0.2, 0.45),
    )
    w.text(x, 148, "2.1 Introdução", f.serif_bold, 14)
    y = w.paragraph(
        x,
        170,
        width,
        [
            ("A estatística descritiva tem como objetivo ", f.serif),
            ("resumir", f.serif_bold),
            (" e ", f.serif),
            ("organizar", f.serif_italic),
            (
                " um conjunto de dados, de modo a facilitar a sua interpretação. Neste capítulo, o "
                "estudante irá aprender a calcular medidas de tendência central e de dispersão, bem "
                "como a construir gráficos adequados a cada tipo de variável¹.",
                f.serif,
            ),
        ],
        11,
    )
    y += 6
    # Paragraph with an inline formula.
    xx = w.text(x, y, "A média amostral de ", f.serif, 11)
    xx = _formula(w, xx, y, [("n", "var")], 11)
    xx = w.text(xx, y, " observações é dada por ", f.serif, 11)
    xx = _formula(
        w,
        xx,
        y,
        [
            ("x", "bar"),
            ("space", "space"),
            ("=", "op"),
            ("space", "space"),
            ("(", "num"),
            ("1", "num"),
            ("/", "num"),
            ("n", "var"),
            (")", "num"),
            ("space", "space"),
            ("Σ", "op"),
            ("x", "var"),
            ("i", "sub"),
        ],
        11,
    )
    w.text(xx, y, ", onde o índice percorre todas as", f.serif, 11)
    y += 11 * 1.35
    w.text(x, y, "observações da amostra recolhida pelo docente.", f.serif, 11)
    y += 11 * 1.35 + 8
    w.text(
        x,
        y,
        "A variância amostral calcula-se através da seguinte expressão:",
        f.serif,
        11,
    )
    y += 30
    # Display formula with a real fraction bar: s² = Σ (xi − x̄)² / (n − 1)
    cx = 230
    xx = _formula(
        w,
        cx,
        y,
        [
            ("s", "var"),
            ("2", "sup"),
            ("space", "space"),
            ("=", "op"),
            ("space", "space"),
        ],
        12,
    )
    num_x = xx + 4
    nx = _formula(
        w,
        num_x,
        y - 9,
        [
            ("Σ", "op"),
            ("(", "num"),
            ("x", "var"),
            ("i", "sub"),
            ("space", "space"),
            ("−", "op"),
            ("space", "space"),
            ("x", "bar"),
            (")", "num"),
            ("2", "sup"),
        ],
        12,
    )
    page.draw_line((num_x - 2, y - 4), (nx + 2, y - 4), width=0.6)
    _formula(
        w,
        num_x + (nx - num_x) / 2 - 18,
        y + 9,
        [
            ("n", "var"),
            ("space", "space"),
            ("−", "op"),
            ("space", "space"),
            ("1", "num"),
        ],
        12,
    )
    w.text(500, y, "(2.1)", f.serif, 11)
    y += 36
    w.text(x, y, "2.2 Medidas de tendência central", f.serif_bold, 14)
    y += 22
    for item in (
        [
            ("Média: ", f.serif_bold),
            ("soma dos valores a dividir pelo número de observações;", f.serif),
        ],
        [
            ("Mediana: ", f.serif_bold),
            ("valor central da amostra depois de ordenada;", f.serif),
        ],
        [
            ("Moda: ", f.serif_bold),
            ("valor que ocorre com maior frequência na amostra.", f.serif),
        ],
    ):
        w.text(x + 10, y, "•", f.serif, 11)
        y = w.paragraph(x + 24, y, width - 24, item, 11, justify=False)
    y += 8
    w.text(
        x,
        y,
        "Tabela 2.1 – Classificações obtidas pelos alunos no teste intercalar",
        f.serif_italic,
        10,
    )
    y += 8
    rows = [
        ("Aluno", "Nota (0–20)", "Situação"),
        ("Ana Sousa", "15,5", "Aprovado"),
        ("Bruno Lopes", "9,2", "Reprovado"),
        ("Carla Dias", "12,0", "Aprovado"),
    ]
    col_x = [x, x + 170, x + 300, x + 483]
    row_h = 20
    for r, row in enumerate(rows):
        top = y + r * row_h
        if r == 0:
            page.draw_rect(
                pymupdf.Rect(col_x[0], top, col_x[-1], top + row_h),
                color=None,
                fill=(0.9, 0.92, 0.96),
            )
        for c, cell in enumerate(row):
            font = f.serif_bold if r == 0 else f.serif
            w.text(col_x[c] + 6, top + 14, cell, font, 10.5)
    for r in range(len(rows) + 1):
        page.draw_line((col_x[0], y + r * row_h), (col_x[-1], y + r * row_h), width=0.6)
    for cxx in col_x:
        page.draw_line((cxx, y), (cxx, y + len(rows) * row_h), width=0.6)
    page.draw_line((x, 760), (x + 140, 760), width=0.5)
    w.text(
        x,
        772,
        "¹ Os dados utilizados foram recolhidos no primeiro semestre.",
        f.serif,
        8.5,
    )


def _page2(doc: pymupdf.Document, f: _Fonts) -> None:
    page = doc.new_page(width=595, height=842)
    w = _Writer(page, f)
    _header_footer(w, 2, 3)
    x, width = 56.0, 483.0
    w.text(x, 80, "2.3 Representação gráfica", f.serif_bold, 14)
    y = w.paragraph(
        x,
        102,
        width,
        [
            (
                "O gráfico de barras seguinte compara o número de alunos aprovados e reprovados em "
                "cada turma. Repare que a turma C apresenta a maior taxa de aprovação.",
                f.serif,
            )
        ],
        11,
    )
    page.insert_image(pymupdf.Rect(97, y, 497, y + 249), stream=_chart_png())
    y += 262
    w.text(150, y, "Figura 2.1 – Distribuição das notas por turma.", f.serif_italic, 10)
    y += 26
    w.text(x, y, "2.4 Etapas de um estudo estatístico", f.serif_bold, 14)
    y += 18
    boxes = ["Recolha de dados", "Organização", "Análise", "Conclusões"]
    bw, bh, gap = 100, 38, 27
    for i, label in enumerate(boxes):
        bx = x + i * (bw + gap)
        rect = pymupdf.Rect(bx, y, bx + bw, y + bh)
        page.draw_rect(rect, color=(0.18, 0.4, 0.7), fill=(0.88, 0.93, 1.0), width=1)
        tw = f.sans_bold.text_length(label, 9.5)
        w.text(
            bx + (bw - tw) / 2,
            y + bh / 2 + 3.5,
            label,
            f.sans_bold,
            9.5,
            (0.1, 0.2, 0.4),
        )
        if i < len(boxes) - 1:
            ax = bx + bw
            page.draw_line((ax + 2, y + bh / 2), (ax + gap - 4, y + bh / 2), width=1.2)
            page.draw_polyline(
                [
                    (ax + gap - 9, y + bh / 2 - 4),
                    (ax + gap - 3, y + bh / 2),
                    (ax + gap - 9, y + bh / 2 + 4),
                ],
                width=1.2,
            )
    y += bh + 30
    # Vector line chart with a rotated axis title.
    ox, oy, cw, ch = x + 50, y + 150, 360, 130
    page.draw_line((ox, oy), (ox + cw, oy), width=0.8)
    page.draw_line((ox, oy), (ox, oy - ch), width=0.8)
    pts = [(0, 40), (1, 65), (2, 58), (3, 90), (4, 110)]
    meses = ["Set.", "Out.", "Nov.", "Dez.", "Jan."]
    poly = [(ox + 30 + i * 75, oy - v) for i, v in pts]
    page.draw_polyline(poly, color=(0.8, 0.1, 0.1), width=1.5)
    for (px, py), mes in zip(poly, meses):
        page.draw_circle((px, py), 2.2, color=(0.8, 0.1, 0.1), fill=(0.8, 0.1, 0.1))
        w.text(px - f.sans.text_length(mes, 8.5) / 2, oy + 12, mes, f.sans, 8.5)
    w.text(ox + cw / 2 - 60, y + 4, "Evolução da média mensal", f.sans_bold, 10)
    w.text(ox + cw / 2 - 10, oy + 26, "Mês", f.sans, 9)
    tw = pymupdf.TextWriter(page.rect)
    tw.append((ox - 10, oy - 30), "Média (valores)", font=f.sans, fontsize=9)
    tw.write_text(page, morph=(pymupdf.Point(ox - 10, oy - 30), pymupdf.Matrix(90)))
    y = oy + 46
    w.paragraph(
        x,
        y,
        width,
        [
            (
                "Em síntese, a análise descritiva constitui o primeiro passo de qualquer estudo, "
                "permitindo detetar valores atípicos e formular hipóteses para a inferência estatística.",
                f.serif,
            )
        ],
        11,
    )


def _scanned_page(doc: pymupdf.Document, f: _Fonts) -> None:
    """Typeset a page, rasterise it with noise, and insert the image only."""
    tmp = pymupdf.open()
    page = tmp.new_page(width=595, height=842)
    w = _Writer(page, f)
    x, width = 70.0, 455.0
    w.text(x, 90, "Exercícios propostos", f.serif_bold, 16)
    y = w.paragraph(
        x,
        125,
        width,
        [
            (
                "1. Uma empresa registou o número de reclamações recebidas durante doze semanas "
                "consecutivas. Calcule a média, a mediana e a moda das observações e comente os "
                "resultados obtidos.",
                f.serif,
            )
        ],
        12,
    )
    y = w.paragraph(
        x,
        y + 10,
        width,
        [
            (
                "2. Construa um histograma com os dados do exercício anterior e indique se a "
                "distribuição é simétrica ou assimétrica.",
                f.serif,
            )
        ],
        12,
    )
    y = w.paragraph(
        x,
        y + 10,
        width,
        [
            (
                "3. Explique por que razão a mediana é uma medida mais robusta do que a média na "
                "presença de valores extremos.",
                f.serif,
            )
        ],
        12,
    )
    w.text(x, y + 30, "Bom trabalho!", f.serif_italic, 12)
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    rnd = random.Random(7)
    img = img.rotate(0.25, fillcolor=255, resample=Image.BICUBIC)
    px = img.load()
    for _ in range(4000):
        px[rnd.randrange(img.width), rnd.randrange(img.height)] = rnd.randrange(
            150, 230
        )
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=80)
    out = doc.new_page(width=595, height=842)
    out.insert_image(out.rect, stream=buf.getvalue())


def make_sample_pdf(path: Optional[Path | str] = None) -> bytes:
    fonts = _Fonts()
    doc = pymupdf.open()
    doc.set_metadata(
        {"title": "Capítulo 2 — Estatística Descritiva", "author": "Exemplo"}
    )
    _page1(doc, fonts)
    _page2(doc, fonts)
    _scanned_page(doc, fonts)
    data = doc.tobytes(garbage=3, deflate=True)
    if path:
        Path(path).write_bytes(data)
    return data


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "curso_exemplo_pt.pdf"
    make_sample_pdf(target)
    print(f"Sample written to {target}")
