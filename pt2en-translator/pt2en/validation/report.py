"""QA report assembly and the annotated review PDF."""

from __future__ import annotations

import time
from collections import Counter

import pymupdf

from pt2en import __version__
from pt2en.model import (
    BlockStatus,
    DocumentModel,
    ElementKind,
    Issue,
    PageStrategy,
    Severity,
)
from pt2en.validation.validator import CheckResult

SEVERITY_COLORS = {
    Severity.ERROR: (0.86, 0.15, 0.15),
    Severity.WARNING: (0.95, 0.6, 0.05),
    Severity.INFO: (0.15, 0.45, 0.85),
}


def dedupe_issues(issues: list[Issue]) -> list[Issue]:
    seen = set()
    out = []
    for issue in issues:
        key = (
            issue.severity,
            issue.category,
            issue.page,
            issue.element_id,
            tuple(round(v) for v in issue.bbox) if issue.bbox else None,
            issue.message[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(issue)
    order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    out.sort(key=lambda i: (order[i.severity], i.page if i.page is not None else -1))
    return out


def quality_score(issues: list[Issue]) -> int:
    errors = sum(1 for i in issues if i.severity == Severity.ERROR)
    warnings = sum(1 for i in issues if i.severity == Severity.WARNING)
    return max(0, min(100, 100 - errors * 8 - warnings * 2))


def build_report(
    model: DocumentModel,
    issues: list[Issue],
    checks: list[CheckResult],
    *,
    provider: str,
    options: dict,
    glossary: list[dict],
    stats: dict,
    timings: dict,
    repairs: dict,
) -> dict:
    issues = dedupe_issues(issues)
    counts = Counter(i.severity.value for i in issues)
    blocks = list(model.blocks())
    status_counts = Counter(b.status.value for b in blocks if b.translatable)
    kinds = Counter(b.kind.value for b in blocks)
    images = [
        img
        for p in model.pages
        if p.strategy == PageStrategy.NATIVE
        for img in p.images
    ]
    image_boxes = [box for img in images for box in img.text_boxes if box.translatable]
    score = quality_score(issues)
    if counts.get("error"):
        status = "needs_review"
    elif counts.get("warning"):
        status = "passed_with_warnings"
    else:
        status = "passed"
    return {
        "generator": f"pt2en-translator {__version__}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "document": {
            "title": model.title,
            "pages": model.page_count,
            "scanned_pages": sum(
                1 for p in model.pages if p.strategy == PageStrategy.SCANNED
            ),
            "body_font_size": model.body_size,
        },
        "translation": {"provider": provider, **options},
        "summary": {
            "status": status,
            "score": score,
            "errors": counts.get("error", 0),
            "warnings": counts.get("warning", 0),
            "info": counts.get("info", 0),
        },
        "stats": {
            "text_blocks": len(blocks),
            "translated_blocks": status_counts.get(BlockStatus.TRANSLATED.value, 0),
            "kept_blocks": status_counts.get(BlockStatus.KEPT.value, 0),
            "failed_blocks": status_counts.get(BlockStatus.FAILED.value, 0),
            "headings": kinds.get(ElementKind.HEADING.value, 0),
            "table_cells": kinds.get(ElementKind.TABLE_CELL.value, 0),
            "figure_labels": kinds.get(ElementKind.LABEL.value, 0),
            "display_formulas": kinds.get(ElementKind.FORMULA.value, 0),
            "inline_formulas": sum(len(b.math) for b in blocks),
            "images": len(images),
            "images_translated": sum(1 for i in images if i.status == "translated"),
            "image_labels_translated": sum(
                1
                for b in image_boxes
                if b.status == BlockStatus.TRANSLATED and b.translation != b.text
            ),
            "font_scaled_blocks": sum(1 for b in blocks if b.font_scale < 0.9),
            **stats,
        },
        "checks": [c.to_dict() for c in checks],
        "issues": [i.to_dict() for i in issues],
        "repairs": repairs,
        "glossary": glossary,
        "timings": {k: round(v, 2) for k, v in timings.items()},
    }


def build_review_pdf(output_pdf: bytes, report: dict) -> bytes:
    """The translated PDF with QA findings as annotations and a summary page."""
    doc = pymupdf.open(stream=output_pdf, filetype="pdf")
    for issue in report["issues"]:
        page_no = issue.get("page")
        if page_no is None or page_no >= doc.page_count:
            continue
        page = doc[page_no]
        severity = Severity(issue["severity"])
        color = SEVERITY_COLORS[severity]
        text = f"[{severity.value.upper()}] {issue['category']}: {issue['message']}"
        if issue.get("source_text"):
            text += f"\n\nSource: {issue['source_text']}"
        if issue.get("translated_text"):
            text += f"\n\nTranslation: {issue['translated_text']}"
        if issue.get("suggestion"):
            text += f"\n\nSuggestion: {issue['suggestion']}"
        bbox = issue.get("bbox")
        if bbox:
            rect = pymupdf.Rect(bbox) & page.rect
            if rect.is_empty:
                continue
            annot = page.add_rect_annot(rect + (-1.5, -1.5, 1.5, 1.5))
            annot.set_colors(stroke=color)
            annot.set_border(width=1.2)
            annot.set_opacity(0.85)
        else:
            annot = page.add_text_annot(pymupdf.Point(12, 12), text, icon="Note")
            annot.set_colors(stroke=color)
        annot.set_info(title="pt2en QA", content=text)
        annot.update()

    summary = doc.new_page(
        pno=0,
        width=doc[0].rect.width if doc.page_count else 595,
        height=doc[0].rect.height if doc.page_count else 842,
    )
    _summary_page(summary, report)
    return doc.tobytes(garbage=3, deflate=True)


def _summary_page(page: pymupdf.Page, report: dict) -> None:
    s = report["summary"]
    lines = [
        ("Translation quality report", 18, True),
        (report["document"].get("title") or "", 11, False),
        ("", 6, False),
        (
            f"Status: {s['status'].replace('_', ' ')}   ·   Score: {s['score']}/100",
            12,
            True,
        ),
        (
            f"Errors: {s['errors']}   Warnings: {s['warnings']}   Info: {s['info']}",
            11,
            False,
        ),
        (f"Provider: {report['translation'].get('provider', '')}", 10, False),
        ("", 8, False),
        ("Checks", 13, True),
    ]
    for c in report["checks"]:
        mark = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "skipped": "SKIP"}.get(
            c["status"], c["status"]
        )
        lines.append((f"[{mark}] {c['label']} — {c['details']}", 9, False))
    lines += [("", 8, False), ("Issues by page", 13, True)]
    for issue in report["issues"][:60]:
        page_label = f"p.{issue['page_number']}" if issue.get("page_number") else "doc"
        lines.append(
            (f"{page_label} [{issue['severity']}] {issue['message']}", 8.5, False)
        )
    if len(report["issues"]) > 60:
        lines.append(
            (f"… {len(report['issues']) - 60} more (see the JSON report)", 8.5, False)
        )
    lines.append(("", 6, False))
    lines.append(
        (
            "Annotations on the following pages mark every finding (colour = severity).",
            8.5,
            False,
        )
    )
    rect = page.rect + (40, 40, -40, -40)
    y = rect.y0
    font = pymupdf.Font("helv")
    bold = pymupdf.Font("hebo")
    tw = pymupdf.TextWriter(page.rect)
    for text, size, is_bold in lines:
        f = bold if is_bold else font
        wrapped: list[str] = []
        current = ""
        for word in text.split(" "):
            candidate = (current + " " + word).strip()
            if current and f.text_length(candidate, size) > rect.width:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        wrapped.append(current)
        for line in wrapped:
            y += size * 1.35
            if y > rect.y1:
                break
            if line:
                tw.append((rect.x0, y), line, font=f, fontsize=size)
    tw.write_text(page)


def issue_counts(issues: list[Issue]) -> dict:
    c = Counter(i.severity.value for i in issues)
    return {
        "errors": c.get("error", 0),
        "warnings": c.get("warning", 0),
        "info": c.get("info", 0),
    }
