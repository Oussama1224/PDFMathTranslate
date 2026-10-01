"""Translation orchestration.

Responsibilities:

* turn text blocks / image labels into translation items (inline markup),
* build the terminology glossary (base + document + user overrides),
* de-duplicate identical segments and reuse the translation memory,
* batch segments with context and translate them concurrently,
* validate every result (markup, formulas, numbers, glossary, residual
  Portuguese) and retry with targeted feedback,
* never silently corrupt content: unrecoverable segments keep their original
  text and are reported as issues.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from pt2en.config import Settings
from pt2en.errors import ProviderConfigurationError
from pt2en.model import (
    BlockStatus,
    DocumentModel,
    ElementKind,
    ImageTextBox,
    Issue,
    Rect,
    Severity,
    TextBlock,
    TextStyle,
)
from pt2en.pipeline.options import JobOptions
from pt2en.translation import markup as mk
from pt2en.translation.base import (
    Segment,
    TranslationContext,
    TranslationError,
    TranslationRefused,
    Translator,
)
from pt2en.translation.glossary import Glossary, extract_candidates, parse_user_glossary
from pt2en.translation.language import (
    is_probably_english,
    looks_untranslated,
    portuguese_residue,
)
from pt2en.translation.memory import TranslationMemory

log = logging.getLogger(__name__)

SHORT_KINDS = {
    ElementKind.LABEL.value,
    ElementKind.TABLE_CELL.value,
    ElementKind.HEADER.value,
    ElementKind.FOOTER.value,
    ElementKind.PAGE_NUMBER.value,
    "image_label",
}
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_PAGE_RE = re.compile(r"^\s*P[áa]g(?:ina|\.)?\s*(\d+)\s*(?:de|/)\s*(\d+)\s*$", re.I)
_PAGE_SINGLE_RE = re.compile(r"^\s*P[áa]g(?:ina|\.)?\s*(\d+)\s*$", re.I)


@dataclass
class TranslationItem:
    id: str
    markup: str
    kind: str
    page: Optional[int] = None
    bbox: Optional[Rect] = None
    section: str = ""
    math_text: dict[str, str] = field(default_factory=dict)
    styles: dict[str, TextStyle] = field(default_factory=dict)
    protected: dict[str, str] = field(default_factory=dict)
    max_ratio: Optional[float] = None
    translation: Optional[str] = None
    status: BlockStatus = BlockStatus.PENDING
    notes: list[str] = field(default_factory=list)

    @property
    def math_ids(self) -> set[str]:
        return set(self.math_text)

    def plain(self, markup: Optional[str] = None) -> str:
        return mk.plain_text(
            markup if markup is not None else self.markup,
            self.protected,
            self.math_text,
        )


@dataclass
class _Unit:
    """A unique source segment (dedup of identical items)."""

    uid: str
    item: TranslationItem  # representative
    members: list[TranslationItem]
    result: Optional[str] = None
    status: BlockStatus = BlockStatus.PENDING
    attempts: int = 0
    feedback: str = ""
    warnings: list[str] = field(default_factory=list)
    preceding: list[str] = field(default_factory=list)


def item_from_block(block: TextBlock) -> TranslationItem:
    if not block.source_markup:
        mk.encode_block(block)
    return TranslationItem(
        id=block.id,
        markup=block.source_markup,
        kind=block.kind.value,
        page=block.page,
        bbox=block.bbox,
        section=block.section,
        math_text={k: v.text for k, v in block.math.items()},
        styles=block.styles,
        protected=block.protected,
        max_ratio=1.6 if block.kind.value in SHORT_KINDS else None,
    )


def item_from_image_box(box: ImageTextBox, page: int, bbox: Rect) -> TranslationItem:
    return TranslationItem(
        id=box.id,
        markup=mk.escape(box.text),
        kind="image_label",
        page=page,
        bbox=bbox,
        max_ratio=1.4,
    )


class TranslationEngine:
    def __init__(
        self,
        translator: Translator,
        settings: Settings,
        options: JobOptions,
        *,
        memory: Optional[TranslationMemory] = None,
        issues: Optional[list[Issue]] = None,
        cancel_check: Optional[Callable[[], None]] = None,
    ):
        self.translator = translator
        self.settings = settings
        self.options = options
        self.memory = memory
        self.issues = issues if issues is not None else []
        self.cancel_check = cancel_check or (lambda: None)
        self.glossary = Glossary.with_base()
        self.user_entries = parse_user_glossary(options.glossary_text)
        self.glossary.extend(self.user_entries)
        self.document_title = ""
        self.stats = Counter()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ context
    @property
    def context(self) -> TranslationContext:
        return TranslationContext(
            document_title=self.document_title,
            style=self.options.style,
            variant=self.options.english_variant,
            preserve_terminology=self.options.preserve_terminology,
            localize_numbers=self.options.localize_numbers,
            glossary=list(self.glossary.entries.values()),
        )

    def namespace(self, kind_group: str) -> str:
        o = self.options
        return "|".join(
            [
                "v1",
                self.translator.describe(),
                o.style.value,
                o.english_variant.value,
                str(o.preserve_terminology),
                str(o.localize_numbers),
                self.glossary.fingerprint(),
                kind_group,
            ]
        )

    # ----------------------------------------------------------- glossary
    def build_document_glossary(
        self, doc: DocumentModel, extra_texts: Iterable[str] = ()
    ) -> None:
        self.document_title = doc.title
        if not (self.options.extract_glossary and self.translator.is_llm):
            return
        texts: list[tuple[str, float]] = []
        for block in doc.blocks():
            if not block.translatable:
                continue
            weight = 3.0 if block.kind == ElementKind.HEADING else 1.0
            texts.append((block.source_text or block.text, weight))
            for line in block.lines:
                for run in line.runs:
                    if not run.is_math and run.style.bold and not block.style.bold:
                        texts.append((run.text, 2.0))
        texts.extend((t, 1.0) for t in extra_texts)
        candidates = extract_candidates(texts, limit=120)
        if not candidates:
            return
        try:
            entries = self.translator.extract_terminology(candidates, self.context)
        except ProviderConfigurationError:
            raise
        except TranslationError as exc:
            log.warning("terminology extraction failed: %s", exc)
            self.issues.append(
                Issue(
                    Severity.INFO,
                    "terminology",
                    "Automatic glossary extraction failed; "
                    "terminology consistency relies on the base glossary only.",
                )
            )
            return
        self.glossary.extend(entries)
        # User overrides always win.
        self.glossary.extend(self.user_entries)
        self.stats["glossary_document_terms"] = len(entries)

    # -------------------------------------------------------- translation
    def translate_items(
        self,
        items: list[TranslationItem],
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        progress = progress or (lambda f, m: None)
        units = self._prepare_units(items)
        pending = [u for u in units if u.status == BlockStatus.PENDING]
        total = max(len(pending), 1)
        done = 0
        rounds = 1 + max(0, self.settings.max_segment_retries)
        for round_no in range(rounds):
            todo = [u for u in pending if u.status == BlockStatus.PENDING]
            if not todo:
                break
            batches = self._batches(todo)
            workers = max(1, min(self.settings.translation_concurrency, len(batches)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._run_batch, b): b for b in batches}
                for fut in concurrent.futures.as_completed(futures):
                    batch = futures[fut]
                    try:
                        fut.result()
                    except ProviderConfigurationError:
                        for f in futures:
                            f.cancel()
                        raise
                    finished = sum(1 for u in batch if u.status != BlockStatus.PENDING)
                    done += finished
                    progress(
                        min(done / total, 1.0),
                        f"Translated {min(done, total)}/{total} segments",
                    )
            if round_no == rounds - 1:
                for u in todo:
                    if u.status == BlockStatus.PENDING:
                        self._accept_or_fail(u, final=True)
        self._fan_out(units)
        progress(1.0, "Translation complete")

    def retranslate(self, pairs: list[tuple[TranslationItem, str]]) -> None:
        """Translate items again with reviewer feedback (QA repair pass).

        Items keep their previous translation unless the new attempt passes
        every validation check.
        """
        units = [
            _Unit(uid=f"r{i}", item=item, members=[item], feedback=note)
            for i, (item, note) in enumerate(pairs)
        ]
        for _ in range(1 + max(0, self.settings.max_segment_retries)):
            todo = [u for u in units if u.status == BlockStatus.PENDING]
            if not todo:
                break
            for batch in self._batches(todo):
                self._run_batch(batch)
        for u in units:
            if u.status == BlockStatus.TRANSLATED and u.result is not None:
                u.item.translation = u.result
                u.item.status = BlockStatus.TRANSLATED
            else:
                u.item.status = BlockStatus.PENDING

    # ----------------------------------------------------------- internals
    def _prepare_units(self, items: list[TranslationItem]) -> list[_Unit]:
        by_key: dict[tuple, _Unit] = {}
        units: list[_Unit] = []
        recent: list[str] = []
        for item in items:
            plain = item.plain()
            # Rule-based and trivial cases never reach the translator.
            direct = self._direct_translation(item, plain)
            if direct is not None:
                item.translation, item.status = direct
                self.stats["rule_based"] += 1
                continue
            group = "short" if item.kind in SHORT_KINDS else "text"
            key = (group, item.markup)
            if key in by_key:
                by_key[key].members.append(item)
                self.stats["deduplicated"] += 1
                continue
            unit = _Unit(
                uid=f"u{len(units)}",
                item=item,
                members=[item],
                preceding=list(recent[-3:]),
            )
            if group == "text":
                recent.append(plain[:400])
            cached = (
                self.memory.get(
                    self.memory.make_key(self.namespace(group), item.markup)
                )
                if self.memory
                else None
            )
            if cached is not None and not self._problems(unit, cached):
                unit.result, unit.status = cached, BlockStatus.TRANSLATED
                self.stats["memory_hits"] += 1
            by_key[key] = unit
            units.append(unit)
        return units

    def _direct_translation(self, item: TranslationItem, plain: str):
        text = plain.strip()
        if not text or not re.search(r"[A-Za-zÀ-ÿ]{2,}", text):
            return item.markup, BlockStatus.KEPT
        m = _PAGE_RE.match(text)
        if m:
            return f"Page {m.group(1)} of {m.group(2)}", BlockStatus.TRANSLATED
        m = _PAGE_SINGLE_RE.match(text)
        if m:
            return f"Page {m.group(1)}", BlockStatus.TRANSLATED
        if is_probably_english(text):
            return item.markup, BlockStatus.KEPT
        return None

    def _batches(self, units: list[_Unit]) -> list[list[_Unit]]:
        batches: list[list[_Unit]] = []
        current: list[_Unit] = []
        size = 0
        limit_chars = self.settings.batch_max_chars
        limit_n = self.settings.batch_max_segments
        for u in units:
            n = len(u.item.markup)
            if current and (size + n > limit_chars or len(current) >= limit_n):
                batches.append(current)
                current, size = [], 0
            current.append(u)
            size += n
        if current:
            batches.append(current)
        return batches

    def _run_batch(self, batch: list[_Unit]) -> None:
        self.cancel_check()
        segments = [
            Segment(
                id=u.uid,
                text=u.item.markup,
                kind=u.item.kind,
                section=u.item.section,
                note=u.feedback,
                max_ratio=u.item.max_ratio,
                preceding=u.preceding,
            )
            for u in batch
        ]
        try:
            results = self._call_provider(segments)
        except TranslationRefused as exc:
            for u in batch:
                u.attempts += 1
                u.status = BlockStatus.FAILED
                u.warnings.append(str(exc))
            return
        except TranslationError as exc:
            if len(batch) > 1:
                mid = len(batch) // 2
                self._run_batch(batch[:mid])
                self._run_batch(batch[mid:])
                return
            u = batch[0]
            u.attempts += 1
            u.feedback = ""
            u.warnings.append(f"translation request failed: {exc}")
            if u.attempts > self.settings.max_segment_retries:
                u.status = BlockStatus.FAILED
            return
        for u in batch:
            u.attempts += 1
            out = results.get(u.uid)
            if out is None:
                u.feedback = (
                    "This segment was missing from the previous answer; translate it."
                )
                continue
            out = self._normalise_output(out)
            problems = self._problems(u, out)
            if not problems:
                self._accept(u, out)
                continue
            u.result = out  # keep the best attempt for the final decision
            u.feedback = (
                "Your previous translation had problems: "
                + "; ".join(problems)
                + ". Fix them while translating the source again."
            )

    def _call_provider(self, segments: list[Segment]) -> dict[str, str]:
        ctx = self.context
        if self.translator.supports_markup:
            return self._with_retries(
                lambda: self.translator.translate_batch(segments, ctx)
            )
        # Providers without markup support: translate text chunks between placeholders.
        chunks: list[Segment] = []
        layout: dict[str, list] = {}
        for s in segments:
            parts = re.split(r"(<[mx]\d+\s*/>)", mk.strip_tags(s.text))
            layout[s.id] = []
            for k, part in enumerate(parts):
                if re.fullmatch(r"<[mx]\d+\s*/>", part):
                    layout[s.id].append(("ph", part))
                elif part.strip():
                    cid = f"{s.id}.{k}"
                    lead = part[: len(part) - len(part.lstrip())]
                    trail = part[len(part.rstrip()) :]
                    layout[s.id].append(("chunk", (cid, lead, trail)))
                    chunks.append(
                        Segment(id=cid, text=mk.unescape(part.strip()), kind=s.kind)
                    )
                else:
                    layout[s.id].append(("ph", part))
        translated = (
            self._with_retries(lambda: self.translator.translate_batch(chunks, ctx))
            if chunks
            else {}
        )
        out = {}
        for s in segments:
            pieces = []
            for kind, value in layout[s.id]:
                if kind == "ph":
                    pieces.append(value)
                else:
                    cid, lead, trail = value
                    pieces.append(lead + mk.escape(translated.get(cid, "")) + trail)
            out[s.id] = "".join(pieces)
        return out

    def _with_retries(self, fn):
        delays = (2.0, 6.0, 15.0)
        for attempt in range(len(delays) + 1):
            self.cancel_check()
            try:
                return fn()
            except (TranslationRefused, ProviderConfigurationError):
                raise
            except TranslationError as exc:
                if "truncated" in str(exc).lower() or attempt == len(delays):
                    raise
                log.warning(
                    "translation request failed (%s); retrying in %ss",
                    exc,
                    delays[attempt],
                )
                time.sleep(delays[attempt])
        raise TranslationError("unreachable")  # pragma: no cover

    @staticmethod
    def _normalise_output(text: str) -> str:
        text = text.strip()
        # Some models self-close tags with a space or emit XML entities differently.
        text = re.sub(r"<([mx]\d+)\s*/\s*>", r"<\1/>", text)
        text = text.replace("&nbsp;", " ")
        return re.sub(r"[ \t]{2,}", " ", text)

    def _problems(self, unit: _Unit, out: str) -> list[str]:
        item = unit.item
        problems = mk.validate_structure(
            item.markup, out, item.styles, item.protected, item.math_ids
        )
        if any(
            p.startswith(("missing", "unexpected", "duplicated", "invalid"))
            for p in problems
        ):
            return problems
        src_plain = item.plain()
        out_plain = item.plain(out)
        if not out_plain.strip():
            return ["the translation is empty"]
        if not self.options.localize_numbers:
            src_nums = Counter(_NUMBER_RE.findall(src_plain))
            out_nums = Counter(_NUMBER_RE.findall(out_plain))
            lost = src_nums - out_nums
            if lost:
                problems.append(
                    "numbers changed or missing: "
                    + ", ".join(sorted(lost))
                    + " (keep numbers exactly as in the source)"
                )
        missing_terms = self.glossary.check_translation(src_plain, out_plain)
        if missing_terms:
            problems.append(
                "mandatory glossary terms not used: "
                + "; ".join(
                    f"{e.pt} => {e.pt if e.keep else e.en}" for e in missing_terms
                )
            )
        allowed = self.glossary.kept_terms() | frozenset(
            w.lower() for w in re.findall(r"\b[A-ZÀ-Ý][\wÀ-ÿ]+", src_plain)
        )
        lexicon = self._lexicon()
        if looks_untranslated(out_plain, allowed, lexicon):
            residue = portuguese_residue(out_plain, allowed, lexicon)
            problems.append(
                "Portuguese words left untranslated: " + ", ".join(residue[:8])
            )
        return problems

    def _lexicon(self) -> frozenset[str]:
        lex = getattr(self, "_lexicon_cache", None)
        if lex is None or lex[0] != len(self.glossary.entries):
            lex = (len(self.glossary.entries), self.glossary.portuguese_lexicon())
            self._lexicon_cache = lex
        return lex[1]

    def _accept(self, unit: _Unit, out: str) -> None:
        unit.result = out
        unit.status = BlockStatus.TRANSLATED
        if self.memory:
            group = "short" if unit.item.kind in SHORT_KINDS else "text"
            self.memory.put(
                self.memory.make_key(self.namespace(group), unit.item.markup),
                unit.item.markup,
                out,
            )

    def _accept_or_fail(self, unit: _Unit, final: bool) -> None:
        """Last chance for a unit whose attempts all had problems."""
        out = unit.result
        if out is None:
            unit.status = BlockStatus.FAILED
            return
        item = unit.item
        problems = self._problems(unit, out)
        structural = [
            p
            for p in problems
            if p.startswith(("missing", "unexpected", "duplicated", "invalid"))
        ]
        if structural:
            # Style tags are recoverable: drop them, keep placeholders.
            stripped = mk.strip_tags(out)
            if not mk.validate_structure(
                mk.strip_tags(item.markup), stripped, {}, item.protected, item.math_ids
            ):
                unit.result = stripped
                unit.status = BlockStatus.TRANSLATED
                unit.warnings.append(
                    "inline formatting (bold/italic) could not be preserved"
                )
                return
            unit.status = BlockStatus.FAILED
            unit.warnings.extend(structural)
            return
        unit.status = BlockStatus.TRANSLATED
        unit.warnings.extend(problems)

    @staticmethod
    def _same_text(a: str, b: str) -> bool:
        norm = (
            lambda t: re.sub(r"\s+", " ", mk.strip_tags(t)).strip().casefold()
        )  # noqa: E731
        return norm(a) == norm(b)

    def _fan_out(self, units: list[_Unit]) -> None:
        for unit in units:
            if (
                unit.status == BlockStatus.TRANSLATED
                and unit.result is not None
                and self._same_text(unit.result, unit.item.markup)
            ):
                # Already English / names / codes: leave the original untouched.
                unit.status = BlockStatus.KEPT
                self.stats["unchanged"] += 1
            for item in unit.members:
                if unit.status == BlockStatus.TRANSLATED and unit.result is not None:
                    item.translation = unit.result
                    item.status = BlockStatus.TRANSLATED
                else:
                    item.translation = item.markup
                    item.status = (
                        BlockStatus.FAILED
                        if unit.status == BlockStatus.FAILED
                        else unit.status
                    )
                item.notes.extend(unit.warnings)
            first = unit.members[0]
            if unit.status == BlockStatus.FAILED:
                self.issues.append(
                    Issue(
                        Severity.ERROR,
                        "translation_failed",
                        "This text could not be translated reliably and was left in Portuguese"
                        + (f" ({unit.warnings[-1]})" if unit.warnings else "")
                        + ". Manual translation required.",
                        page=first.page,
                        bbox=first.bbox,
                        element_id=first.id,
                        source_text=first.plain()[:500],
                    )
                )
                self.stats["failed"] += 1
            elif unit.warnings:
                category = (
                    "terminology"
                    if any("glossary" in w for w in unit.warnings)
                    else (
                        "untranslated"
                        if any("Portuguese" in w for w in unit.warnings)
                        else (
                            "numbers"
                            if any("numbers" in w for w in unit.warnings)
                            else "formatting"
                        )
                    )
                )
                self.issues.append(
                    Issue(
                        Severity.WARNING,
                        category,
                        "Check this translation: "
                        + "; ".join(dict.fromkeys(unit.warnings)),
                        page=first.page,
                        bbox=first.bbox,
                        element_id=first.id,
                        source_text=first.plain()[:500],
                        translated_text=mk.plain_text(
                            unit.result or "", first.protected, first.math_text
                        )[:500],
                    )
                )
            if unit.status == BlockStatus.TRANSLATED:
                self.stats["translated"] += 1
