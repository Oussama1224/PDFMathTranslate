"""Terminology layer: keeps every Portuguese term translated the same way.

Three sources are merged (later ones override earlier ones):

1. the curated base glossary (advisory guidance),
2. a document glossary extracted from the PDF itself (strict),
3. user overrides from the UI / CLI (strict, highest priority).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from pt2en.translation.base_glossary import BASE_GLOSSARY
from pt2en.translation.language import PT_STOPWORDS

KEEP_MARKERS = {"=", "[keep]", "(keep)", "keep", "manter", "[manter]"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


def _plural_variants(term: str) -> set[str]:
    """Portuguese plural forms of the last word of a term (normalised)."""
    words = term.split(" ")
    last = words[-1]
    variants = {last}
    if last.endswith("ao"):
        variants |= {last[:-2] + "oes", last[:-2] + "aes", last + "s"}
    elif last.endswith("l"):
        variants |= {last[:-1] + "is", last[:-1] + "es"}
    elif last.endswith("m"):
        variants.add(last[:-1] + "ns")
    elif last.endswith(("r", "z", "s")):
        variants.add(last + "es")
    else:
        variants.add(last + "s")
    return {" ".join(words[:-1] + [v]) for v in variants}


@dataclass
class GlossaryEntry:
    pt: str
    en: str
    note: str = ""
    source: str = "base"  # base | document | user
    strict: bool = False
    keep: bool = False  # keep the Portuguese term (preserve terminology)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Glossary:
    entries: dict[str, GlossaryEntry] = field(default_factory=dict)
    _patterns: dict[str, re.Pattern] = field(default_factory=dict, repr=False)

    # --------------------------------------------------------------- build
    @classmethod
    def with_base(cls) -> "Glossary":
        g = cls()
        for pt, en, note in BASE_GLOSSARY:
            g.add(GlossaryEntry(pt=pt, en=en, note=note, source="base", strict=False))
        return g

    def add(self, entry: GlossaryEntry, override: bool = True) -> None:
        key = normalize(entry.pt)
        if not key:
            return
        if key in self.entries and not override:
            return
        self.entries[key] = entry
        self._patterns.pop(key, None)

    def extend(self, entries: Iterable[GlossaryEntry], override: bool = True) -> None:
        for e in entries:
            self.add(e, override=override)

    # -------------------------------------------------------------- lookup
    def _pattern(self, key: str) -> re.Pattern:
        pat = self._patterns.get(key)
        if pat is None:
            alts = sorted(
                (re.escape(v) for v in _plural_variants(key)), key=len, reverse=True
            )
            pat = re.compile(r"(?<![\w])(" + "|".join(alts) + r")(?![\w])")
            self._patterns[key] = pat
        return pat

    def match(self, text: str) -> list[GlossaryEntry]:
        norm = normalize(text)
        found = []
        for key, entry in self.entries.items():
            if key.split(" ")[0][:3] not in norm:
                continue
            if self._pattern(key).search(norm):
                found.append(entry)
        # Prefer longer terms first (more specific).
        found.sort(key=lambda e: -len(e.pt))
        return found

    def strict_entries(self) -> list[GlossaryEntry]:
        return [e for e in self.entries.values() if e.strict]

    def portuguese_lexicon(self) -> frozenset[str]:
        """Accent-stripped Portuguese words of glossary terms (minus kept ones)."""
        from pt2en.translation.language import EN_STOPWORDS, HOMOGRAPHS

        english = set()
        for e in self.entries.values():
            english.update(re.findall(r"[a-z]+", normalize(e.en)))
        words = set()
        for key, e in self.entries.items():
            if e.keep:
                continue
            for w in key.split():
                if (
                    len(w) >= 4
                    and w not in english
                    and w not in HOMOGRAPHS
                    and w not in EN_STOPWORDS
                ):
                    words.add(w)
        return frozenset(words)

    def kept_terms(self) -> frozenset[str]:
        words = set()
        for e in self.entries.values():
            if e.keep or normalize(e.en) == normalize(e.pt):
                for w in e.pt.split():
                    words.add(w.lower())
                    words.add(normalize(w))
        return frozenset(words)

    def check_translation(self, source: str, target: str) -> list[GlossaryEntry]:
        """Strict entries present in ``source`` but whose English is missing."""
        missing = []
        tnorm = normalize(target)
        for entry in self.match(source):
            if not entry.strict:
                continue
            expected = entry.pt if entry.keep else entry.en
            options = [
                normalize(o) for o in re.split(r"\s*/\s*", expected) if o.strip()
            ]
            if not any(_contains_term(tnorm, o) for o in options):
                missing.append(entry)
        return missing

    # ----------------------------------------------------------- serialise
    def fingerprint(self) -> str:
        data = sorted(
            (k, e.en, e.keep, e.strict)
            for k, e in self.entries.items()
            if e.source != "base"
        )
        return hashlib.sha256(
            json.dumps(data, ensure_ascii=False).encode()
        ).hexdigest()[:16]

    def to_list(self, include_base: bool = False) -> list[dict]:
        return [
            e.to_dict()
            for e in sorted(self.entries.values(), key=lambda e: e.pt.lower())
            if include_base or e.source != "base"
        ]

    def prompt_table(self, entries: Optional[Iterable[GlossaryEntry]] = None) -> str:
        rows = []
        for e in entries if entries is not None else self.entries.values():
            target = f"{e.pt} (keep in Portuguese)" if e.keep else e.en
            flag = "MUST" if e.strict else "prefer"
            note = f" — {e.note}" if e.note else ""
            rows.append(f"- {e.pt} => {target} [{flag}]{note}")
        return "\n".join(rows)


def _contains_term(haystack_norm: str, term_norm: str) -> bool:
    if not term_norm:
        return True
    words = term_norm.split()
    # Allow simple English inflection on the last word (plural, -ed, -ing).
    last = re.escape(words[-1])
    head = r"\s+".join(re.escape(w) for w in words[:-1])
    pattern = (head + r"\s+" if head else "") + last + r"(?:s|es|ed|d|ing)?"
    return re.search(r"(?<![\w])" + pattern + r"(?![\w])", haystack_norm) is not None


# ------------------------------------------------------------------ parsing
def parse_user_glossary(text: str) -> list[GlossaryEntry]:
    """Parse user overrides.

    Accepted formats (one per line): ``termo = term``, ``termo => term``,
    ``termo<TAB>term``, ``termo;term``, CSV with a header, or a JSON list/dict.
    Use ``termo = =`` or ``termo = [keep]`` to keep the Portuguese term.
    """
    text = (text or "").strip()
    if not text:
        return []
    if text[:1] in "[{":
        try:
            data = json.loads(text)
            items = (
                data.items()
                if isinstance(data, dict)
                else (
                    (d.get("pt") or d.get("source"), d.get("en") or d.get("target"))
                    for d in data
                )
            )
            return [_entry(pt, en) for pt, en in items if pt and en is not None]
        except (json.JSONDecodeError, AttributeError):
            pass
    entries = []
    lines = text.splitlines()
    if (
        len(lines) > 1
        and "," in lines[0]
        and not any(sep in lines[0] for sep in ("=", "\t", ";"))
    ):
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if rows and {c.strip().lower() for c in rows[0]} & {
            "pt",
            "portuguese",
            "source",
            "termo",
        }:
            rows = rows[1:]
        for row in rows:
            if len(row) >= 2 and row[0].strip():
                entries.append(_entry(row[0], row[1]))
        return entries
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ("=>", "->", "\t", "=", ";", "|", ","):
            if sep in line:
                pt, en = line.split(sep, 1)
                if pt.strip():
                    entries.append(_entry(pt, en))
                break
    return entries


def _entry(pt: str, en: str) -> GlossaryEntry:
    pt = pt.strip().strip('"')
    en = (en or "").strip().strip('"')
    keep = en.lower() in KEEP_MARKERS or en == "" or normalize(en) == normalize(pt)
    return GlossaryEntry(
        pt=pt, en=pt if keep else en, source="user", strict=True, keep=keep
    )


# ------------------------------------------------------- term candidates
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\-]+")
_CONNECTORS = {"de", "da", "do", "das", "dos", "e"}


def extract_candidates(
    texts: Iterable[tuple[str, float]], limit: int = 120, min_freq: int = 2
) -> list[tuple[str, int, str]]:
    """Find candidate domain terms: (term, frequency, example context).

    ``texts`` yields (text, weight) pairs; headings and bold text should carry
    a higher weight so their terms are favoured.
    """
    counts: Counter = Counter()
    weights: Counter = Counter()
    example: dict[str, str] = {}
    for text, weight in texts:
        tokens = _TOKEN_RE.findall(text)
        lowered = [t.lower() for t in tokens]
        n = len(tokens)
        for size in (1, 2, 3, 4):
            for i in range(n - size + 1):
                gram = lowered[i : i + size]
                if gram[0] in PT_STOPWORDS or gram[-1] in PT_STOPWORDS:
                    continue
                if size == 1 and (len(gram[0]) < 5 or gram[0] in PT_STOPWORDS):
                    continue
                inner = gram[1:-1]
                if any(w in PT_STOPWORDS and w not in _CONNECTORS for w in inner):
                    continue
                term = " ".join(tokens[i : i + size])
                key = term.lower()
                counts[key] += 1
                weights[key] += weight
                if key not in example:
                    start = max(0, text.lower().find(key) - 60)
                    example[key] = text[start : start + 160].strip()
    scored = []
    for key, freq in counts.items():
        if freq < min_freq and weights[key] < 2.0:
            continue
        words = key.split()
        score = weights[key] * (1.0 + 0.6 * (len(words) - 1))
        scored.append((score, key, freq))
    scored.sort(reverse=True)
    out: list[tuple[str, int, str]] = []
    chosen: set[str] = set()
    for _, key, freq in scored:
        # Skip sub-terms fully covered by an already chosen longer term with equal frequency.
        if any(key in c and counts[c] >= freq for c in chosen if c != key):
            continue
        chosen.add(key)
        out.append((key, freq, example.get(key, "")))
        if len(out) >= limit:
            break
    return out
