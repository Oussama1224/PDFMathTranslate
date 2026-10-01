"""Prompt construction for LLM-based providers.

The system prompt is identical for every request of a document (style rules
plus the full document glossary) so it can be prompt-cached; per-batch content
goes in the user message.
"""

from __future__ import annotations

import json

from pt2en.config import EnglishVariant, TranslationStyle
from pt2en.translation.base import Segment, TranslationContext
from pt2en.translation.glossary import GlossaryEntry

STYLE_GUIDES = {
    TranslationStyle.ACADEMIC: (
        "Academic register: natural, formal English suitable for university course material. "
        "Prefer clear, idiomatic phrasing over word-for-word renderings, while keeping every "
        "piece of information, nuance and the instructional tone of the source."
    ),
    TranslationStyle.TECHNICAL: (
        "Technical register: precise, concise English using the standard terminology of the "
        "field (mathematics, statistics, engineering, computing, economics...). Prefer the term "
        "an English-language textbook would use; keep sentences direct."
    ),
    TranslationStyle.LITERAL: (
        "Literal/faithful register: stay as close as possible to the source structure and word "
        "choice while remaining grammatical English. Do not paraphrase, merge, split, reorder "
        "or embellish; translate idioms only when a literal rendering would be wrong."
    ),
}

VARIANT_NOTES = {
    EnglishVariant.UK: "Use British English spelling and conventions (colour, analyse, modelling, organisation).",
    EnglishVariant.US: "Use American English spelling and conventions (color, analyze, modeling, organization).",
}

KIND_NOTES = """Segment kinds:
- heading: translate as a heading; keep the source capitalisation pattern (sentence case stays sentence case).
- paragraph / list_item / footnote / caption: full sentences; keep figure/table numbering formats ("Figura 2.1 –" -> "Figure 2.1 –").
- table_cell, label, image_label: short labels inside tables, charts or diagrams; be concise and never longer than needed.
- header / footer / page_number: running headers, footers and page numbers ("Página 3 de 10" -> "Page 3 of 10").
"""


def build_system_prompt(ctx: TranslationContext) -> str:
    preserve = (
        "Keep acronyms (e.g. ECTS, IRS, IVA when used as names), official names of Portuguese "
        "laws, programmes and bodies without an established English name, and terms the "
        "glossary marks 'keep in Portuguese' in their original form."
        if ctx.preserve_terminology
        else "Translate every term, including official names, when an English equivalent exists; "
        "keep only proper names of people and established acronyms."
    )
    numbers = (
        "Localise number formatting to English conventions (decimal comma -> decimal point, "
        "thousand separators), without changing any value."
        if ctx.localize_numbers
        else "Keep every number exactly as written in the source, including decimal commas "
        "(15,5 stays 15,5), dates and units."
    )
    glossary = format_glossary(ctx.glossary)
    title = (
        f'The document is titled "{ctx.document_title}".' if ctx.document_title else ""
    )
    return f"""You are an expert translator of educational and scientific course material from European Portuguese (pt-PT) into English ({ctx.variant.value}). {title}

## Style
{STYLE_GUIDES[ctx.style]}
{VARIANT_NOTES[ctx.variant]}

## European Portuguese
The source is European Portuguese, possibly written before or after the 1990 orthographic agreement (acção/ação, óptimo/ótimo, facto). Interpret vocabulary with its European meaning, never the Brazilian one: e.g. "facto" = fact, "fato" = suit, "equipa" = team, "ecrã" = screen, "ficheiro" = file, "registo" = record, "perceber" = understand, "repare que" = note that, "frequência" (assessment) = mid-term test, "cadeira" (university) = course.

## Inline markup — mandatory rules
The text contains lightweight markup that must survive translation:
- <b>…</b>, <i>…</i>, <u>…</u>, <sup>…</sup>, <sub>…</sub> and <s1>…</s1>, <s2>…</s2>, … mark formatted words. Wrap the English words that correspond to the formatted Portuguese words with the same tag. Keep the same number of each tag; never invent new tags.
- <m1/>, <m2/>, … are mathematical formulas and <x1/>, <x2/>, … are protected literals (URLs, code). Each must appear exactly once, unchanged, at the grammatically correct position in the English sentence. Never translate, remove, duplicate or expand them.
- Characters &, < and > in the text are escaped as &amp;, &lt;, &gt;; keep them escaped.

## Content rules
- Translate accurately and completely: no omissions, additions, summaries or explanations.
- Formulas, mathematical symbols, variable names, units, chemical formulas, citations ([1], (Silva, 2020)), and code identifiers stay exactly as in the source.
- {numbers}
- Keep proper names of people unchanged. Translate names of institutions only when an established English name exists (Universidade de Lisboa -> University of Lisbon).
- {preserve}
- Use the same English term for the same Portuguese term everywhere. Follow the glossary: entries marked MUST are mandatory; entries marked prefer are guidance.
- If a segment is already in English or contains no translatable words, return it unchanged.
{KIND_NOTES}
## Glossary
{glossary or "(no glossary entries)"}

## Output
Return only JSON matching the schema: one object per input segment, with the same "id", in the same order, holding the English markup in "text"."""


def format_glossary(entries: list[GlossaryEntry]) -> str:
    rows = []
    for e in sorted(entries, key=lambda e: (not e.strict, e.pt.lower())):
        target = f"{e.pt} (keep in Portuguese)" if e.keep else e.en
        flag = "MUST" if e.strict else "prefer"
        note = f" — {e.note}" if e.note else ""
        rows.append(f"- {e.pt} => {target} [{flag}]{note}")
    return "\n".join(rows)


def build_batch_message(
    segments: list[Segment],
    ctx: TranslationContext,
    section: str,
    preceding: list[str] | None = None,
) -> str:
    items = []
    for s in segments:
        item = {"id": s.id, "kind": s.kind, "text": s.text}
        if s.note:
            item["note"] = s.note
        if s.max_ratio:
            item["max_length_chars"] = max(4, int(len(s.text) * s.max_ratio))
        items.append(item)
    parts = []
    if section:
        parts.append(f"Current section: {section}")
    if preceding:
        prev = "\n".join(f"- {src}" for src in preceding[-3:])
        parts.append(
            "Text immediately preceding these segments (context only, do not translate):\n"
            + prev
        )
    parts.append(
        "Translate these segments:\n" + json.dumps(items, ensure_ascii=False, indent=1)
    )
    return "\n\n".join(parts)


TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}

TERMINOLOGY_SCHEMA = {
    "type": "object",
    "properties": {
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pt": {"type": "string"},
                    "en": {"type": "string"},
                    "keep": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                "required": ["pt", "en", "keep", "note"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["terms"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["minor", "major", "critical"],
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "mistranslation",
                            "omission",
                            "addition",
                            "untranslated",
                            "terminology",
                            "numbers",
                            "grammar",
                        ],
                    },
                    "explanation": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["id", "severity", "category", "explanation", "suggestion"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}


def build_terminology_prompt(
    candidates: list[tuple[str, int, str]], ctx: TranslationContext
) -> str:
    lines = [
        f'- "{term}" (x{freq}) — context: "{example}"'
        for term, freq, example in candidates
    ]
    keep_rule = (
        "Set keep=true (and en equal to pt) for acronyms, official Portuguese names and terms "
        "without an English equivalent that should stay in Portuguese."
        if ctx.preserve_terminology
        else "Translate every term; set keep=true only for acronyms and proper names."
    )
    return f"""Below are candidate terms extracted from a European Portuguese course document{f' ("{ctx.document_title}")' if ctx.document_title else ''}.
Build the terminology glossary that a professional translator would use to translate this document into English ({ctx.variant.value}) consistently.

- Include only domain-specific, technical or recurring key terms whose translation must stay consistent; skip general vocabulary and fragments that are not real terms.
- Give the standard English term used in English-language textbooks of the field (European Portuguese meanings, never Brazilian).
- Use the base (singular, lower-case unless a proper noun) form of the Portuguese term.
- {keep_rule}
- Add a short note only when the choice needs justification (ambiguity, false friend).

Candidates:
{chr(10).join(lines)}"""


def build_review_prompt(pairs: list[tuple[str, str, str]]) -> str:
    items = [{"id": i, "source_pt": s, "translation_en": t} for i, s, t in pairs]
    return (
        "Review these translations from European Portuguese into English. Report only real "
        "problems: meaning changed (mistranslation), information missing (omission) or added "
        "(addition), Portuguese left untranslated, glossary/terminology violations, changed "
        "numbers or formulas, or clearly ungrammatical English. Do not report stylistic "
        "preferences. Placeholders such as <m1/> and tags such as <b> are formatting markup: "
        "they must be preserved, and are not errors.\n"
        "For major/critical findings, put a complete corrected English translation in "
        "'suggestion' (same markup tags and placeholders as the translation); otherwise use an "
        "empty string.\n\n" + json.dumps(items, ensure_ascii=False, indent=1)
    )


REVIEW_SYSTEM = (
    "You are a meticulous senior reviewer of Portuguese (pt-PT) to English translations of "
    "academic course material. You verify meaning, completeness, terminology consistency and "
    "the preservation of formulas, numbers and inline markup."
)
