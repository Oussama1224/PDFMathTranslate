import re

from pt2en.model import BlockStatus
from pt2en.pipeline.options import JobOptions
from pt2en.translation.base import TranslationRefused, Translator
from pt2en.translation.engine import TranslationEngine, TranslationItem
from pt2en.translation.memory import TranslationMemory


class ScriptedTranslator(Translator):
    """Fake provider: returns scripted answers per attempt."""

    name = "scripted"
    is_llm = True

    def __init__(self, answers):
        self.answers = answers  # source text -> list of answers (one per attempt)
        self.calls = []
        self.notes = []

    def translate_batch(self, segments, ctx):
        self.calls.append([s.text for s in segments])
        self.notes.extend(s.note for s in segments if s.note)
        out = {}
        for s in segments:
            seq = self.answers.get(s.text)
            if seq is None:
                continue
            if isinstance(seq, Exception):
                raise seq
            idx = min(
                sum(1 for c in self.calls for t in c if t == s.text) - 1, len(seq) - 1
            )
            out[s.id] = seq[idx]
        return out


def item(i, text, kind="paragraph", math=None):
    return TranslationItem(
        id=f"b{i}",
        markup=text,
        kind=kind,
        page=0,
        bbox=(0, 0, 10, 10),
        math_text=math or {},
    )


def engine(settings, translator, glossary=""):
    return TranslationEngine(
        translator, settings, JobOptions.from_settings(settings, glossary_text=glossary)
    )


def test_retries_when_formula_placeholder_is_lost(settings):
    src = "A média <m1/> é importante."
    t = ScriptedTranslator(
        {src: ["The mean is important.", "The mean <m1/> is important."]}
    )
    e = engine(settings, t)
    it = item(0, src, math={"m1": "x"})
    e.translate_items([it])
    assert it.status == BlockStatus.TRANSLATED
    assert it.translation == "The mean <m1/> is important."
    assert len(t.calls) == 2
    assert any("missing placeholders" in n for n in t.notes)


def test_keeps_original_when_placeholders_never_come_back(settings):
    src = "Valor <m1/> final."
    t = ScriptedTranslator({src: ["Final value."]})
    e = engine(settings, t)
    it = item(0, src, math={"m1": "x"})
    e.translate_items([it])
    assert it.status == BlockStatus.FAILED
    assert it.translation == src  # original kept, never corrupted
    assert any(i.category == "translation_failed" for i in e.issues)


def test_refusal_is_reported(settings):
    src = "Texto qualquer em português."
    e = engine(settings, ScriptedTranslator({src: TranslationRefused("declined")}))
    it = item(0, src)
    e.translate_items([it])
    assert it.status == BlockStatus.FAILED


def test_duplicates_are_translated_once_and_numbers_checked(settings):
    src = "Página de exemplo com 3 alunos"
    t = ScriptedTranslator(
        {src: ["Example page with 4 students", "Example page with 3 students"]}
    )
    e = engine(settings, t)
    items = [item(i, src, kind="paragraph") for i in range(3)]
    e.translate_items(items)
    assert all(i.translation == "Example page with 3 students" for i in items)
    assert sum(len(c) for c in t.calls) == 2  # one unit, one retry
    assert e.stats["deduplicated"] == 2


def test_rule_based_and_english_segments_skip_the_provider(settings):
    t = ScriptedTranslator({})
    e = engine(settings, t)
    items = [
        item(0, "Página 2 de 10", kind="page_number"),
        item(1, "12,5"),
        item(2, "This sentence is already written in English."),
    ]
    e.translate_items(items)
    assert items[0].translation == "Page 2 of 10"
    assert items[1].status == BlockStatus.KEPT
    assert items[2].status == BlockStatus.KEPT
    assert t.calls == []


def test_glossary_violation_triggers_feedback(settings):
    src = "O ficheiro contém a frequência."
    t = ScriptedTranslator(
        {
            src: [
                "The file contains the frequency.",
                "The file contains the mid-term test.",
            ]
        }
    )
    e = engine(settings, t, glossary="frequência = mid-term test")
    it = item(0, src)
    e.translate_items([it])
    assert it.translation == "The file contains the mid-term test."
    assert any("frequência => mid-term test" in n for n in t.notes)


def test_translation_memory_is_reused(settings, tmp_path):
    memory = TranslationMemory(tmp_path / "tm.sqlite3")
    src = "Conjunto de dados."
    t1 = ScriptedTranslator({src: ["Data set."]})
    e1 = TranslationEngine(
        t1, settings, JobOptions.from_settings(settings), memory=memory
    )
    e1.translate_items([item(0, src)])
    t2 = ScriptedTranslator({})
    t2.describe = t1.describe
    e2 = TranslationEngine(
        t2, settings, JobOptions.from_settings(settings), memory=memory
    )
    it = item(0, src)
    e2.translate_items([it])
    assert it.translation == "Data set." and t2.calls == []


def test_residual_portuguese_is_retried(settings):
    src = "A variância amostral."
    t = ScriptedTranslator({src: ["The variância amostral.", "The sample variance."]})
    e = engine(settings, t)
    it = item(0, src)
    e.translate_items([it])
    assert it.translation == "The sample variance."
    assert any(
        re.search("Portuguese words left untranslated: .*variância", n) for n in t.notes
    )
