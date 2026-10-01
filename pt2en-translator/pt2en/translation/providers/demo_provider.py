"""Offline demo translator.

A deterministic phrase/word dictionary translator that keeps the inline
markup intact. It exists so the whole pipeline (layout, formulas, images, QA)
can be exercised and tested without API keys. Its English is approximate and
it must not be used for real documents: unknown words are left untouched and
the QA stage reports them as untranslated Portuguese.
"""

from __future__ import annotations

import re

from pt2en.config import Settings
from pt2en.translation.base import Segment, TranslationContext, Translator
from pt2en.translation.glossary import normalize

PHRASES = {
    "estatística descritiva": "descriptive statistics",
    "unidade curricular": "course unit",
    "métodos quantitativos para a gestão": "quantitative methods for management",
    "métodos quantitativos": "quantitative methods",
    "faculdade de ciências": "Faculty of Sciences",
    "universidade de lisboa": "University of Lisbon",
    "ano letivo": "academic year",
    "tem como objetivo": "aims to",
    "de modo a": "in order to",
    "bem como": "as well as",
    "medidas de tendência central": "measures of central tendency",
    "tendência central": "central tendency",
    "média amostral": "sample mean",
    "variância amostral": "sample variance",
    "é dada por": "is given by",
    "calcula-se através da seguinte expressão": "is computed with the following expression",
    "a dividir pelo": "divided by the",
    "depois de ordenada": "once sorted",
    "com maior frequência": "most frequently",
    "teste intercalar": "mid-term test",
    "primeiro semestre": "first semester",
    "representação gráfica": "graphical representation",
    "gráfico de barras": "bar chart",
    "taxa de aprovação": "pass rate",
    "estudo estatístico": "statistical study",
    "recolha de dados": "data collection",
    "valores atípicos": "outliers",
    "inferência estatística": "statistical inference",
    "em síntese": "in summary",
    "primeiro passo": "first step",
    "página": "page",
    "número de alunos": "number of students",
    "média mensal": "monthly mean",
    "exercícios propostos": "proposed exercises",
    "bom trabalho": "good work",
    "por que razão": "why",
    "na presença de": "in the presence of",
    "do que": "than",
    "repare que": "note that",
}

WORDS = {
    "a": "the",
    "o": "the",
    "as": "the",
    "os": "the",
    "um": "a",
    "uma": "a",
    "de": "of",
    "do": "of the",
    "da": "of the",
    "dos": "of the",
    "das": "of the",
    "em": "in",
    "no": "in the",
    "na": "in the",
    "nos": "in the",
    "nas": "in the",
    "e": "and",
    "ou": "or",
    "com": "with",
    "para": "for",
    "por": "by",
    "pelo": "by the",
    "pela": "by the",
    "pelos": "by the",
    "ao": "to the",
    "à": "to the",
    "que": "that",
    "se": "if",
    "não": "not",
    "é": "is",
    "são": "are",
    "sua": "its",
    "seu": "its",
    "neste": "in this",
    "nesta": "in this",
    "este": "this",
    "esta": "this",
    "cada": "each",
    "qualquer": "any",
    "todas": "all",
    "todos": "all",
    "seguinte": "following",
    "seguintes": "following",
    "maior": "highest",
    "mais": "more",
    "onde": "where",
    "capítulo": "chapter",
    "introdução": "introduction",
    "estatística": "statistics",
    "resumir": "summarise",
    "organizar": "organise",
    "conjunto": "set",
    "dados": "data",
    "facilitar": "facilitate",
    "interpretação": "interpretation",
    "estudante": "student",
    "irá": "will",
    "aprender": "learn",
    "calcular": "calculate",
    "medidas": "measures",
    "dispersão": "dispersion",
    "construir": "build",
    "gráficos": "charts",
    "gráfico": "chart",
    "adequados": "suitable",
    "tipo": "type",
    "variável": "variable",
    "variáveis": "variables",
    "observações": "observations",
    "observação": "observation",
    "índice": "index",
    "percorre": "runs over",
    "amostra": "sample",
    "recolhida": "collected",
    "docente": "lecturer",
    "variância": "variance",
    "expressão": "expression",
    "média": "mean",
    "mediana": "median",
    "moda": "mode",
    "soma": "sum",
    "valores": "values",
    "valor": "value",
    "número": "number",
    "central": "central",
    "ocorre": "occurs",
    "tabela": "table",
    "classificações": "grades",
    "obtidas": "obtained",
    "alunos": "students",
    "aluno": "student",
    "nota": "grade",
    "situação": "status",
    "aprovado": "passed",
    "reprovado": "failed",
    "aprovados": "passed",
    "reprovados": "failed",
    "utilizados": "used",
    "foram": "were",
    "recolhidos": "collected",
    "compara": "compares",
    "turma": "class",
    "turmas": "classes",
    "apresenta": "shows",
    "figura": "figure",
    "distribuição": "distribution",
    "notas": "grades",
    "etapas": "stages",
    "organização": "organisation",
    "análise": "analysis",
    "conclusões": "conclusions",
    "evolução": "evolution",
    "mês": "month",
    "set.": "Sep.",
    "out.": "Oct.",
    "nov.": "Nov.",
    "dez.": "Dec.",
    "jan.": "Jan.",
    "descritiva": "descriptive",
    "constitui": "is",
    "estudo": "study",
    "permitindo": "allowing",
    "detetar": "to detect",
    "formular": "to formulate",
    "hipóteses": "hypotheses",
    "empresa": "company",
    "registou": "recorded",
    "reclamações": "complaints",
    "recebidas": "received",
    "durante": "during",
    "doze": "twelve",
    "semanas": "weeks",
    "consecutivas": "consecutive",
    "calcule": "calculate",
    "comente": "comment on",
    "resultados": "results",
    "construa": "build",
    "histograma": "histogram",
    "exercício": "exercise",
    "anterior": "previous",
    "indique": "indicate",
    "simétrica": "symmetric",
    "assimétrica": "asymmetric",
    "explique": "explain",
    "robusta": "robust",
    "extremos": "extreme",
    "dividir": "dividing",
    "unidade": "unit",
    "curricular": "course",
    "gestão": "management",
    "faculdade": "faculty",
    "ciências": "sciences",
    "universidade": "university",
    "letivo": "academic",
    "ano": "year",
    "trabalho": "work",
    "bom": "good",
    "medida": "measure",
    "obtidos": "obtained",
    "através": "through",
    "dada": "given",
    "presença": "presence",
    "razão": "reason",
}


class DemoTranslator(Translator):
    name = "demo"
    supports_markup = True
    is_llm = False

    def __init__(self, settings: Settings | None = None):
        self._phrase_items = sorted(PHRASES.items(), key=lambda kv: -len(kv[0]))

    def describe(self) -> str:
        return "Offline demo dictionary (not for production)"

    def translate_batch(
        self, segments: list[Segment], ctx: TranslationContext
    ) -> dict[str, str]:
        glossary = sorted(
            (
                (normalize(e.pt), e.pt if e.keep else e.en.split("/")[0].strip())
                for e in ctx.glossary
                if e.strict
            ),
            key=lambda kv: -len(kv[0]),
        )
        return {s.id: self._translate_markup(s.text, glossary) for s in segments}

    def _translate_markup(self, markup: str, glossary) -> str:
        parts = re.split(r"(<[^>]+>)", markup)
        return "".join(
            p if p.startswith("<") else self._translate_text(p, glossary) for p in parts
        )

    def _translate_text(self, text: str, glossary) -> str:
        m = re.match(r"^\s*P[áa]gina\s+(\d+)\s+de\s+(\d+)\s*$", text, re.I)
        if m:
            return f"Page {m.group(1)} of {m.group(2)}"
        out = text
        for pt, en in glossary + [(normalize(k), v) for k, v in self._phrase_items]:
            out = _replace_phrase(out, pt, en)
        tokens = re.split(r"(\s+|[,;:()!?\"“”«»]|(?<=\w)\.(?=\s|$))", out)
        translated = []
        for tok in tokens:
            if not tok or tok.isspace() or not re.search(r"[A-Za-zÀ-ÿ]", tok):
                translated.append(tok)
                continue
            key = tok.lower()
            if len(tok) == 1 and tok.isupper() and any(t.strip() for t in translated):
                translated.append(tok)  # labels such as "Turma A"
                continue
            en = WORDS.get(key) or WORDS.get(key.rstrip("."))
            if en is None:
                translated.append(tok)
                continue
            if tok[:1].isupper():
                en = en[:1].upper() + en[1:]
            translated.append(en)
        return "".join(translated)


def _replace_phrase(text: str, pt_norm: str, en: str) -> str:
    """Replace a phrase, matching case/accent-insensitively."""
    if not pt_norm:
        return text
    norm = normalize(text)
    if len(norm) != len(text):  # normalisation changed lengths; fall back to exact
        return re.sub(re.escape(pt_norm), en, text, flags=re.I)
    out = []
    pos = 0
    for m in re.finditer(r"(?<!\w)" + re.escape(pt_norm) + r"(?!\w)", norm):
        out.append(text[pos : m.start()])
        original = text[m.start() : m.end()]
        out.append(en[:1].upper() + en[1:] if original[:1].isupper() else en)
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)
