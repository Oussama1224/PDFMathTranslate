from pt2en.translation.glossary import (
    Glossary,
    GlossaryEntry,
    extract_candidates,
    parse_user_glossary,
)
from pt2en.translation.language import (
    detect,
    is_probably_english,
    looks_untranslated,
    portuguese_residue,
)


def test_language_detection():
    assert (
        detect("A estatística descritiva tem como objetivo resumir os dados.") == "pt"
    )
    assert detect("Descriptive statistics aims to summarise the data.") == "en"
    assert is_probably_english(
        "The following essay owes its origin to a conversation with a friend."
    )
    assert not is_probably_english(
        "O ensaio seguinte deve a sua origem a uma conversa com um amigo."
    )


def test_residual_portuguese():
    assert looks_untranslated("The mean amostral of the sample")
    assert portuguese_residue("value with highest frequência") == ["frequência"]
    assert not looks_untranslated("Measures of central tendency")
    assert not looks_untranslated("The capital base of the media")
    assert looks_untranslated("Turma A")


def test_parse_user_glossary_formats():
    entries = parse_user_glossary(
        "unidade curricular = course unit\nfrequência => mid-term test\nECTS = [keep]\n# comment"
    )
    assert [(e.pt, e.en, e.keep) for e in entries] == [
        ("unidade curricular", "course unit", False),
        ("frequência", "mid-term test", False),
        ("ECTS", "ECTS", True),
    ]
    assert all(e.strict and e.source == "user" for e in entries)
    csv = parse_user_glossary("pt,en\nficheiro,file\necrã,screen")
    assert [(e.pt, e.en) for e in csv] == [("ficheiro", "file"), ("ecrã", "screen")]
    js = parse_user_glossary('{"rato": "mouse"}')
    assert js[0].en == "mouse"


def test_glossary_matching_plurals_and_accents():
    g = Glossary()
    g.add(GlossaryEntry("variável", "variable", strict=True))
    g.add(GlossaryEntry("função", "function", strict=True))
    assert {e.pt for e in g.match("As variáveis e as funções")} == {
        "variável",
        "função",
    }
    assert g.check_translation("A função f", "The function f") == []
    missing = g.check_translation("A função f", "The map f")
    assert [e.pt for e in missing] == ["função"]


def test_user_overrides_win_over_base():
    g = Glossary.with_base()
    assert g.entries["ficheiro"].en == "file"
    g.extend(parse_user_glossary("ficheiro = document"))
    assert g.entries["ficheiro"].en == "document"
    assert g.entries["ficheiro"].strict


def test_term_candidates():
    texts = [
        ("A média amostral é um estimador. A média amostral converge.", 1.0),
        ("Média amostral", 3.0),
    ]
    terms = [t for t, _, _ in extract_candidates(texts)]
    assert "média amostral" in terms
