"""Lightweight Portuguese/English language identification.

Used to skip text that is already English, to verify translator output and
for the QA second pass that hunts for untranslated Portuguese. It is tuned for
short, technical course text where general-purpose detectors are unreliable.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")

PT_STOPWORDS = frozenset("""
    de da do das dos em na no nas nos um uma uns umas o os ao aos à às por para com sem
    é são está estão ser estar foi foram era eram será serão seria seriam sido sendo
    tem têm tinha tinham ter terá teria haver há havia houve que se não nem ou mas também
    já ainda apenas só sempre nunca muito muita muitos muitas pouco pouca poucos poucas
    mais menos maior menor melhor pior bem mal assim então porque porquê pois portanto
    contudo todavia entretanto embora enquanto quanto tanto tão quando onde como qual
    quais quem cujo cuja cujos cujas este esta estes estas esse essa esses essas aquele
    aquela aqueles aquelas isto isso aquilo deste desta destes destas desse dessa desses
    dessas daquele daquela neste nesta nestes nestas nesse nessa naquele naquela pelo pela
    pelos pelas num numa numas nuns dum duma seu sua seus suas nosso nossa nossos nossas
    dele dela deles delas lhe lhes você vocês eles elas ela ele nós vós tu te ti eu meu
    minha meus minhas teu tua teus tuas cada outro outra outros outras todo toda todos
    todas qualquer quaisquer mesmo mesma mesmos mesmas através sobre entre até desde após
    perante segundo durante dentro acima abaixo junto ali aqui lá cá hoje ontem amanhã
    agora depois deve devem pode podem podemos devemos sejam seja fosse fossem tenha
    tenham estiver estiverem houver vez vezes seguinte seguintes exemplo exemplos ainda
    caso conforme além aliás inclusive nenhum nenhuma algum alguma alguns algumas
    tal tais vários várias diversos diversas próprio própria próprios próprias
    """.split())

EN_STOPWORDS = frozenset("""
    the of and to in is that for it was with be by on not he this are or his from at
    which but have an they you were their one all we can her has there been if more when
    will would who she other its may these into what should each than then them some
    also between such must only used using where how our any both through while about
    after before however therefore because could does did here those very well within
    without given shown following example values value its it's we're is are were
    """.split())

# Frequent Portuguese words of course material that carry no diacritics and are
# not English words (so stop-words and accents alone would miss them).
PT_COMMON = frozenset("""
    amostral amostra amostras dados aluno alunos aluna alunas turma turmas valor valores
    nota notas curso cursos aula aulas exemplo exemplos exercicio exercicios resposta
    respostas pergunta perguntas tabela tabelas figura figuras quadro quadros capitulo
    capitulos objetivo objetivos objectivo objectivos conjunto conjuntos modelo modelos
    processo processos resultado resultados metodo metodos trabalho trabalhos estudo estudos
    empresa empresas custo custos preco precos taxa taxas lucro lucros receita receitas
    mercado mercados procura oferta quantidade quantidades tempo medida medidas forma formas
    parte partes ponto pontos linha linhas coluna colunas numero numeros soma produto
    produtos dado nome nomes ano anos dias semana semanas horas primeiro primeira segundo
    segunda terceiro terceira ultimo ultima igual iguais diferente diferentes novo novos
    novas grande grandes pequeno pequena pequenos pequenas fazer feito feita existe existem
    obter obtido obtida calcular determinar considerar considere sabendo indique justifique
    resolva calcule explique apresente descreva compare analise defina verifique mostre
    prove demonstre escreva represente construa identifique classifique complete responda
    seguinte anterior anteriores respetivo respetiva respectivo respectiva correspondente
    correspondentes relativamente sobretudo nomeadamente consiste permite permitem utiliza
    utilizar utilizado utilizada utilizados utilizadas trata tratar sendo podemos devemos
    temos fazemos vamos sabemos dizer disse diz dito tipo tipos caso casos meio meios fim
    fins lado lados grupo grupos campo campos nivel niveis sistema sistemas problema
    problemas solucao exame exames teste testes avaliacao aprovado reprovado aprovados
    reprovados docente docentes estudante estudantes professor professores semestre
    licenciatura mestrado escola faculdade universidade disciplina disciplinas cadeira
    cadeiras matriz matrizes vetor vetores funcao funcoes derivada derivadas integrais
    limite limites reta retas plano planos ponto curva curvas grafico graficos mediana moda
    variancia desvio frequencia probabilidade amostragem populacao hipotese hipoteses
    estimador estimadores intervalo intervalos regressao correlacao variavel variaveis
    teorema teoremas lema definicao propriedade propriedades exercicio sejam seja tais tal
    outro outra outros outras todo toda todos todas cada qualquer entao portanto assim ainda
    apenas quando onde como porque pois mas tambem
    """.split())

# Morphology that is characteristic of Portuguese and rare in English.
_PT_SUFFIX_RE = re.compile(
    r"(ção|ções|são|sões|ãos|ães|ões|dade|dades|mente|ência|ências|ância|âncias|ório|ória"
    r"|ível|áveis|íveis|ável|agem|agens|ismo|ismos|eiro|eira|eiros|eiras|inho|inha|zinho"
    r"|ando|endo|indo|aram|eram|iram|ava|avam|ido|idos|ida|idas|ado|ados|ada|adas)$"
)
_PT_STRONG_CHARS = re.compile(r"[ãõçêôâ]")
_PT_WEAK_CHARS = re.compile(r"[áéíóúà]")
_EN_SUFFIX_RE = re.compile(
    r"(tion|tions|ing|ness|ship|ally|ight|ould|ough|th|wh)$|^(th|wh)"
)

# Portuguese/English homographs: never evidence of untranslated text.
HOMOGRAPHS = frozenset("""
    base capital total final normal real central global local social ideal animal hospital
    material natural general original plural regional manual crime data media radio video
    piano solo cobra panorama drama dilema area era fora antes logo nova dia hora pilar
    tensor vector factor sector motor error horror terror doctor actor
    """.split())

# Words that look Portuguese by morphology but are ordinary English words.
_EN_EXCEPTIONS = frozenset("""
    data media made trade grade code mode node side wide guide inside outside decade
    upgrade update needed used based stated rated dated added ended tended aided
    leading reading heading ending bending sending lending reliance agenda manda
    commando panda veranda armada parade cascade ball sole mental fundamental element
    elements comment moment segment fragment cement statement department government
    management development environment experiment assessment requirement agreement
    eminent prominent document documents instrument instruments argument arguments
    """.split())


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def words_of(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def word_strength(word: str, lexicon: frozenset[str] = frozenset()) -> int:
    """How strongly a single word indicates Portuguese: 0 (no), 1 (weak), 2 (strong)."""
    w = word.lower()
    if len(w) < 2:
        return 0
    if w in EN_STOPWORDS or w in _EN_EXCEPTIONS or w in HOMOGRAPHS:
        return 0
    bare = _strip_accents(w)
    if (w in PT_STOPWORDS and len(w) >= 3) or bare in PT_COMMON or bare in lexicon:
        return 2
    if _PT_STRONG_CHARS.search(w):
        return 2
    if _PT_WEAK_CHARS.search(w) and len(w) >= 3:
        return 2
    if len(w) >= 6 and _PT_SUFFIX_RE.search(w) and not _EN_SUFFIX_RE.search(w):
        return 1
    return 0


def word_is_portuguese(word: str, lexicon: frozenset[str] = frozenset()) -> bool:
    return word_strength(word, lexicon) > 0


@dataclass
class LanguageScore:
    pt: float
    en: float
    words: int

    @property
    def label(self) -> str:
        if self.words == 0:
            return "unknown"
        if self.pt >= 1.0 and self.pt > self.en * 1.2:
            return "pt"
        if self.en >= 1.0 and self.en > self.pt * 1.5:
            return "en"
        return "unknown"


def score(text: str) -> LanguageScore:
    words = words_of(text)
    pt = en = 0.0
    for raw in words:
        w = raw.lower()
        if w in PT_STOPWORDS and w not in EN_STOPWORDS:
            pt += 1.0 if len(w) > 1 else 0.4
        if w in EN_STOPWORDS and w not in PT_STOPWORDS:
            en += 1.0
        if word_is_portuguese(raw) and w not in PT_STOPWORDS:
            pt += 1.2 if _PT_STRONG_CHARS.search(w) else 0.8
        elif _PT_WEAK_CHARS.search(w) and not raw[:1].isupper():
            pt += 0.4
        if _EN_SUFFIX_RE.search(w) and len(w) > 4:
            en += 0.6
    return LanguageScore(pt=pt, en=en, words=len(words))


def detect(text: str) -> str:
    return score(text).label


def is_probably_english(text: str) -> bool:
    s = score(text)
    if s.words < 3:
        return False
    if s.words >= 12:
        return s.en >= 3.0 and s.en >= 2.5 * s.pt
    return s.label == "en" and s.pt < 1.0


def _residue(
    text: str, allowed: frozenset[str], lexicon: frozenset[str]
) -> list[tuple[str, int]]:
    found = []
    words = words_of(text)
    for i, raw in enumerate(words):
        w = raw.lower()
        if w in allowed or _strip_accents(w) in allowed:
            continue
        strength = word_strength(raw, lexicon)
        if not strength:
            continue
        # Capitalised words inside a sentence are often names (José, São Paulo).
        if (
            raw[:1].isupper()
            and i > 0
            and w not in PT_STOPWORDS
            and _strip_accents(w) not in PT_COMMON
        ):
            continue
        found.append((raw, strength))
    return found


def portuguese_residue(
    text: str,
    allowed: frozenset[str] = frozenset(),
    lexicon: frozenset[str] = frozenset(),
) -> list[str]:
    """Return the Portuguese-looking words left in a supposedly English text.

    ``allowed`` holds lower-cased words that are legitimately kept (glossary
    terms preserved on purpose, proper names...); ``lexicon`` holds extra
    (accent-stripped) Portuguese words, e.g. from the document glossary.
    """
    return [w for w, _ in _residue(text, allowed, lexicon)]


def looks_untranslated(
    text: str,
    allowed: frozenset[str] = frozenset(),
    lexicon: frozenset[str] = frozenset(),
) -> bool:
    residue = _residue(text, allowed, lexicon)
    if not residue:
        return False
    strong = sum(1 for _, s in residue if s >= 2)
    weak = len(residue) - strong
    return strong >= 1 or weak >= 2
