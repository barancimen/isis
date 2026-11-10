"""The optional Rust extension gives the same answers as the Python code it stands in for.
Skipped when it isn't built (pip install ./native)."""

import pytest

pytest.importorskip("isis_native")

from isis.engine import native
from isis.engine.search import phrase_pattern
from isis.engine.text import term_spans

LINES = [
    "The Whale, the WHALE and a whale's tail; whales are no match.",
    "Café naïve, café with a combining accent, and snake_case word_2 42nd.",
    "鯨は海にいる 🐋 whale 🐋 Whale after two emoji",
    "İstanbul ISTANBUL ıstanbul, ΟΔΟΣ οδος, Straße STRASSE",
    "Ⅷ and ² and ٣ are numbers too; x is one letter",
    "",
]
TERMS = {"whale", "café", "naïve", "cafe", "snake", "case", "word", "42nd", "鯨は海にいる", "i̇stanbul", "istanbul",
         "ıstanbul", "οδος", "straße", "strasse", "ⅷ", "٣", "x", "the", "and"}
TEXTS = [
    "Natural selection, natural\n  SELECTION and unnatural selections: three. Natural, selection: none.",
    "The sperm\twhale.\nSperm whale\x1csperm\x1fwhale sperm-whale",
    "İstanbul is istanbul IS ISTANBUL ıs ıstanbul",
    "a.b a.b axb A.B",
    "ΟΔΟΣ ΕΙΝΑΙ οδος ειναι; straße ist STRASSE ist",
    "Heading\ntext under the heading",
    "",
]
PHRASES = ["natural selection", "sperm whale", "is istanbul", "İS İSTANBUL", "a.b", "οδος ειναι", "straße ist", "heading text", "whale"]


def test_highlights_are_those_of_the_python_tokeniser():
    assert native.line_spans(LINES, TERMS) == [term_spans(line, TERMS) for line in LINES]
    assert native.line_spans(LINES, {"the"}) == [[] for _ in LINES]  # a stopword is never a term
    assert native.line_spans(LINES, set()) == [[] for _ in LINES]


@pytest.mark.parametrize("phrase", PHRASES)
def test_phrase_counts_are_those_of_the_python_regex(phrase):
    pattern = phrase_pattern(phrase)
    assert native.phrase_counts(pattern, TEXTS) == [len(pattern.findall(text)) for text in TEXTS]


def test_only_phrases_are_counted_natively():
    import re

    untouched = iter(TEXTS)
    assert native.phrase_counts(re.compile(r"wh\w+"), untouched) is None
    assert next(untouched) == TEXTS[0]  # left for the Python code to scan


def test_search_is_the_same_without_the_extension(engine, monkeypatch):
    queries = ['"sperm whale" squid', 'whale "white whale"', "+whale -ship", '"the pequod"']
    with_it = [engine.search(q, mode="hybrid") for q in queries]
    assert any(with_it)
    monkeypatch.setattr(native, "isis_native", None)
    engine._publish()  # a new snapshot, so the phrase counts are worked out again
    assert [engine.search(q, mode="hybrid") for q in queries] == with_it
