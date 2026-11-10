from isis.engine.text import term_spans, tokenize, tokenize_with_surface


def test_tokenize_lowercases_and_drops_stopwords_but_never_stems():
    assert tokenize("The Whales were diving, and a whale dives!") == ["whales", "diving", "whale", "dives"]
    assert tokenize("Conditional condition") == ["conditional", "condition"]


def test_single_letters_and_underscores_are_not_terms():
    assert tokenize("a b_c x 42") == ["42"]


def test_surface_forms_are_kept():
    assert tokenize_with_surface("Whaling ships") == [("whaling", "whaling"), ("ships", "ships")]


def test_term_spans_highlight_the_word_in_any_case():
    text = "Whales and the whaling ship, whales"
    assert [text[s:e] for s, e in term_spans(text, {"whales"})] == ["Whales", "whales"]
