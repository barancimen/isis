from isis.engine.text import term_spans, tokenize, tokenize_with_surface


def test_tokenize_lowercases_stems_and_drops_stopwords():
    assert tokenize("The Whales were diving, and a whale dives!") == ["whale", "dive", "whale", "dive"]


def test_single_letters_and_underscores_are_not_terms():
    assert tokenize("a b_c x 42") == ["42"]


def test_surface_forms_are_kept():
    assert tokenize_with_surface("Whaling ships") == [("whale", "whaling"), ("ship", "ships")]


def test_term_spans_highlight_by_stem():
    text = "Whales and the whaling ship"
    assert [text[s:e] for s, e in term_spans(text, {"whale"})] == ["Whales", "whaling"]
