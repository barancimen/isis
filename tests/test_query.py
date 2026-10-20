import pytest

from isis.engine.search import QueryError, exclusion_filters, split_query


def paths(hits):
    return [h.path for h in hits]


def test_query_tokens():
    assert split_query('whale -ship "white whale" -"sperm whale" /cap\\w+/ -/ahab/ /typ') == [
        (False, "word", "whale"),
        (True, "word", "ship"),
        (False, "phrase", "white whale"),
        (True, "phrase", "sperm whale"),
        (False, "regex", "cap\\w+"),
        (True, "regex", "ahab"),
        (False, "partial", "/typ"),
    ]
    assert split_query('"half typed phrase') == [(False, "phrase", "half typed phrase")]
    assert split_query("well-known -") == [(False, "word", "well-known")]


def test_regex_must_match_and_ranks_with_the_words(engine):
    assert paths(engine.search("/sq.id/")) == ["whales.md"]
    assert engine.search("/SQUID/") == []  # a capital letter makes it case-sensitive
    hits = engine.search("captain /whal(e|ing)/")
    assert paths(hits)[0] == "ships/pequod.md" and "Ahab" in hits[0].text
    assert {"whales.md", "ships/pequod.md"} <= set(paths(hits))  # matching the pattern is enough
    marked = [line.text[a:b] for h in engine.search("/sq.id/") for line in h.snippet for a, b in line.marks]
    assert marked == ["squid"]
    marked = [line.text[a:b] for h in engine.search('"white whale"') for line in h.snippet for a, b in line.marks]
    assert marked == ["white whale"]  # the phrase and its words make one mark


def test_exclusions_drop_passages_in_both_modes(engine):
    for mode in ("lexical", "hybrid"):
        assert "whales.md" not in paths(engine.search("whale -squid -ambergris", mode=mode))
        assert "ships/pequod.md" not in paths(engine.search('whale -"white whale" -/Pequod/', mode=mode))
        assert all("Ahab" not in h.text for h in engine.search("whale -/Ahab/", mode=mode))
    assert "ships/pequod.md" not in paths(engine.search("whale -whaling"))  # by stem: "whaling" drops "whale" too
    assert engine.search("-whale") == []  # nothing to look for


def test_invalid_patterns_are_reported(engine):
    with pytest.raises(QueryError):
        engine.search("/(unclosed/")
    with pytest.raises(QueryError):
        exclusion_filters(["/[a-/"])


def test_exclusion_lists(engine):
    f = exclusion_filters(["Whales", '"white whale"', "/beak/", "of"])
    assert f.excluded_terms == {"whale"} and len(f.excluded) == 3
    texts = [p.text for p in engine.snapshot.passages]
    assert all(not ok for ok in f.allowed(engine.snapshot))  # every passage has one of them
    assert any("finches" in t.lower() for t in texts)
