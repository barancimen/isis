import pytest

from isis.engine.search import QueryError, split_query, steering


def paths(hits):
    return [h.path for h in hits]


def test_query_tokens():
    clauses = split_query('whale +ship -"sperm whale" /cap\\w+/ +/x/ -/ahab/ /typ')
    assert [(c.op, c.kind, c.value) for c in clauses] == [
        ("", "word", "whale"),
        ("+", "word", "ship"),
        ("-", "phrase", "sperm whale"),
        ("", "regex", "cap\\w+"),
        ("+", "regex", "x"),
        ("-", "regex", "ahab"),
        ("", "partial", "/typ"),
    ]
    assert " ".join(map(str, clauses[:-1])) == 'whale +ship -"sperm whale" /cap\\w+/ +/x/ -/ahab/'
    assert [str(c) for c in split_query('"half typed phrase')] == ['"half typed phrase"']
    assert [str(c) for c in split_query("well-known - +")] == ["well-known"]


def test_prefer_require_and_never(engine):
    assert paths(engine.search('"white whale"')) == ["ships/pequod.md"]  # a phrase alone: exact matches only
    hits = engine.search("whale /white/")  # prefer: matches first, the rest still found
    assert "white whale" in hits[0].text and len(hits) == 4
    assert [h.text for h in engine.search("whale +/white/")] == [hits[0].text]  # require
    assert {h.path for h in engine.search("+whale")} == {"whales.md", "ships/pequod.md"}
    assert all("Ambergris" not in h.text for h in engine.search("+whale -ambergris"))
    assert engine.search("+whale +finch") == []


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
        steering(split_query("-/[a-/"))


def test_steering_sorts_clauses_by_use():
    steer = steering(split_query('Whales +harpoon "white whale" +/beak/ -of -squid -/Ahab/'))
    assert dict(steer.words) == {"whale": 1, "harpoon": 1}
    assert [p.pattern for p in steer.patterns] == [r"white\s+whale", "beak"]
    assert steer.filters.required_terms == {"harpoon"} and [p.pattern for p in steer.filters.required] == ["beak"]
    assert steer.filters.excluded_terms == {"squid"} and len(steer.filters.excluded) == 2  # "of" is a stopword
    assert steer.text == "Whales harpoon white whale"
