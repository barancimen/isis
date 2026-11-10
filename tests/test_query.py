import json

import pytest

from conftest import paths, write_corpus
from isis.engine.index import Engine
from isis.engine.search import QueryError, SearchConfig, correct_query, split_query, steering
from isis.engine.semantic import HashEmbedder


# ---- ranking and snippets ----

def test_search_returns_location_snippet_and_highlights(engine):
    (hit,) = engine.search("ambergris")
    assert (hit.path, hit.heading, hit.start_line, hit.end_line) == ("whales.md", "Whales", 4, 4)
    assert hit.match == (4, 0, len("Ambergris"))
    line = next(s for s in hit.snippet if s.line == 4)
    assert line.in_passage and line.marks == [(0, 9)]
    assert [s.line for s in hit.snippet] == [2, 3, 4, 5]  # two lines of context either side


def test_ranking_prefers_more_and_rarer_matches(engine):
    hits = engine.search("whale captain")
    assert hits[0].path == "ships/pequod.md" and "Ahab" in hits[0].text


def test_partly_typed_last_word_is_expanded(engine):
    assert "finches.md" in paths(engine.search("galap"))
    assert engine.search("galap ") == []  # a finished word is not expanded


def test_hybrid_mode_keeps_the_best_of_each_signal(engine):
    hits = engine.search("finches beaks", mode="hybrid", limit=3)
    assert hits[0].path == "finches.md" and hits[0].why["keyword_rank"] == 1 and hits[0].why["meaning_rank"] == 1
    assert {"bm25", "cosine"} <= set(hits[0].why)
    meaning_only = [h for h in engine.search("finches beaks", mode="hybrid") if h.why["keyword_rank"] is None]
    assert meaning_only and all(h.why["meaning_rank"] for h in meaning_only)  # found by meaning alone, and labelled so


def test_hybrid_search_does_not_bury_passages_without_embeddings(tmp_path):
    root = tmp_path / "c"
    (root / "plain").mkdir(parents=True)
    (root / "plain" / ".no-embed").touch()
    (root / "plain" / "exact.md").write_text("Coral reefs and coral atolls.\n")
    for i in range(6):  # embedded passages that share a word and so are close in meaning
        (root / f"e{i}.md").write_text(f"Coral and the sea, number {i}, with other words here.\n")
    e = Engine(root, embedder=HashEmbedder(), index_dir=tmp_path / "index")
    e.refresh()
    rank = lambda w: paths(e.search("coral reefs", mode="hybrid", config=SearchConfig(unembedded_weight=w))).index("plain/exact.md")
    assert rank(1.0) == 0 and rank(0.0) > 0
    hit = next(h for h in e.search("coral reefs", mode="hybrid") if h.path == "plain/exact.md")
    assert hit.why["embedded"] is False and hit.why["meaning_rank"] is None


# ---- passages without embeddings ----

@pytest.fixture(scope="module")
def plain_engine(tmp_path_factory):
    """Coral passages, five of them in a folder that is kept to keywords."""
    root = tmp_path_factory.mktemp("corpus")
    (root / "plain").mkdir()
    (root / "plain" / ".no-embed").touch()
    for i in range(5):
        (root / "plain" / f"p{i}.md").write_text(f"The coral reef number {i} is plain.\n")
    for i in range(10):
        (root / f"e{i}.md").write_text(f"Coral and fish number {i} in the sea.\n")
    e = Engine(root, embedder=HashEmbedder(), index_dir=tmp_path_factory.mktemp("index"))
    e.refresh()
    return e


@pytest.mark.parametrize("query", ["coral reef", '"coral reef"', "/coral/", "+coral", "coral +/reef|fish/"])
def test_with_nothing_to_count_for_them_passages_without_embeddings_are_left_out(plain_engine, query):
    e = plain_engine
    plain = lambda hits: [h for h in hits if h.path.startswith("plain/")]
    hits = e.search(query, mode="hybrid", config=SearchConfig(keyword_weight=0, unembedded_weight=0))
    assert hits and not plain(hits)
    assert plain(e.search(query, mode="hybrid", config=SearchConfig(keyword_weight=0.5, unembedded_weight=0)))
    kept = e.search(query, mode="hybrid", config=SearchConfig(keyword_weight=0, unembedded_weight=1))
    assert plain(kept) and all(h.score > 0 for h in kept)  # either weight keeps them
    assert plain(e.search(query, mode="hybrid"))  # the defaults keep them
    assert plain(e.search(query, mode="lexical"))
    lexical = e.search(query, mode="lexical", config=SearchConfig(keyword_weight=0, unembedded_weight=0))
    assert not plain(lexical)  # with or without 'meaning' ticked


# ---- the query language ----

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
    assert engine.search('"whale white"') == []
    hits = engine.search("whale /white/")  # prefer: matches first, the rest still found
    assert "white whale" in hits[0].text and len(hits) == 2
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


def test_phrases_and_regexes_see_headings_like_words_do(engine):
    assert paths(engine.search("/^Whales/")) == ["whales.md", "whales.md"]  # the heading starts each passage
    assert paths(engine.search('+"The Pequod" +ahab')) == ["ships/pequod.md"]  # its body never names the ship
    assert engine.search("ambergris -/^Whales/") == []

def test_exclusions_drop_passages_in_both_modes(engine):
    for mode in ("lexical", "hybrid"):
        assert "whales.md" not in paths(engine.search("whale -squid -ambergris", mode=mode))
        assert "ships/pequod.md" not in paths(engine.search('whale -"white whale" -/Pequod/', mode=mode))
        assert all("Ahab" not in h.text for h in engine.search("whale -/Ahab/", mode=mode))
    texts = [h.text for h in engine.search("whale -whaling")]  # whaling is another word, so whale stays
    assert "Ahab was her captain, and he hunted the white whale." in texts and not any("whaling" in t for t in texts)
    assert engine.search("-whale") == []  # nothing to look for


def test_invalid_patterns_are_reported(engine):
    with pytest.raises(QueryError):
        engine.search("/(unclosed/")
    with pytest.raises(QueryError):
        steering(split_query("-/[a-/"))


def test_steering_sorts_clauses_by_use():
    steer = steering(split_query('Whales +harpoon "white whale" +/beak/ -of -squid -/Ahab/'))
    assert dict(steer.words) == {"whales": 1, "harpoon": 1}
    assert [p.pattern for p in steer.patterns] == [r"white\s+whale", "beak"]
    assert steer.filters.required_terms == {"harpoon"} and [p.pattern for p in steer.filters.required] == ["beak"]
    assert steer.filters.excluded_terms == {"squid"} and len(steer.filters.excluded) == 2  # "of" is a stopword
    assert steer.text == "Whales harpoon white whale"


def test_a_match_in_the_heading_is_shown_and_opened_there(engine):
    (hit,) = [h for h in engine.search('"The Pequod"') if "Ahab" in h.text]  # its body never names the ship
    assert hit.match[0] == 0 and hit.focus_line == 0  # the heading line of ships/pequod.md
    assert hit.snippet[0].text == "# The Pequod" and hit.snippet[0].marks


def test_hits_list_every_highlight_in_the_passage(engine):
    (hit,) = engine.search("+ahab captain")
    words = [hit.text.split("\n")[n - hit.start_line][s:e] for n, s, e in hit.marks]
    assert sorted(words) == ["Ahab", "captain"]


def test_meaning_weight_zero_is_keyword_search(engine):
    for query in ("whale", "whale ship", '"white whale"', "/sq.id/", "whale +ship", "whale -ahab", "+captain"):
        lexical = [(h.path, h.start_line, h.score) for h in engine.search(query, mode="lexical")]
        off = [(h.path, h.start_line, h.score) for h in engine.search(query, mode="hybrid", config=SearchConfig(meaning_weight=0))]
        assert off == lexical, query
    assert engine.search("whale", mode="hybrid", config=SearchConfig(meaning_weight=0, keyword_weight=0)) == []


MOSTLY_RIGHT = {**{f"right{i}.md": f"The information desk number {i}." for i in range(12)},
                "typo.md": "The informaton desk was closed."}


def typo_corpus(tmp_path, **files):
    folder = tmp_path / "typos"
    write_corpus(folder, files)
    engine = Engine(folder, index_dir=tmp_path / "typo-index")
    engine.refresh()
    return engine


def test_words_with_typos_in_documents_are_found(tmp_path):
    engine = typo_corpus(tmp_path, **MOSTLY_RIGHT, **{"swap.md": "Some infromation about it.", "other.md": "Nothing to see here."})
    found = paths(engine.search("information"))
    assert found[-2:] in (["typo.md", "swap.md"], ["swap.md", "typo.md"])  # after every right spelling
    assert "other.md" not in found
    assert paths(engine.search("information", config=SearchConfig(typo_weight=0))) == found[:-2]
    assert "typo.md" not in paths(engine.search("+information"))  # + means exactly this word
    hit = next(h for h in engine.search("information") if h.path == "typo.md")
    assert hit.marks  # the misspelt word is highlighted


def test_words_about_as_common_as_yours_are_not_typos(tmp_path):
    engine = typo_corpus(tmp_path, **{"form.md": "A form to fill in.", "farm.md": "A farm with cows."})
    assert paths(engine.search("form")) == ["form.md"]


def test_a_misspelt_query_word_also_finds_the_right_spelling(tmp_path):
    engine = typo_corpus(tmp_path, **MOSTLY_RIGHT)
    found = paths(engine.search("informaton"))
    assert found[0] == "typo.md" and len(found) == 13


def test_typos_are_measured_on_words_not_stems(tmp_path):
    engine = typo_corpus(tmp_path, **{"a.md": "Information and naturally."})
    assert correct_query(engine.snapshot, "informaton") == "information"  # the stem is inform
    assert correct_query(engine.snapshot, "naturaly") == "naturally"


def test_words_survive_the_cache_and_old_caches_get_them(tmp_path):
    engine = typo_corpus(tmp_path, **{"a.md": "The informaton desk.", "b.md": "Plain words."})
    words = set(engine.snapshot.words.words)
    assert "informaton" in words
    again = Engine(tmp_path / "typos", index_dir=tmp_path / "typo-index")
    assert set(again.snapshot.words.words) == words
    index = tmp_path / "typo-index"
    meta_path = next(index.rglob("index.json"))
    meta = json.loads(meta_path.read_text())
    del meta["words"]  # a cache from before words were kept
    meta_path.write_text(json.dumps(meta))
    old = Engine(tmp_path / "typos", index_dir=index)
    assert set(old.snapshot.words.words) == words
    old.refresh()  # saves them
    assert "words" in json.loads(meta_path.read_text())


def test_only_unknown_plain_words_are_corrected(engine):
    snapshot = engine.snapshot
    assert correct_query(snapshot, "whlae").startswith("whal")
    assert correct_query(snapshot, "whlae  ship").endswith("  ship")
    for query in ("whale", '"whlae"', "+whlae", "-whlae", "/whlae/", "wha", "zzzzzzzz"):
        assert correct_query(snapshot, query) is None, query
    assert correct_query(snapshot, "whal") is None  # still being typed


def test_a_word_no_passage_contains_still_searches_by_meaning(engine):
    hits = engine.search("zzzzzzzz", mode="hybrid")
    assert hits and all(h.why["keyword_rank"] is None for h in hits)
    assert engine.search("zzzzzzzz", mode="lexical") == []
    assert engine.search("zz", mode="hybrid") == []  # too short to mean anything yet


SITES = {
    "hub.md": '---\nurl: "https://github.com/a"\nfinal_url: "https://web.archive.org/b"\n---\n\nThe whale in a repository.\n',
    "gist.md": "---\nurl: https://gist.github.com/c\n---\n\nThe whale in a gist.\n\nNo big fish here.\n",
    "archive.md": '---\nurl: "https://web.archive.org/d"\n---\n\nThe whale in an archive.\n',
    "plain.md": "The whale with no address.\n",
}


@pytest.fixture
def site_engine(tmp_path):
    engine = Engine(write_corpus(tmp_path / "sites", SITES), index_dir=tmp_path / "index", target_words=0)
    engine.refresh()
    return engine


def test_site_filters_by_the_host_of_url(site_engine):
    found = lambda query: sorted(set(paths(site_engine.search(query))))
    assert found("whale") == ["archive.md", "gist.md", "hub.md", "plain.md"]
    assert found("whale site:github.com") == ["gist.md", "hub.md"]  # with its subdomains
    assert found("whale SITE:https://www.GitHub.com/x") == ["gist.md", "hub.md"]
    assert found("whale site:gist.github.com") == ["gist.md"]
    assert found("whale site:hub.com") == []  # the whole name, not its end
    assert found("whale site:web.archive.org") == ["archive.md"]  # not final_url
    assert found("whale site:github.com site:web.archive.org") == ["archive.md", "gist.md", "hub.md"]
    assert found("whale -site:github.com") == ["archive.md", "plain.md"]
    assert found("-site:github.com") == []  # only exclusions, like -word
    assert found("whale site:") == ["archive.md", "gist.md", "hub.md", "plain.md"]  # still being typed


def test_site_narrows_words_and_alone_lists_the_site(site_engine):
    assert [h.text for h in site_engine.search("whale site:gist.github.com")] == ["The whale in a gist."]
    assert [h.text for h in site_engine.search("site:gist.github.com")] == ["The whale in a gist.", "No big fish here."]


def test_site_is_not_corrected_or_kept_as_a_word(site_engine):
    assert correct_query(site_engine.snapshot, "whale site:githab.com") is None
    assert [str(c) for c in split_query("+site:a.org -site:b.org")] == ["+site:a.org", "-site:b.org"]


def test_a_cache_from_before_sites_finds_them(site_engine, tmp_path):
    site_engine._save()
    cache = next((tmp_path / "index").rglob("index.json"))
    meta = json.loads(cache.read_text())
    for entry in meta["files"]:
        del entry["site"]
    cache.write_text(json.dumps(meta))
    engine = Engine(tmp_path / "sites", index_dir=tmp_path / "index", target_words=0)
    assert sorted(set(paths(engine.search("whale site:github.com")))) == ["gist.md", "hub.md"]


def test_a_word_finds_only_itself_not_other_forms(tmp_path):
    engine = typo_corpus(tmp_path, **{"a.md": "A conditional clause.", "b.md": "In good condition.", "c.md": "Three conditions."})
    assert paths(engine.search("conditional")) == ["a.md"]
    assert paths(engine.search("Condition")) == ["b.md"]  # case never counts
