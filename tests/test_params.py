import dataclasses
import json
import warnings

import numpy as np
import pytest
from conftest import write_corpus
from fastapi.testclient import TestClient

from isis.engine.index import Engine
from isis.engine.lexical import BM25, Vocabulary, term_counts
from isis.engine.params import configure, specs
from isis.engine.related import RelatedConfig, TaskPrefs
from isis.engine.search import SearchConfig
from isis.engine.semantic import HashEmbedder
from isis.server.app import create_app
from isis.server.tasks import TaskStore


def test_specs_describe_every_tunable_value():
    by_name = {s["name"]: s for s in specs(RelatedConfig)}
    assert by_name["z_strong"]["default"] == 3.25 and by_name["z_strong"]["type"] == "float"
    assert by_name["total_slots"]["type"] == "int"
    assert by_name["query"]["type"] == "choice" and by_name["query"]["choices"] == ["document", "paragraphs"]
    assert by_name["mmr_lambda.hidden"]["default"] == 0.6
    assert "lane_prior" not in by_name and "mmr_pool" not in by_name  # not tunable from the page
    assert {s["name"] for s in specs(SearchConfig)} == {"k1", "b", "rrf_k", "keyword_weight", "meaning_weight", "unembedded_weight", "candidates"}


def test_configure_applies_valid_overrides_and_ignores_the_rest():
    config = configure(
        RelatedConfig,
        {"z_strong": "3.25", "total_slots": 10.6, "query": "paragraphs", "mmr_lambda.core": 0.2,
         "hidden_z": 99, "nonsense": 1, "mmr_lambda.nope": 0.5, "mmr_lambda": 0.5, "query.x": 1, "z_weak": "abc",
         "mmr_pool": 5},
    )
    assert config.z_strong == 3.25 and config.total_slots == 11 and config.query == "paragraphs"
    assert config.mmr_lambda == {"core": 0.2, "hidden": 0.6, "bridges": 0.6, "gaps": 0.6}
    assert config.hidden_z == 5.0  # clamped to the range
    assert config.z_weak == 1.75 and config.mmr_pool == 100
    assert configure(RelatedConfig, {"query": "sideways"}).query == "document"
    assert RelatedConfig().mmr_lambda["core"] == 0.7  # the default dict is not shared or changed


def test_bm25_variants_match_a_fresh_index():
    docs = [["whale", "sea", "whale"], ["ship", "sea"], ["ship", "captain", "ship", "sea", "storm", "storm"]]
    vocab = Vocabulary()
    pieces = term_counts(docs, vocab)
    base = BM25(*pieces, n_terms=len(vocab))
    ids, weights = [vocab.get("ship"), vocab.get("storm")], [1.0, 1.0]
    fresh = BM25(*pieces, n_terms=len(vocab), k1=2.0, b=0.2).scores(ids, weights)
    assert base.with_params(2.0, 0.2).scores(ids, weights) == pytest.approx(fresh)
    assert base.with_params(1.2, 0.75) is base


def test_snapshot_keeps_a_few_variants(engine):
    snap = engine.snapshot
    assert snap.bm25_for(1.2, 0.75) is snap.bm25
    first = snap.bm25_for(0.5, 0.5)
    assert snap.bm25_for(0.5, 0.5) is first
    for k1 in (0.1, 0.2, 0.3, 0.4, 0.6):
        snap.bm25_for(k1, 0.5)
    assert len(snap._bm25_variants) <= 4


def test_search_config_changes_the_fusion(engine):
    meaning_only = engine.search("finches beaks", mode="hybrid", config=SearchConfig(keyword_weight=0))
    keywords_only = engine.search("finches beaks", mode="hybrid", config=SearchConfig(meaning_weight=0))
    by_meaning = [h.why["meaning_rank"] for h in meaning_only]
    assert by_meaning == sorted(by_meaning)
    assert keywords_only and all("meaning_rank" not in h.why for h in keywords_only)  # plain keyword search
    scores = [h.score for h in keywords_only]
    assert scores == sorted(scores, reverse=True)


@pytest.fixture
def client(engine, corpus):
    with TestClient(create_app(engine, TaskStore(corpus / "_tasks"), background=False, watch=False)) as c:
        yield c


def test_api_lists_params_and_accepts_overrides(client):
    body = client.get("/api/params").json()
    assert {"search", "related"} <= set(body) and any(p["name"] == "rrf_k" for p in body["search"])

    plain = client.get("/api/search", params={"q": "whale ship", "mode": "hybrid"}).json()["hits"]
    tuned = client.get(
        "/api/search", params={"q": "whale ship", "mode": "hybrid", "params": json.dumps({"keyword_weight": 0})}
    ).json()["hits"]
    assert [h["why"]["meaning_rank"] for h in tuned] == sorted(h["why"]["meaning_rank"] for h in tuned)
    assert plain and tuned
    assert client.get("/api/search", params={"q": "whale", "params": "[1]"}).status_code == 400
    assert client.get("/api/search", params={"q": "whale", "params": "{nope"}).status_code == 400

    client.post("/api/tasks", json={"name": "t"})
    small = client.post("/api/tasks/t/related", json={"text": "The sperm whale and ambergris.", "params": {"total_slots": 4}})
    assert small.status_code == 200 and small.json()["stats"]["slots"]
    assert sum(small.json()["stats"]["slots"].values()) <= 4


ODD_FILES = {
    "words.md": "# Whales\n\nThe white whale and the sea.\n\nAhab hunted the whale across the sea.\n",
    "stopwords.md": "the and of it\n\nthe\n\n# of\n\nand it is\n",
    "one.md": "whale\n",
    "symbols.md": "---\n\n***\n\n1 2 3\n\n$$$ ??? !!!\n",
}


def _is_finite(x) -> bool:
    if dataclasses.is_dataclass(x):
        return all(_is_finite(getattr(x, f.name)) for f in dataclasses.fields(x))
    if isinstance(x, dict):
        return all(_is_finite(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(_is_finite(v) for v in x)
    return not isinstance(x, (float, np.floating)) or bool(np.isfinite(x))


@pytest.mark.parametrize("cls", [SearchConfig, RelatedConfig])
def test_every_slider_end_gives_finite_results_without_warnings(tmp_path, cls):
    root = write_corpus(tmp_path / "corpus", ODD_FILES)
    engine = Engine(root, embedder=HashEmbedder())
    engine.refresh()
    text = "# Whales\n\nThe white whale.\n\n# Stopwords\n\nthe and of\n\nThe sea and Ahab.\n"
    prefs = TaskPrefs(steer='whale "the sea" /wh.le/')
    settings = [{}]
    for spec in specs(cls):
        ends = spec["choices"] if spec.get("choices") else [spec["min"], spec["max"]]
        settings += [{spec["name"]: end} for end in ends]
    settings += [{s["name"]: s["min"] for s in specs(cls) if not s.get("choices")},
                 {s["name"]: s["max"] for s in specs(cls) if not s.get("choices")}]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for overrides in settings:
            config = configure(cls, overrides)
            if cls is SearchConfig:
                for query, mode in (("white whale", "hybrid"), ('"the sea"', "lexical"), ("+/wh.le/ sea", "hybrid"), ("the", "hybrid")):
                    assert _is_finite(engine.search(query, limit=20, mode=mode, config=config)), (overrides, query)
            else:
                assert _is_finite(engine.related(text, prefs=prefs, config=config)), overrides
