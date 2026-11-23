import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import write_corpus
from isis import cli
from isis.engine import semantic
from isis.engine.index import Engine, saved_vectors
from isis.engine.models import ModelError, ModelList, made_by, repo_id
from isis.engine.semantic import MODELS, HashEmbedder

LEAF, MICRO = MODELS
SMALL = "BAAI/bge-small-en-v1.5"


class FakeEmbedder(HashEmbedder):
    """Stands in for a model from Hugging Face: named as one, and noted when it is loaded."""

    loaded: list[str] = []
    embedded: list[str] = []  # the passages any of them embedded
    refused: dict[str, Exception] = {}  # what loading these models raises

    def __init__(self, model_id: str):
        if "broken" in model_id:
            raise FileNotFoundError(f"{model_id} has no onnx/model.onnx")
        if model_id in FakeEmbedder.refused:
            raise FakeEmbedder.refused[model_id]
        super().__init__(dim=32 + 16 * sum(map(ord, model_id)) % 64)
        self.model_id, self.name = model_id, f"{model_id}:onnx/model.onnx:256"
        self.seen: list[str] = []
        FakeEmbedder.loaded.append(model_id)

    def embed(self, texts, cache=False, query=False):
        self.seen.extend(texts)
        FakeEmbedder.embedded.extend(t for t in texts if t != cli.TRIAL)
        return super().embed(texts, cache, query)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(semantic, "Embedder", FakeEmbedder)
    monkeypatch.setattr(semantic, "fetch_settings", lambda model_id: None)  # nothing to ask Hugging Face
    monkeypatch.setattr(FakeEmbedder, "loaded", [])
    monkeypatch.setattr(FakeEmbedder, "embedded", [])
    monkeypatch.setattr(FakeEmbedder, "refused", {})
    return FakeEmbedder


def run(*argv) -> None:
    cli.main([str(a) for a in argv])


def refused(*argv) -> str:
    with pytest.raises(SystemExit) as stopped:
        run(*argv)
    return str(stopped.value)


def chosen(corpus, *named, all_models=False, no_model=False) -> tuple[list[str], list[str]]:
    """The models isis would load for this corpus, and its other models, which the page greys out."""
    return cli.models(SimpleNamespace(model=list(named), all_models=all_models, no_model=no_model), [corpus])


def vector_files(corpus) -> dict[str, bytes]:
    """Each model's saved embeddings in a corpus folder, by model."""
    return {name.partition(":")[0]: b"".join(p.read_bytes() for p in files if p.suffix != ".json")
            for name, files in saved_vectors(corpus / ".isis").items()}


def test_a_corpus_starts_with_the_built_in_models_and_reading_them_writes_nothing(corpus):
    listed = ModelList.of(corpus)
    assert (listed.models, listed.active) == ([LEAF, MICRO], LEAF)
    assert not (corpus / ".isis").exists()
    assert chosen(corpus) == ([LEAF], [MICRO])  # the active one alone
    assert not (corpus / ".isis").exists()


def test_isis_can_be_started_with_several_models_or_all_of_them(corpus, fake):
    assert chosen(corpus, MICRO, SMALL, MICRO.lower()) == ([MICRO, SMALL], [LEAF])  # for one run, the first the default
    assert chosen(corpus, all_models=True) == ([LEAF, MICRO], [])
    assert chosen(corpus, MICRO, all_models=True) == ([MICRO, LEAF], [])  # those named go first
    assert chosen(corpus, MICRO, no_model=True) == chosen(corpus, all_models=True, no_model=True) == ([], [])
    run("model", "add", SMALL, corpus)
    assert chosen(corpus) == ([SMALL], [LEAF, MICRO])
    assert chosen(corpus, all_models=True) == ([SMALL, LEAF, MICRO], [])  # the active one first
    run("model", "remove", LEAF, corpus)
    assert chosen(corpus, all_models=True) == ([SMALL, MICRO], [])  # a removed model is neither loaded nor shown
    assert chosen(corpus, LEAF) == ([LEAF], [MICRO, SMALL])  # unless it is named


def test_a_repository_is_named_by_its_id_or_its_address():
    assert repo_id(" BAAI/bge-small-en-v1.5 ") == SMALL
    assert repo_id("https://huggingface.co/BAAI/bge-small-en-v1.5") == SMALL
    assert repo_id("huggingface.co/BAAI/bge-small-en-v1.5/tree/main/onnx?x=1") == SMALL
    for bad in ("", "a/b/c", "two words", "a,b", "../x", "org/"):
        with pytest.raises(ModelError):
            repo_id(bad)
    assert made_by(f"{LEAF}:onnx/model.onnx:256:passage: ", LEAF.lower()) and made_by("hash:64", "hash:64")
    assert not made_by(f"{LEAF}-v2:onnx/model.onnx:256", LEAF)


def test_adding_a_model_tries_it_and_makes_it_active(corpus, fake, capsys):
    run("model", "add", f"https://huggingface.co/{SMALL}", corpus)
    assert fake.loaded == [SMALL]  # loaded once, which also downloads it
    saved = corpus / ".isis" / "models.json"
    assert json.loads(saved.read_text()) == {"models": [LEAF, MICRO, SMALL], "active": SMALL}
    assert saved.stat().st_mode & 0o044 == 0o044 & ~_umask()  # a server running as someone else can read it
    assert "made it the active model" in capsys.readouterr().out
    assert chosen(corpus) == ([SMALL], [LEAF, MICRO])

    run("model", "add", "other/model", corpus, "--inactive")
    run("model", "add", SMALL.upper(), corpus)  # the same repository, as Hugging Face reads names
    listed = ModelList.of(corpus)
    assert (listed.models, listed.active) == ([LEAF, MICRO, SMALL, "other/model"], SMALL)
    out = capsys.readouterr().out
    assert f"Added other/model. {SMALL} stays the active model." in out and "already the active model" in out
    assert [p.name for p in (corpus / ".isis").iterdir()] == ["models.json"]  # no temporary file left


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def test_a_model_that_cannot_be_used_is_not_added(corpus, fake, capsys):
    assert "broken/model has no onnx/model.onnx" in refused("model", "add", "broken/model", corpus)
    assert "not the name of a Hugging Face repository" in refused("model", "add", "two words", corpus)
    assert not (corpus / ".isis").exists()
    assert refused("model", "list") == "2" and "name a folder" in capsys.readouterr().err


def test_why_a_model_cannot_be_used_is_said_plainly(corpus, fake):
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    def raised(kind, message):  # as Hugging Face raises them, without the answer they are made from
        error = kind.__new__(kind)
        error.args = (message,)
        return error

    fake.refused.update({
        "gated/model": raised(GatedRepoError, "401 Client Error. (Request ID: Root=1-6b04)\n\nCannot access gated repo"),
        "no/model": raised(RepositoryNotFoundError, "401 Client Error. (Request ID: Root=1-6b04)"),
        "down/model": raised(HfHubHTTPError, "503 Server Error. (Request ID: Root=1-6b04)\n\nService Unavailable"),
        "odd/model": TypeError("expected int\nand more"),
    })
    assert refused("model", "add", "gated/model", corpus).endswith("only to accounts that were granted access")
    assert refused("model", "add", "no/model", corpus).endswith("no repository by that name, or it is private")
    assert refused("model", "add", "down/model", corpus) == "isis: cannot use down/model: Hugging Face answered 503 Server Error"
    assert refused("model", "add", "odd/model", corpus) == "isis: cannot use odd/model: expected int"
    assert not (corpus / ".isis").exists()


def test_a_model_is_not_tried_before_its_settings_are_known(corpus, fake, monkeypatch):
    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    monkeypatch.undo()  # the real fetch_settings, over a Hugging Face that is stood in for below
    monkeypatch.setattr(semantic, "Embedder", FakeEmbedder)
    monkeypatch.setattr(FakeEmbedder, "loaded", [])
    asked = []
    known = {"1_Pooling/config.json": object()}  # known to be missing; nothing is known of the other file

    def download(repo, name):
        asked.append(name)
        raise unreachable if unreachable else EntryNotFoundError("404")

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, name: known.get(name))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    unreachable = LocalEntryNotFoundError("offline")
    assert "Hugging Face can't be reached" in refused("model", "add", SMALL, corpus)
    assert FakeEmbedder.loaded == [] and not (corpus / ".isis").exists()  # not loaded with a setting unknown
    unreachable = None
    run("model", "add", SMALL, corpus)  # the repo has no such file, which is fine
    assert asked == ["config_sentence_transformers.json"] * 2 and FakeEmbedder.loaded == [SMALL]


def test_serve_loads_the_active_model_alone(corpus, fake, monkeypatch):
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    for key in ("ISIS_CORPUS", "ISIS_OSIRIS", "ISIS_MODELS", "ISIS_OTHER_MODELS", "ISIS_HOSTS", "ISIS_BUILD_NATIVE"):
        monkeypatch.setenv(key, "")  # put back as they were afterwards
    monkeypatch.delenv("ISIS_URL", raising=False)

    def serve(*options) -> tuple[str, str]:  # the models it loads, and those the page greys out
        run("serve", corpus, "--no-build", *options)
        return os.environ["ISIS_MODELS"], os.environ["ISIS_OTHER_MODELS"]

    assert serve() == (LEAF, MICRO)
    assert serve("--model", MICRO, "--model", LEAF) == (f"{MICRO},{LEAF}", "")  # several, for one run
    assert serve("--all-models") == (f"{LEAF},{MICRO}", "")
    assert serve("--no-model") == ("", "")
    run("model", "use", MICRO, corpus)
    assert serve() == (MICRO, LEAF)
    assert serve("--all-models") == (f"{MICRO},{LEAF}", "")
    run("model", "remove", LEAF, corpus)
    assert serve() == (MICRO, "")  # a removed model is no longer shown
    run("model", "remove", MICRO, corpus)
    assert serve() == ("", "")  # no model left: keywords only


def test_only_the_active_model_is_loaded_and_the_others_keep_their_embeddings(corpus, fake):
    run("index", corpus, "--model", LEAF, "--model", MICRO)  # both, as before there was an active model
    before = vector_files(corpus)
    assert set(before) == {LEAF, MICRO}

    (corpus / "new.md").write_text("Ambergris washed up on the beach.\n")
    fake.loaded.clear()
    run("index", corpus)
    assert fake.loaded == [LEAF]  # the other model isn't even loaded
    after = vector_files(corpus)
    assert after[MICRO] == before[MICRO] and after[LEAF] != before[LEAF]

    run("model", "use", MICRO.lower(), corpus)
    fake.loaded.clear()
    engine = Engine(corpus, embedder=[FakeEmbedder(m) for m in chosen(corpus)[0]])
    assert fake.loaded == [MICRO] and engine.refresh()["embedded"] == 1  # only what changed meanwhile
    assert engine.embedder.seen == ["Ambergris washed up on the beach."]
    assert vector_files(corpus)[LEAF] == after[LEAF]


def test_removing_a_model_never_deletes_its_embeddings(tmp_path, fake, capsys):
    a, b = write_corpus(tmp_path / "a"), write_corpus(tmp_path / "b", {"x.md": "Finches differ in their beaks.\n"})
    run("index", a, b, "--model", LEAF, "--model", MICRO)
    old = b / ".isis" / "models" / "old.npz"  # as embeddings were saved before
    np.savez(old, name=f"{MICRO}:onnx/model.onnx:256", keys=np.zeros(0, np.uint64), vectors=np.zeros((0, 64), np.float16))
    kept = vector_files(a), vector_files(b)
    files = sorted(p.name for folder in (a, b) for p in (folder / ".isis" / "models").iterdir())

    run("model", "remove", MICRO, a, b)
    listed = ModelList.of(a)
    assert (listed.models, listed.active) == ([LEAF], LEAF)
    assert not (b / ".isis" / "models.json").exists()  # the list is the first folder's
    assert (vector_files(a), vector_files(b)) == kept  # in every folder, to the byte
    assert sorted(p.name for folder in (a, b) for p in (folder / ".isis" / "models").iterdir()) == files
    out = capsys.readouterr().out
    assert f"Removed {MICRO}. isis keeps its" in out and "uses them again if the model is added back" in out
    assert "not one of this corpus's models" in refused("model", "remove", MICRO, a, b)
    run("model", "list", a, b)
    assert f"{MICRO} is not one of the models. isis keeps its" in capsys.readouterr().out

    run("model", "add", MICRO, a, b)  # back again: nothing to embed
    fake.embedded.clear()
    run("index", a, b)
    assert fake.loaded[-1] == MICRO and fake.embedded == []
    assert (vector_files(a), vector_files(b)) == kept


def test_removing_the_active_model_needs_the_next_one_named(corpus, fake, capsys):
    run("index", corpus, "--model", LEAF, "--model", MICRO)
    before = vector_files(corpus)
    message = refused("model", "remove", LEAF, corpus)
    assert f"{LEAF} is the active model" in message and f"--next, which can be {MICRO}." in message
    assert "can't be the next active model" in refused("model", "remove", LEAF, corpus, "--next", SMALL)
    assert "can't be the next active model" in refused("model", "remove", LEAF, corpus, "--next", LEAF)
    assert not (corpus / ".isis" / "models.json").exists()  # nothing was done

    fake.loaded.clear()
    run("model", "remove", LEAF, corpus, "--next", MICRO)
    listed = ModelList.of(corpus)
    assert (listed.models, listed.active) == ([MICRO], MICRO)
    assert fake.loaded == [MICRO]  # tried before it becomes the active one
    assert f"{MICRO} is now the active model" in capsys.readouterr().out

    run("model", "remove", MICRO, corpus)  # the last one: nothing to name
    assert json.loads((corpus / ".isis" / "models.json").read_text()) == {"models": [], "active": None}
    assert chosen(corpus) == ([], [])
    assert "keywords only" in capsys.readouterr().out
    assert vector_files(corpus) == before  # none of this deleted anything
    run("model", "add", SMALL, corpus, "--inactive")  # the only model is the active one
    assert ModelList.of(corpus).active == SMALL


def test_embeddings_of_a_model_that_is_not_listed_are_kept_too(corpus, fake, capsys):
    run("index", corpus, "--model", SMALL)  # for one run, without adding it
    before = vector_files(corpus)
    run("model", "list", corpus)
    out = capsys.readouterr().out
    assert f"{LEAF}" in out and "active" in out and "no embeddings yet" in out
    assert f"{SMALL} is not one of the models. isis keeps its" in out
    assert "not one of this corpus's models" in refused("model", "remove", SMALL, corpus)
    assert vector_files(corpus) == before and not (corpus / ".isis" / "models.json").exists()


def test_choosing_the_active_model(corpus, fake, capsys):
    assert "Add it first" in refused("model", "use", SMALL, corpus)
    run("model", "use", LEAF, corpus)
    assert "already the active model" in capsys.readouterr().out and fake.loaded == []
    run("model", "use", MICRO, corpus)
    assert ModelList.of(corpus).active == MICRO and fake.loaded == [MICRO]
    run("model", "list", corpus)
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[:2] for line in lines[-2:]] == [[LEAF, "no"], [MICRO, "active"]]


def test_a_list_of_models_that_cannot_be_read_stops_isis(corpus, fake):
    saved = corpus / ".isis" / "models.json"
    saved.parent.mkdir()
    for text in ("not json", '{"models": "a/b"}', '{"models": ["a/b"], "active": "c/d"}', '["a/b"]', '{"models": [3]}'):
        saved.write_text(text)
        assert "cannot use the list of models" in refused("index", corpus)
        assert "cannot use the list of models" in refused("model", "add", SMALL, corpus)
        assert saved.read_text() == text
    saved.write_text('{"models": ["a/b", "c/d"]}')  # written by hand, without the active one: the first
    assert ModelList.of(corpus).active == "a/b"
    saved.write_text('{"models": ["a/b", "c/d"], "active": "C/D"}')
    assert ModelList.of(corpus).active == "c/d"


def test_a_model_added_back_in_another_case_finds_its_embeddings(corpus, fake):
    run("index", corpus, "--model", LEAF, "--model", MICRO)
    kept = vector_files(corpus)
    run("model", "remove", MICRO, corpus)
    (corpus / "new.md").write_text("Ambergris washed up on the beach.\n")  # changed while the model was away
    run("model", "add", MICRO.lower(), corpus)
    assert ModelList.of(corpus).active == MICRO  # spelt as its embeddings are
    fake.embedded.clear()
    run("index", corpus)
    assert fake.embedded == ["Ambergris washed up on the beach."]  # only the new passage
    assert set(vector_files(corpus)) == set(kept)  # and no second set of embeddings
