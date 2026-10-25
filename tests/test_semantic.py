import numpy as np
import pytest

from isis.engine.semantic import DEFAULT_MODEL, HashEmbedder


def test_hash_embedder_is_normalised_and_follows_vocabulary():
    e = HashEmbedder(dim=64)
    v = e.embed(["the whale dives", "whales diving", "stock market prices", ""])
    assert v.shape == (4, 64)
    assert np.allclose(np.linalg.norm(v[:3], axis=1), 1, atol=1e-5)
    assert v[0] @ v[1] > 0.99 and abs(v[0] @ v[2]) < 0.5 and not v[3].any()


def _model_cached() -> bool:
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(DEFAULT_MODEL, "onnx/model.onnx", local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _model_cached(), reason="bge-micro-v2 not downloaded")
def test_bge_micro_ranks_paraphrase_above_unrelated_and_handles_long_text():
    from isis.engine.semantic import Embedder

    e = Embedder(max_tokens=32)
    a, b, c = e.embed(
        ["The whale swam through the cold ocean.", "A huge sea mammal swam in icy water.", "Stock prices fell today."]
    )
    assert a @ b > a @ c
    long = " ".join(["The whale swam through the cold ocean."] * 20)  # several 32-token windows
    (v,) = e.embed([long])
    assert np.isclose(np.linalg.norm(v), 1) and v @ a > 0.8
    cached = e.embed(["The whale swam through the cold ocean."], cache=True)
    assert np.allclose(cached[0], a, atol=1e-5)


def test_a_cached_model_never_touches_the_network(monkeypatch, tmp_path):
    import huggingface_hub

    from isis.engine.semantic import Embedder

    cached = tmp_path / "tokenizer.json"
    cached.write_text("{}")
    downloads = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: downloads.append(a) or str(tmp_path / "new"))
    lookups = {"tokenizer.json": str(cached), "1_Pooling/config.json": object(), "missing.json": None}
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, name: lookups[name])
    e = Embedder.__new__(Embedder)
    e.model_id = "some/model"
    assert e._fetch("tokenizer.json") == str(cached)
    with pytest.raises(FileNotFoundError):
        e._fetch("1_Pooling/config.json")  # known to be absent: no request either
    assert downloads == []
    e._fetch("missing.json")  # not in the cache yet: the one case that downloads
    assert downloads == [("some/model", "missing.json")]
