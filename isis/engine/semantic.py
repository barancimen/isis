"""Sentence embeddings from a small ONNX model on the CPU. The default is TaylorAI/bge-micro-v2
(3-layer BERT, 384 dimensions, mean pooling, 17 MB quantised)."""

from __future__ import annotations

import json
import zlib
from collections import OrderedDict
from functools import lru_cache

import numpy as np

from .text import tokenize

DEFAULT_MODEL = "TaylorAI/bge-micro-v2"


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


class Embedder:
    """L2-normalised embeddings from any Hugging Face repo that ships ONNX weights and a
    tokenizer.json. Pooling (mean or CLS) is read from the repo's sentence-transformers config."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        onnx_file: str = "onnx/model.onnx",  # the int8 file is no faster on CPU here and shifts neighbours
        max_tokens: int = 256,
        batch_size: int = 8,  # larger batches need hundreds of MB for attention buffers
        threads: int | None = None,
        cache_size: int = 4096,
    ):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_id = model_id
        self.name = f"{model_id}:{onnx_file}:{max_tokens}"  # identifies vectors in the index cache
        self.batch_size = batch_size

        self.tokenizer = Tokenizer.from_file(self._fetch("tokenizer.json"))
        # Text longer than max_tokens is split into overlapping windows whose embeddings are averaged.
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=max_tokens, stride=max_tokens // 8)
        self.pad_id = self.tokenizer.token_to_id("[PAD]") or 0

        try:
            with open(self._fetch("1_Pooling/config.json")) as f:
                self.pooling = "cls" if json.load(f).get("pooling_mode_cls_token") else "mean"
        except Exception:
            self.pooling = "mean"

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(self._fetch(onnx_file), options, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.dim = self.session.get_outputs()[0].shape[-1]

        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_size = cache_size

    def _fetch(self, filename: str) -> str:
        """A model file from the local Hugging Face cache, downloading it only if it isn't there."""
        from huggingface_hub import hf_hub_download

        try:
            return hf_hub_download(self.model_id, filename, local_files_only=True)
        except Exception:
            return hf_hub_download(self.model_id, filename)

    def embed(self, texts: list[str], cache: bool = False) -> np.ndarray:
        """One normalised row per text. `cache=True` remembers results (for task paragraphs,
        which are re-embedded on every refresh of the related panel)."""
        if not cache:
            return self._embed(texts)
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            for text, vec in zip(missing, self._embed(missing)):
                self._cache[text] = vec
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        for t in texts:
            self._cache.move_to_end(t)
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.stack([self._cache[t] for t in texts])

    def _embed(self, texts: list[str]) -> np.ndarray:
        windows, owner = [], []
        for i, enc in enumerate(self.tokenizer.encode_batch(texts)):
            for w in [enc, *enc.overflowing]:
                windows.append(w)
                owner.append(i)
        pooled = np.zeros((len(windows), self.dim), np.float32)
        order = sorted(range(len(windows)), key=lambda i: len(windows[i].ids))  # less padding
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            width = max(len(windows[i].ids) for i in idx)
            feeds = {
                name: np.full((len(idx), width), pad, np.int64)
                for name, pad in (("input_ids", self.pad_id), ("attention_mask", 0), ("token_type_ids", 0))
            }
            for row, i in enumerate(idx):
                w, n = windows[i], len(windows[i].ids)
                feeds["input_ids"][row, :n] = w.ids
                feeds["attention_mask"][row, :n] = w.attention_mask
                feeds["token_type_ids"][row, :n] = w.type_ids
            hidden = self.session.run(None, {k: v for k, v in feeds.items() if k in self.input_names})[0]
            if self.pooling == "cls":
                pooled[idx] = hidden[:, 0]
            else:
                mask = feeds["attention_mask"][..., None].astype(np.float32)
                pooled[idx] = (hidden * mask).sum(1) / np.maximum(mask.sum(1), 1e-9)
        out = np.zeros((len(texts), self.dim), np.float32)
        np.add.at(out, np.array(owner, dtype=np.int64), _normalize(pooled))
        return _normalize(out)


@lru_cache(maxsize=100_000)
def _term_vector(term: str, dim: int) -> np.ndarray:
    return np.random.default_rng(zlib.crc32(term.encode())).standard_normal(dim).astype(np.float32)


class HashEmbedder:
    """Stand-in for tests and model-free experiments: a random projection of the bag of stemmed
    words. Passages that share vocabulary get similar vectors; it knows nothing about meaning."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.name = f"hash:{dim}"

    def embed(self, texts: list[str], cache: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, text in enumerate(texts):
            for term in tokenize(text):
                out[i] += _term_vector(term, self.dim)
        return _normalize(out)
