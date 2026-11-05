"""Sentence embeddings from small ONNX models on the CPU. The app offers the models in MODELS:
MongoDB/mdbr-leaf-ir (6-layer BERT, 384 dimensions, 23M parameters), the default, and
TaylorAI/bge-micro-v2 (3 layers, 17M parameters), which is about 1.4 times faster to embed with
but finds less by meaning."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import zlib
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .text import tokenize

log = logging.getLogger(__name__)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # read when huggingface_hub is imported

# The models the app offers, each by its Hugging Face repo; the first is the default. Any repo with
# ONNX weights and a tokenizer.json can join the list.
MODELS = ("MongoDB/mdbr-leaf-ir", "TaylorAI/bge-micro-v2")
DEFAULT_MODEL = MODELS[0]


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


class Embedder:
    """L2-normalised embeddings from any Hugging Face repo that ships ONNX weights and a
    tokenizer.json. Pooling (mean or CLS) is read from the repo's sentence-transformers config, and
    so are the texts some models want in front of a query or a passage (MongoDB/mdbr-leaf-ir wants
    "Represent this sentence for searching relevant passages: " before a query)."""

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

        log.info("loading the embedding model %s", model_id)
        self.model_id = model_id
        self.query_prompt, self.document_prompt = self._prompts()
        # Identifies vectors in the index cache. A passage prompt changes every vector, so it is part of the name.
        self.name = f"{model_id}:{onnx_file}:{max_tokens}" + (f":{self.document_prompt}" if self.document_prompt else "")
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
        model_path, data_path = self._fetch(onnx_file), None
        try:  # a large model keeps its weights in a second file beside it, which onnxruntime opens itself
            data_path = self._fetch(f"{onnx_file}_data")
        except Exception:
            pass  # most models have none
        try:
            self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        except Exception as e:
            # Newer onnxruntime (1.31 does, 1.23 doesn't) wants the weights file in the same folder as
            # the model, but the Hugging Face cache keeps each file in a folder of its own.
            if data_path is None or "External data path" not in str(e):
                raise
            model_path = _side_by_side(self.model_id, model_path, data_path)
            self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.dim = self.session.get_outputs()[0].shape[-1]

        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_size = cache_size
        self._cache_lock = threading.Lock()  # searches and the related panel embed at the same time

    def _fetch(self, filename: str) -> str:
        """A model file from the local Hugging Face cache. Only a file the cache doesn't have is
        downloaded (the first run); with the model cached, nothing goes over the network. A file
        the repo is known not to have raises FileNotFoundError, also without the network."""
        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        cached = try_to_load_from_cache(self.model_id, filename)  # reads the cache folder only
        if isinstance(cached, str):
            return cached
        if cached is not None:  # cached as missing from the repo
            raise FileNotFoundError(f"{self.model_id} has no {filename}")
        log.info("downloading %s of %s (only the first time)", filename, self.model_id)
        return hf_hub_download(self.model_id, filename)

    def _prompts(self) -> tuple[str, str]:
        """What goes before a query and before a passage, from the "prompts" of the repo's
        sentence-transformers config. Most models want nothing."""
        try:
            with open(self._fetch("config_sentence_transformers.json")) as f:
                prompts = json.load(f).get("prompts") or {}
        except Exception:
            return "", ""
        return prompts.get("query") or "", prompts.get("document") or prompts.get("passage") or ""

    def embed(self, texts: list[str], cache: bool = False, query: bool = False) -> np.ndarray:
        """One normalised row per text. `query=True` embeds search queries, which some models want
        marked; everything else, task paragraphs included, is embedded like a passage of the corpus.
        `cache=True` remembers results (for task paragraphs, which are re-embedded on every
        refresh of the related panel)."""
        prompt = self.query_prompt if query else self.document_prompt
        if prompt:
            texts = [prompt + t for t in texts]
        if not cache:
            return self._embed(texts)
        with self._cache_lock:
            found = {t: self._cache[t] for t in texts if t in self._cache}
            for t in found:
                self._cache.move_to_end(t)
        missing = list(dict.fromkeys(t for t in texts if t not in found))
        if missing:
            fresh = dict(zip(missing, self._embed(missing)))
            found.update(fresh)
            with self._cache_lock:
                self._cache.update(fresh)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.stack([found[t] for t in texts])

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


def _side_by_side(model_id: str, *files: str) -> str:
    """The first of these model files, with the others beside it in one folder of isis's own cache:
    hard links to the Hugging Face cache where the filesystem allows it, copies otherwise. Returns
    the path of the first."""
    folder = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "isis" / "onnx" / model_id.replace("/", "--")
    folder.mkdir(parents=True, exist_ok=True)
    for file in files:
        source, target = Path(os.path.realpath(file)), folder / Path(file).name
        if target.exists() and (os.path.samefile(source, target) or
                                (target.stat().st_size, target.stat().st_mtime) == (source.stat().st_size, source.stat().st_mtime)):
            continue
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        tmp.unlink(missing_ok=True)
        try:
            os.link(source, tmp)
        except OSError:  # another filesystem
            log.info("copying %s of %s to %s", Path(file).name, model_id, folder)
            shutil.copy2(source, tmp)
        os.replace(tmp, target)
    return str(folder / Path(files[0]).name)


@lru_cache(maxsize=100_000)
def _term_vector(term: str, dim: int) -> np.ndarray:
    return np.random.default_rng(zlib.crc32(term.encode())).standard_normal(dim).astype(np.float32)


class HashEmbedder:
    """Stand-in for tests and model-free experiments: a random projection of the bag of stemmed
    words. Passages that share vocabulary get similar vectors; it knows nothing about meaning."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.name = self.model_id = f"hash:{dim}"

    def embed(self, texts: list[str], cache: bool = False, query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, text in enumerate(texts):
            for term in tokenize(text):
                out[i] += _term_vector(term, self.dim)
        return _normalize(out)
