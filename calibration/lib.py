"""Shared helpers for the calibration scripts (see CALIBRATE.md).

Environment:
    ISIS_CORPUS   the corpus (default: sample_corpus/, which the judgments in data/ were made on)
    ISIS_MODEL    a Hugging Face model with ONNX weights (default: bge-micro-v2). Another model gets its
                  own index in calibration/.cache/, so the app's index in <corpus>/.isis is left alone.

Overrides are written as on the page and in the notebook: `--set z_strong=3 --set mmr_lambda.core=0.8`
fixes values, and `--grid z_strong=2.5,3,3.5` tries each value (several grids multiply).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isis.engine.index import Engine  # noqa: E402
from isis.engine.params import configure, specs  # noqa: E402
from isis.engine.related import RelatedConfig  # noqa: E402
from isis.engine.search import SearchConfig  # noqa: E402
from isis.engine.semantic import Embedder  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
CACHE = Path(__file__).resolve().parent / ".cache"
CORPUS = Path(os.environ.get("ISIS_CORPUS", ROOT / "sample_corpus"))
MODEL = os.environ.get("ISIS_MODEL", "")

# Utility of one Related suggestion by its grade: relevant +1, partly relevant +0.5, irrelevant -1.
UTILITY = {2: 1.0, 1: 0.5, 0: -1.0}


def load_engine() -> Engine:
    started = time.time()
    index_dir = CACHE / re.sub(r"[^\w.-]+", "_", MODEL) if MODEL else None
    engine = Engine(CORPUS, embedder=Embedder(MODEL) if MODEL else Embedder(), index_dir=index_dir,
                    exclude=(CORPUS / "_tasks",))
    engine.refresh(on_progress=lambda s: print(f"  {s.state}: {s.embedded}/{s.to_embed} passages embedded", file=sys.stderr))
    print(f"engine ready in {time.time() - started:.0f}s: {len(engine.snapshot)} passages, "
          f"model {getattr(engine.embedder, 'model_id', '?')}", file=sys.stderr)
    return engine


def key(path: str, start_line: int) -> str:
    """How the judgments name a passage. Changing the chunking (target_words) invalidates them."""
    return f"{path}:{start_line}"


def jload(name: str, default):
    path = DATA / name
    return json.loads(path.read_text()) if path.exists() else default


def jsave(name: str, obj) -> None:
    path = DATA / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(path)


def _value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="fix a parameter")
    parser.add_argument("--grid", action="append", default=[], metavar="NAME=V1,V2,...", help="try each value")


def overrides(args, cls) -> list[dict]:
    """Every combination of the --grid values, each with the --set values applied."""
    known = {s["name"] for s in specs(cls)}
    fixed = {}
    for item in args.set:
        name, _, value = item.partition("=")
        fixed[name] = _value(value)
    grids = []
    for item in args.grid:
        name, _, values = item.partition("=")
        grids.append([(name, _value(v)) for v in values.split(",")])
    for name in [*fixed, *(g[0][0] for g in grids)]:
        if name not in known:
            sys.exit(f"unknown parameter {name!r}; known: {', '.join(sorted(known))}")
    return [{**fixed, **dict(combo)} for combo in itertools.product(*grids)] or [fixed]


def label(over: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in over.items()) or "(defaults)"


def search_config(over: dict) -> SearchConfig:
    return configure(SearchConfig, over)


def related_config(over: dict) -> RelatedConfig:
    return configure(RelatedConfig, over)
