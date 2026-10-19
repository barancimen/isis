"""Command line: `isis index` builds or updates the index, `isis serve` runs the web app."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from .engine.semantic import DEFAULT_MODEL


def make_engine(corpus: str, tasks: str | None, model: str | None):
    from .engine.index import Engine
    from .engine.semantic import Embedder

    root = Path(corpus).resolve()
    tasks_dir = Path(tasks).resolve() if tasks else root / "_tasks"
    embedder = Embedder(model) if model else None
    return Engine(root, embedder=embedder, exclude=(tasks_dir,)), tasks_dir


def cmd_index(args) -> None:
    engine, _ = make_engine(args.corpus, args.tasks, None if args.no_model else args.model)
    last = [0.0]

    def progress(status):
        now = time.monotonic()
        if status.state == "embedding" and now - last[0] > 5:
            last[0] = now
            print(f"  embedding {status.embedded:,}/{status.to_embed:,} passages", flush=True)

    print(f"indexing {engine.root}", flush=True)
    stats = engine.refresh(progress)
    print(
        f"{stats['files']:,} files, {engine.status.passages:,} passages "
        f"({stats['changed']} changed, {stats['removed']} removed, {stats['embedded']:,} embedded) "
        f"in {stats['seconds']}s"
    )


def cmd_serve(args) -> None:
    import uvicorn

    os.environ["ISIS_CORPUS"] = str(Path(args.corpus).resolve())
    if args.tasks:
        os.environ["ISIS_TASKS"] = str(Path(args.tasks).resolve())
    os.environ["ISIS_MODEL"] = "" if args.no_model else args.model
    package = Path(__file__).parent
    print(f"isis: http://{args.host}:{args.port}", flush=True)
    uvicorn.run(
        "isis.server.app:app_from_env",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(package)] if args.reload else None,
        log_level="warning",
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="isis", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("corpus", help="folder of markdown files")
        p.add_argument("--tasks", help="folder for task documents (default: <corpus>/_tasks)")
        p.add_argument("--model", default=DEFAULT_MODEL, help=f"embedding model (default: {DEFAULT_MODEL})")
        p.add_argument("--no-model", action="store_true", help="lexical search only, no embeddings")

    p = sub.add_parser("index", help="build or update the search index")
    common(p)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("serve", help="run the web app")
    common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="restart when the Python code changes")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
