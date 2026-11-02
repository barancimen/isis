"""Command line: `isis index` builds or updates the index, `isis serve` runs the web app."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from .engine.index import configure_logging, corpus_exclude
from .engine.semantic import DEFAULT_MODEL

EVERY_INTERFACE = ("0.0.0.0", "::", "")


def folders_of(corpus: list[str]) -> list[Path]:
    folders = [Path(d).resolve() for d in corpus]
    for d in folders:
        if not d.is_dir():
            sys.exit(f"isis: not a folder: {d}")
    return folders


def cmd_index(args) -> None:
    from .engine.index import Engine
    from .engine.semantic import Embedder

    folders = folders_of(args.corpus)
    tasks_dir = Path(args.tasks).resolve() if args.tasks else folders[0] / "_tasks"
    embedder = Embedder(args.model) if args.model and not args.no_model else None
    engine = Engine(folders, embedder=embedder, exclude=corpus_exclude(folders, tasks_dir))
    engine.refresh()  # it reports its progress


def allowed_hosts(host: str, extra: list[str]) -> list[str]:
    """Host headers the server answers: this machine, the address it listens on, and --allow-host.
    Listening on every interface without --allow-host answers any name, as before."""
    every_interface = host in EVERY_INTERFACE
    if every_interface and not extra:
        print("isis: listening on every interface; add --allow-host NAME to answer only that name", flush=True)
        return ["*"]
    from .server.app import LOOPBACK

    return list(dict.fromkeys([*LOOPBACK, *([] if every_interface else [host]), *extra]))


def check_public(args) -> None:
    """Public mode trusts the visitor address that the proxy in front of it reports, and keeps
    visitors' tasks apart. These settings would undo that."""
    from .server.app import LOOPBACK

    if args.host not in LOOPBACK:
        sys.exit("isis: --public needs a loopback --host: only the proxy in front may reach the server")
    if not args.allow_host:
        sys.exit("isis: --public needs --allow-host NAME, the public name of the site")
    if not args.tasks:
        sys.exit("isis: --public needs --tasks FOLDER, outside the corpus")
    tasks = Path(args.tasks).resolve()
    if any(d == tasks or d in tasks.parents or tasks in d.parents for d in folders_of(args.corpus)):
        sys.exit("isis: --public needs a --tasks folder outside the corpus")
    if args.base_path and not re.fullmatch(r"(/[A-Za-z0-9._~-]+)+", args.base_path):
        sys.exit("isis: --base-path looks like /isis, with no final slash")
    try:
        import regex  # noqa: F401
    except ImportError:
        sys.exit('isis: --public needs the regex library: pip install -e ".[public]"')


def cmd_serve(args) -> None:
    import uvicorn

    if args.base_path and not args.public:
        sys.exit("isis: --base-path works only with --public")
    if args.public:
        check_public(args)
        os.environ["ISIS_PUBLIC"] = "1"
        os.environ["ISIS_BASE_PATH"] = args.base_path
    os.environ["ISIS_CORPUS"] = os.pathsep.join(map(str, folders_of(args.corpus)))
    if args.tasks:
        os.environ["ISIS_TASKS"] = str(Path(args.tasks).resolve())
    os.environ["ISIS_MODEL"] = "" if args.no_model else args.model
    os.environ["ISIS_HOSTS"] = ",".join(allowed_hosts(args.host, args.allow_host))
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
    configure_logging()
    parser = argparse.ArgumentParser(prog="isis", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("corpus", nargs="+", help="one or more folders of markdown files")
        p.add_argument("--tasks", help="folder for task documents (default: _tasks in the first folder)")
        p.add_argument("--model", default=DEFAULT_MODEL, help=f"embedding model (default: {DEFAULT_MODEL})")
        p.add_argument("--no-model", action="store_true", help="lexical search only, no embeddings")

    p = sub.add_parser("index", help="build or update the search index")
    common(p)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("serve", help="run the web app")
    common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                   help="also answer requests addressed to this host name or IP (repeatable)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="restart when the Python code changes")
    p.add_argument("--public", action="store_true",
                   help="serve the public behind a proxy: private tasks per visitor, and limits on requests")
    p.add_argument("--base-path", default="", metavar="PATH",
                   help="with --public: where the site lives on its domain, such as /isis")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)
