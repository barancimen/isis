"""`isis index` builds or updates the index, `isis serve` runs the web app, `isis model` changes a
corpus's embedding models while isis is stopped, and `isis search` and `isis related` ask a running
server."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

EVERY_INTERFACE = ("0.0.0.0", "::", "")
_noted: set[Path] = set()  # archives already said to have no manifest


def folders_of(corpus: list[str], osiris: list[str] = ()) -> list[Path]:
    """The corpus folders: those named, and then any osiris archive not among them."""
    folders = [Path(d).resolve() for d in corpus]
    for d in folders:
        if not d.is_dir():
            sys.exit(f"isis: not a folder: {d}")
    return list(dict.fromkeys([*folders, *archives_of(osiris)]))


def archives_of(osiris: list[str]) -> dict[Path, Path | None]:
    """The osiris archives given with --osiris: each one's folder of documents, and its manifest."""
    from .engine.osiris import MANIFEST, archive_folders

    found = {}
    for d in osiris:
        if not Path(d).is_dir():
            sys.exit(f"isis: not a folder: {d}")
        md, manifest = archive_folders(Path(d))
        if manifest is None and md not in _noted:  # once, however often the folders are worked out
            _noted.add(md)
            print(f"isis: no {MANIFEST} next to {md}: each document's kind comes from its source alone, "
                  "so a page osiris fetched again another way counts as a web page", flush=True)
        found[md] = manifest
    return found


def cmd_index(args) -> None:
    from .engine.index import Engine
    from .engine.semantic import Embedder

    from .engine.index import corpus_exclude

    folders = folders_of(args.corpus, args.osiris)
    tasks_dir = Path(args.tasks).resolve() if args.tasks else folders[0] / "_tasks"
    embedders = [Embedder(m) for m in models(args, folders)[0]]
    engine = Engine(folders, embedder=embedders, exclude=corpus_exclude(folders, tasks_dir), osiris=archives_of(args.osiris))
    engine.defer_archives = False  # a single run: save everything, whatever osiris is doing
    engine.refresh()  # it reports its progress


def models(args, folders: list[Path]) -> tuple[list[str], list[str]]:
    """The models to load, the default one first, and the corpus's other models, which the page
    shows greyed out. Without options the corpus's active model alone is loaded. --model names the
    ones to load instead, and --all-models adds every model of the corpus's list, after those named
    and the active one. A model that isn't loaded keeps its embeddings on disk as they are."""
    if args.no_model:
        return [], []
    from .engine.models import ModelError, ModelList

    try:
        listed = ModelList.of(folders[0])
    except ModelError as e:
        sys.exit(f"isis: {e}")
    load = [listed.find(m) or m for m in args.model or []]  # as the list spells it, so its embeddings are found
    if args.all_models or not load:
        load += [m for m in [listed.active] if m]
    if args.all_models:
        load += listed.models
    load = list(dict.fromkeys(load))
    return load, [m for m in listed.models if m not in load]


RESTART = "Start isis again to use it."
TRIAL = "The sperm whale dives deep for squid."


def _size(n: int) -> str:
    if n < 1e6:
        return f"{max(1, round(n / 1e3))} KB"
    return f"{n / 1e6:.1f} MB" if n < 1e9 else f"{n / 1e9:.2f} GB"


def check_model(repo: str) -> None:
    """Loads a model and embeds a sentence with it, so that one isis can't use is refused before
    it is made active. Loading it also downloads its files, which a server kept off the network
    (HF_HUB_OFFLINE=1) can't do when it starts."""
    import numpy as np
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError

    from .engine import semantic
    from .engine.models import ModelError

    try:
        semantic.fetch_settings(repo)  # before it is loaded, so a file the network kept back isn't taken as missing
        embedder = semantic.Embedder(repo)
        vector = embedder.embed([TRIAL])
        if vector.shape != (1, embedder.dim) or not np.isfinite(vector).all() or abs(np.linalg.norm(vector) - 1) > 1e-3:
            raise ValueError("it doesn't turn a sentence into one vector")
    except Exception as e:
        first = next((line.strip() for line in str(e).splitlines() if line.strip()), type(e).__name__)
        if isinstance(e, GatedRepoError):
            why = "Hugging Face gives its files only to accounts that were granted access"
        elif isinstance(e, RepositoryNotFoundError):
            why = "Hugging Face has no repository by that name, or it is private"
        elif isinstance(e, LocalEntryNotFoundError):
            why = "its files aren't downloaded yet and Hugging Face can't be reached"
        elif isinstance(e, FileNotFoundError):  # as Embedder._fetch says it
            why = f"{e}, and isis needs a tokenizer.json and ONNX weights in onnx/model.onnx"
        elif isinstance(e, HfHubHTTPError):  # "500 Server Error. (Request ID: ...)"
            why = f"Hugging Face answered {re.sub(r'[.]? *[(]Request ID.*', '', first)}"
        else:
            why = first
        raise ModelError(f"cannot use {repo}: {why.rstrip('.')}") from None


def embeddings_of(folders: list[Path], model: str) -> list[Path]:
    """The files that hold a model's embeddings, in every folder of the corpus."""
    from .engine.index import saved_vectors
    from .engine.models import made_by

    return [path for folder in folders for name, files in saved_vectors(folder / ".isis").items()
            if made_by(name, model) for path in files]


def cmd_model(args) -> None:
    """`isis model`: the corpus's embedding models. It changes files only, so isis is stopped
    first and started again afterwards, by hand."""
    from .engine.models import ModelError, ModelList, repo_id

    folders = folders_of(args.corpus, args.osiris)
    try:
        listed = ModelList.of(folders[0])
        if args.action == "list":
            list_models(listed, folders)
        elif args.action == "add":
            add_model(listed, repo_id(args.repo), active=not args.inactive)
        elif args.action == "use":
            use_model(listed, repo_id(args.repo))
        else:
            remove_model(listed, folders, repo_id(args.repo), repo_id(args.next) if args.next else None)
    except ModelError as e:
        sys.exit(f"isis: {e}")
    except OSError as e:
        sys.exit(f"isis: cannot change the models of {folders[0]}: {e.strerror or e}"
                 + (f", {e.filename}" if e.filename else ""))


def list_models(listed, folders: list[Path]) -> None:
    from .engine.index import saved_vectors
    from .engine.models import made_by

    if not listed.models:
        print("This corpus has no embedding model, so isis searches it by keywords only.")
    width = max(map(len, listed.models), default=0)
    for model in listed.models:
        size = sum(path.stat().st_size for path in embeddings_of(folders, model))
        print(f"{model:<{width}}  {'active' if model == listed.active else '':<6}  "
              f"{f'embeddings {_size(size)}' if size else 'no embeddings yet'}")
    others: dict[str, int] = {}  # embeddings of models that are not on the list
    for folder in folders:
        for name, files in saved_vectors(folder / ".isis").items():
            if not any(made_by(name, model) for model in listed.models):
                model = name.partition(":")[0]
                others[model] = others.get(model, 0) + sum(path.stat().st_size for path in files)
    for model, size in others.items():
        print(f"{model} is not one of the models. isis keeps its {_size(size)} of embeddings, and uses them again "
              "if the model is added.")


def add_model(listed, repo: str, active: bool) -> None:
    known, before = listed.find(repo), listed.active
    name = known or repo
    check_model(name)
    listed.add(name, active)
    listed.save()
    if name == before:
        print(f"{name} is already the active model.")
    elif listed.active == before:
        print(f"{name} is already one of the models." if known else f"Added {name}.", f"{before} stays the active model.")
    elif known:
        print(f"{name} was already one of the models. It is now the active one. {RESTART}")
    else:
        print(f"Added {name} and made it the active model. When isis starts again, it embeds the passages "
              "this model hasn't embedded yet.")


def use_model(listed, repo: str) -> None:
    before = listed.active
    name = listed.use(repo)
    if name == before:
        print(f"{name} is already the active model.")
        return
    check_model(name)
    listed.save()
    print(f"{name} is now the active model. {RESTART}")


def remove_model(listed, folders: list[Path], repo: str, then: str | None) -> None:
    """Takes a model off the list. Nothing is deleted: its embeddings stay where they are, and are
    used again if the model is added back."""
    from .engine.models import ModelError

    name, before = listed.find(repo), listed.active
    if name is None:
        raise ModelError(f"{repo} is not one of this corpus's models. {listed.has()}")
    listed.remove(name, then)
    if listed.active and listed.active != before:
        check_model(listed.active)
    listed.save()
    size = sum(path.stat().st_size for path in embeddings_of(folders, name))
    print(f"Removed {name}. isis keeps its {_size(size)} of embeddings, and uses them again if the model is added back."
          if size else f"Removed {name}.")
    if listed.active != before:
        print(f"{listed.active} is now the active model. {RESTART}" if listed.active else
              "No model is left, so isis now searches by keywords only.")


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
    if any(d == tasks or d in tasks.parents or tasks in d.parents for d in folders_of(args.corpus, args.osiris)):
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
    folders = folders_of(args.corpus, args.osiris)
    # As JSON: a folder's name may hold the ":" or "|" that would otherwise separate them.
    os.environ["ISIS_CORPUS"] = json.dumps(list(map(str, folders)))
    os.environ["ISIS_OSIRIS"] = json.dumps({str(md): str(manifest or "") for md, manifest in archives_of(args.osiris).items()})
    if args.tasks:
        os.environ["ISIS_TASKS"] = str(Path(args.tasks).resolve())
    load, others = models(args, folders)
    os.environ["ISIS_MODELS"] = ",".join(load)
    os.environ["ISIS_OTHER_MODELS"] = ",".join(others)  # listed for the corpus and not loaded: greyed out in the page
    os.environ["ISIS_HOSTS"] = ",".join(allowed_hosts(args.host, args.allow_host))
    os.environ["ISIS_BUILD_NATIVE"] = "0" if args.no_build else "1"
    if args.llm_key_file:
        os.environ["ISIS_LLM_KEY_FILE"] = str(Path(args.llm_key_file).resolve())
    package = Path(__file__).parent
    host = f"[{args.host}]" if ":" in args.host else args.host  # an IPv6 address goes in brackets
    url = f"http://{host}:{args.port}{args.base_path and args.base_path + '/'}"
    if sys.stderr.isatty():
        os.environ["ISIS_URL"] = url  # the server shows it at the bottom, with its memory and disk use
    else:
        print(f"isis: {url}", flush=True)
    uvicorn.run(
        "isis.server.app:app_from_env",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(package)] if args.reload else None,
        log_level="warning",
        log_config=None,  # its messages go where isis's own do, above the line at the bottom
    )


CLIENT_COMMANDS = ("search", "related")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in CLIENT_COMMANDS:  # before the engine is imported, so they start at once
        from .client import main as client_main

        sys.exit(client_main(argv))
    from .engine.index import configure_logging
    from .engine.semantic import MODELS

    configure_logging()
    parser = argparse.ArgumentParser(prog="isis", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def folders(p):
        p.add_argument("corpus", nargs="*", help="one or more folders of markdown files")
        p.add_argument("--osiris", action="append", default=[], metavar="DIR",
                       help="an archive made by osiris (its data folder, or the md folder in it), whose sources "
                            "can each be left out of Search and Related; it joins the corpus if not named among its "
                            "folders (repeatable)")

    def common(p):
        folders(p)
        p.add_argument("--tasks", help="folder for task documents (default: _tasks in the first folder)")
        p.add_argument("--model", action="append", metavar="REPO",
                       help="Load this embedding model for this run, in place of the corpus's active one. It can be "
                            "repeated to load several, and the first one named is then the default. A corpus's "
                            f"active model is {MODELS[0]} until isis model changes it.")
        p.add_argument("--all-models", action="store_true",
                       help="Load every model of the corpus's list, the active one first. The models named with "
                            "--model come before it.")
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
    p.add_argument("--llm-key-file", metavar="FILE",
                   help="the DeepSeek API key for LLM help with writing queries (default: DEEPSEEK_API_KEY, "
                        "or .deepseek_api_key in the isis folder; without one, LLM help is off)")
    p.add_argument("--no-build", action="store_true",
                   help="don't build the Rust extension when it is missing or older than native/")
    p.add_argument("--public", action="store_true",
                   help="serve the public behind a proxy: private tasks per visitor, and limits on requests")
    p.add_argument("--base-path", default="", metavar="PATH",
                   help="with --public: where the site lives on its domain, such as /isis")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("model", help="list, add or remove the corpus's embedding models, or choose the active one",
                       description="The embedding models of a corpus. isis loads the active one, unless it is started "
                                   "with --model or --all-models. These commands change the list in the corpus's .isis "
                                   "folder and never delete embeddings. Stop isis first and start it again afterwards. "
                                   "Name the same folders as for isis serve.")
    actions = p.add_subparsers(dest="action", required=True)
    a = actions.add_parser("list", help="the corpus's models, with the active one marked")
    folders(a)
    a = actions.add_parser("add", help="add a model from Hugging Face and make it the active one")
    a.add_argument("repo", metavar="REPO", help="its Hugging Face repository, such as BAAI/bge-small-en-v1.5, or "
                                                "the address of its page. It needs ONNX weights in onnx/model.onnx "
                                                "and a tokenizer.json")
    folders(a)
    a.add_argument("--inactive", action="store_true", help="leave the active model as it is")
    a = actions.add_parser("use", help="make one of the corpus's models the active one")
    a.add_argument("repo", metavar="REPO", help="the model, as isis model list names it")
    folders(a)
    a = actions.add_parser("remove", help="take a model off the list, keeping its embeddings on disk")
    a.add_argument("repo", metavar="REPO", help="the model, as isis model list names it")
    folders(a)
    a.add_argument("--next", metavar="REPO",
                   help="the model to be active afterwards, which has to be named when the active one is removed")
    p.set_defaults(func=cmd_model)

    # Listed for --help only: main() hands them to client.py before this parser exists.
    sub.add_parser("search", help="search the corpus of a running server (isis search -h)", add_help=False)
    sub.add_parser("related", help="passages related to a task document, from a running server (isis related -h)",
                   add_help=False)

    args = parser.parse_args(argv)
    if not args.corpus and not args.osiris:
        parser.error("name a folder of markdown files, or an osiris archive with --osiris")
    if args.osiris:  # each archive is served as its folder of documents
        from .engine.osiris import archive_folders

        archived = {archive_folders(Path(d))[0]: d for d in args.osiris if Path(d).is_dir()}
        args.corpus = [str(next((md for md, given in archived.items() if Path(given).resolve() == Path(d).resolve()), d))
                       for d in args.corpus]
    args.func(args)
