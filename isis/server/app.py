"""The web app: a JSON API over the engine and the task store, and the static page in web/."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import re
import threading
import time
import zlib
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Literal

import numpy as np
import pydantic_core
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..engine import native
from ..engine.index import MARKDOWN, NO_EMBED, Engine, configure_logging, corpus_exclude
from ..engine.params import configure, specs
from ..engine.related import PRESETS as RELATED_PRESETS
from ..engine.related import LANES, RelatedConfig, TaskPrefs
from ..engine.search import PRESETS as SEARCH_PRESETS
from ..engine.search import QueryError, SearchConfig, correct_query, pattern_time_limit
from .llm import QuerySynth, Refused, load_key
from .public import PATTERN_SECONDS, PublicGuard, Workspaces
from .tasks import Conflict, TaskError, TaskStore

log = logging.getLogger(__name__)
WEB = Path(__file__).resolve().parents[2] / "web"
LOOPBACK = ("localhost", "127.0.0.1", "::1")
WATCH_RETRY = 30  # seconds
CATCH_UP_EVERY = 30  # seconds between checks whether osiris is done with an archive
REFRESH_EVERY = 60  # seconds: the index takes up changed files at most this often
PAGE_SETTLE_SECONDS = 5
PAGE_VERSION_SLOT = '<meta name="page-version" content="">'
# Seeds the draw that shares Related's results between its groups. It is the same for every task
# and for `isis related`, so the same text and settings give the same results in the page and on
# the command line. Feedback still moves it, as the draw adds the number of feedback events.
RELATED_SEED = 0
PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


class PageVersion:
    """Changes when the page's files do, but only once they have been still for a few seconds:
    an update saved file by file is not announced halfway through."""

    def __init__(self, folder: Path = WEB, settle: float = PAGE_SETTLE_SECONDS):
        self.folder, self.settle = folder, settle
        self.value = None

    def current(self) -> int | None:
        try:
            found = sorted((str(p), p.stat()) for p in self.folder.rglob("*") if p.suffix in (".html", ".js", ".css"))
        except OSError:  # a file replaced just now
            return self.value
        newest = max((stat.st_mtime for _, stat in found), default=0)
        if self.value is None or time.time() - newest >= self.settle:
            self.value = zlib.crc32(repr([(name, s.st_mtime_ns, s.st_size) for name, s in found]).encode())
        return self.value


class PageFiles(StaticFiles):
    """The page's files, which the browser may not keep. If it could, a tab reopened or reached
    through history would run its old copy without asking, whatever else the headers say. The page
    itself carries the version of these files, which is how a tab that stays open knows it is old."""

    def __init__(self, *, version: PageVersion, **options):
        super().__init__(**options)
        self.version = version

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if path in (".", "index.html") and response.status_code in (200, 304):
            current = self.version.current()
            page = (WEB / "index.html").read_text(encoding="utf-8")
            response = HTMLResponse(page.replace(PAGE_VERSION_SLOT, f'{PAGE_VERSION_SLOT[:-2]}{"" if current is None else current}">'))
        response.headers["Cache-Control"] = "no-store"
        return response


def plain(x):
    """Dataclasses, tuples and numpy scalars -> JSON-ready Python values."""
    if dataclasses.is_dataclass(x):
        return {f.name: plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [plain(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _plain_scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(f"{type(x).__name__} can't go into JSON")


def as_json(data) -> Response:
    """`data` as a JSON answer, encoded in one pass by pydantic_core (Rust), with the same result
    as plain(): dataclasses as objects, tuples as lists, numpy scalars as numbers, and null for a
    number that isn't finite. For the large answers, Search and Related: encoding them in Python
    (plain(), then FastAPI's own pass, then json) took three times as long as ranking them."""
    return Response(pydantic_core.to_json(data, inf_nan_mode="null", fallback=_plain_scalar), media_type="application/json")


def whole(value):
    """`value` with any half of a surrogate pair in its strings replaced by U+FFFD. JSON may hold
    one ("\\ud800"), and it can neither be saved as UTF-8 nor sent back in an answer."""
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return value.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
        return value
    if isinstance(value, dict):
        return {whole(k): whole(v) for k, v in value.items()}
    if isinstance(value, list):
        return [whole(v) for v in value]
    return value


class Body(BaseModel):
    """What a request sends as JSON."""

    @model_validator(mode="before")
    @classmethod
    def _whole_strings(cls, data):
        return whole(data)


class NewTask(Body):
    name: str


class SaveTask(Body):
    content: str
    base_mtime: str | None = None


class RenameTask(Body):
    new_name: str


class MoveTask(Body):
    before: str | None = None  # the task to go in front of; None for the end of the list


class RelatedRequest(Body):
    text: str
    params: dict = {}  # overrides of RelatedConfig, as named in /api/params
    skip_folders: list[str] = []  # corpus folders to leave out, as named in /api/status
    model: str = ""  # the embedding model, as named in /api/status; the default if empty or unknown


class StatelessRelatedRequest(RelatedRequest):
    steer: str = ""  # word rules, as on the steer line of a task: x prefer, +x require, -x never show


class TermRequest(Body):
    term: str  # boost and exclude: the word as shown on its chip; unboost, ignore, unignore: its term
    action: Literal["boost", "unboost", "ignore", "unignore", "exclude"]


class SteerRequest(Body):
    text: str  # clauses as in a search query: x prefer, +x require, -x never show


class QueryRequest(Body):
    target: Literal["search", "related"]  # the box the query is for
    request: str  # what the person wants, in English
    current: str = ""  # what the box holds now


class SettingsRequest(Body):
    steer: str = ""
    ignored: list[str] = []  # terms
    excluded_docs: list[str] = []


class FeedbackRequest(Body):
    action: Literal["add", "dismiss", "avoid", "restore", "like", "unlike", "exclude_doc", "include_doc"]
    lane: str = ""  # where the passage was shown; restoring finds it from the dismissal
    path: str = ""
    text: str = ""
    id: str = ""  # restore, unlike: the passage's id from the preferences the page was given
    start_line: int | None = None
    end_line: int | None = None


def create_app(engine: Engine, tasks: TaskStore | Workspaces, *, background: bool = True, watch: bool = True,
               allowed_hosts: list[str] | None = None, build_native: bool = False,
               llm: QuerySynth | None = None, other_models: list[str] = ()) -> FastAPI:
    """`allowed_hosts`: the Host headers the API answers (None: any). A page on another site can
    point its own domain at 127.0.0.1 (DNS rebinding); checking the Host header stops it from
    reading the corpus or overwriting tasks through the browser.

    `other_models`: the models on the corpus's list that the engine wasn't given, by id. The page
    shows them greyed out in its menu of models.

    `tasks` as Workspaces serves the public: each visitor gets private tasks, and requests are
    limited (see public.py)."""
    public = isinstance(tasks, Workspaces)
    time_limit = (lambda: pattern_time_limit(PATTERN_SECONDS)) if public else nullcontext

    def refresh(only: set[str] | None = None) -> None:
        try:
            engine.refresh(only=only)  # it reports what changed
        except Exception:
            log.exception("index refresh failed")

    def watch_corpus(stop: threading.Event) -> None:
        from watchfiles import watch as watch_files

        tasks_dir = tasks.folder.resolve()
        roots = list(engine.roots.values())

        def relevant(_, path: str) -> bool:
            p = Path(path)
            root = next((r for r in roots if r == p or r in p.parents), None)
            if root is None:
                return False
            if p.name == NO_EMBED:  # a hidden file, but it changes what gets embedded
                return True
            if any(part.startswith(".") for part in p.relative_to(root).parts):
                return False
            if any(x == p or x in p.parents for x in (tasks_dir, *engine.exclude)):
                return False
            # Besides markdown files: folders and links, which may bring many files at once (a
            # moved folder is one event), and deleted paths, which may have been either.
            return p.suffix.lower() in MARKDOWN or not p.is_file()

        def folder_of(path: str) -> str | None:
            p = Path(path)
            return next((name for name, r in engine.roots.items() if r == p or r in p.parents), None)

        # Changes are taken up at most once every REFRESH_EVERY seconds: an editor that saves as you
        # type would otherwise have the whole index made again every second or two. The first change
        # after a quiet spell goes at once, and the folders without changes aren't looked at.
        said, waiting, last = "", set(), -REFRESH_EVERY
        while not stop.is_set():  # a watch covers the linked folders that exist when it starts
            links = engine.links
            try:
                for changes in watch_files(*roots, watch_filter=relevant, stop_event=stop, debounce=800,
                                           rust_timeout=1000, yield_on_timeout=True):
                    waiting |= {folder_of(path) for _, path in changes}
                    if not waiting or time.monotonic() - last < REFRESH_EVERY:
                        continue
                    only = None if None in waiting else set(waiting)  # None: a path outside the folders, look at all
                    waiting, last = set(), time.monotonic()
                    refresh(only)
                    if engine.links != links:
                        break
                else:
                    return
            except Exception as e:  # no inotify capacity left, a folder gone: look every so often instead
                if str(e) != said:
                    said = str(e)
                    log.warning("cannot watch the corpus for changes (%s): checking every %d s instead", e, WATCH_RETRY)
                if not stop.wait(WATCH_RETRY):
                    refresh()

    def catch_up(stop: threading.Event) -> None:
        """Takes up an osiris archive's changes, and saves its cache, once osiris is done writing to
        it or ARCHIVE_EVERY has passed: no file event says so (see Engine.waiting)."""
        while not stop.wait(CATCH_UP_EVERY):
            if engine.waiting():
                refresh()

    @asynccontextmanager
    async def lifespan(_):
        stop = threading.Event()
        if background:
            threading.Thread(target=refresh, daemon=True, name="isis-index").start()
        if watch:
            threading.Thread(target=watch_corpus, args=(stop,), daemon=True, name="isis-watch").start()
            if engine.osiris:
                threading.Thread(target=catch_up, args=(stop,), daemon=True, name="isis-catch-up").start()
        if public and background:
            threading.Thread(target=tasks.keep_clean, args=(stop,), daemon=True, name="isis-clean").start()
        if build_native:  # the Rust extension, when it is missing or outdated
            native.start_build()
        yield
        stop.set()

    # No /docs or /redoc: FastAPI's pages for them load scripts from a public CDN.
    app = FastAPI(title="isis", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    if public:
        app.add_middleware(PublicGuard, workspaces=tasks)
    if allowed_hosts is not None:  # added last, so it runs first
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @app.exception_handler(TaskError)
    async def task_error(_, e: TaskError):
        body = {"detail": str(e)}
        if isinstance(e, Conflict):
            body["mtime"] = e.mtime
        return JSONResponse(body, status_code=e.status)

    # ---- corpus ----

    page_version = PageVersion()

    @app.get("/api/status")
    def status():
        state = plain(engine.status)
        if public and state["error"]:
            state["error"] = "the index could not be updated"  # the full message can name server paths
        return {
            **state,
            "index_bytes": engine.index_bytes(),
            "folders": [name for name in engine.roots if name],  # when serving several folders
            "sources": engine.sources(),  # of osiris archives
            "model": getattr(engine.embedder, "model_id", None),  # the default
            "models": [
                {"id": e.model_id, "label": e.model_id,  # the organisation and the model
                 "unembedded": int((~engine.view(e).has_vector).sum()), "index_bytes": engine.index_bytes(e)}
                for e in engine.embedders
            ],
            "other_models": list(other_models),  # on the corpus's list, and not loaded
            "page_version": page_version.current(),
            "llm": bool(llm and llm.available),  # LLM help for writing queries
        }

    @app.get("/api/files")
    def files():
        return [{"path": p} for p in sorted(engine.snapshot.file_rows)]

    @app.get("/api/file")
    def file(path: str):
        if public and path not in engine.snapshot.file_rows:
            raise HTTPException(404, f"there is no file named {path}")  # only what search can show
        try:
            content = engine.resolve(path).read_text(encoding="utf-8-sig", errors="replace")  # as the index reads it
        except ValueError as e:
            raise HTTPException(400, str(e))
        except OSError:
            raise HTTPException(404, f"there is no file named {path}")
        return {"path": path, "content": content}

    @app.get("/api/params")
    def params():
        """The parameters the page may tune, with defaults, ranges and help, and the built-in presets."""
        return {"search": specs(SearchConfig), "related": specs(RelatedConfig),
                "presets": {"search": SEARCH_PRESETS, "related": RELATED_PRESETS}}

    @app.get("/api/search")
    def search(q: str = "", mode: Literal["lexical", "hybrid"] = "lexical", limit: int = 50, params: str = "", skip: str = "", hide: str = "", as_typed: bool = False, model: str = ""):
        try:
            overrides = json.loads(params) if params else {}
            skip_folders = json.loads(skip) if skip else []
            hidden_files = json.loads(hide) if hide else []
        except ValueError:
            raise HTTPException(400, "params, skip and hide must be JSON")
        if not isinstance(overrides, dict) or not isinstance(skip_folders, list) or not isinstance(hidden_files, list):
            raise HTTPException(400, "params must be a JSON object, skip a list of folders and hide a list of files")
        started = time.perf_counter()
        try:
            with time_limit():
                corrected = None if as_typed else correct_query(engine.snapshot, q)
                hits = engine.search(corrected or q, limit=max(1, min(limit, 200)), mode=mode, config=configure(SearchConfig, overrides),
                                     skip_folders=[str(f) for f in skip_folders],
                                     skip_files=[str(f) for f in hidden_files], model=model or None)
        except QueryError as e:
            raise HTTPException(400, str(e))
        return as_json({"hits": hits, "corrected": corrected, "ms": round((time.perf_counter() - started) * 1000, 1)})

    # ---- tasks ----

    @app.get("/api/tasks")
    def list_tasks():
        return tasks.list()

    @app.post("/api/tasks")
    def create_task(body: NewTask):
        return tasks.create(body.name)

    @app.get("/api/tasks/{name}")
    def read_task(name: str):
        return tasks.read(name)

    @app.put("/api/tasks/{name}")
    def save_task(name: str, body: SaveTask):
        return {"mtime": tasks.write(name, body.content, body.base_mtime)}

    @app.post("/api/tasks/{name}/rename")
    def rename_task(name: str, body: RenameTask):
        return tasks.rename(name, body.new_name)

    @app.post("/api/tasks/{name}/move")
    def move_task(name: str, body: MoveTask):
        return tasks.move(name, body.before)

    @app.delete("/api/tasks/{name}")
    def delete_task(name: str):
        tasks.delete(name)
        return {"ok": True}

    def run_related(body: RelatedRequest, prefs: TaskPrefs) -> dict:
        if public and len(PARAGRAPH_BREAK.findall(body.text)) >= tasks.limits.related_paragraphs:
            raise HTTPException(413, "this task has too many paragraphs for Related here "
                                     f"(the limit is {tasks.limits.related_paragraphs:,})")
        try:
            with time_limit():
                result = engine.related(
                    body.text,
                    prefs=prefs,
                    config=configure(RelatedConfig, body.params),
                    skip_folders=body.skip_folders,
                    seed=RELATED_SEED,
                    model=body.model or None,
                )
        except QueryError as e:  # a regex that ran out of time
            raise HTTPException(400, str(e))
        return {f.name: getattr(result, f.name) for f in dataclasses.fields(result)}

    @app.post("/api/tasks/{name}/related")
    def related(name: str, body: RelatedRequest):
        prefs = tasks.prefs(name)
        return as_json({**run_related(body, prefs), "prefs": prefs.summary()})

    @app.post("/api/related")
    def related_to_text(body: StatelessRelatedRequest):
        """Related for any text, with word rules and settings given in the request and nothing
        stored: what `isis related` asks for. The same text, rules and settings give what a task
        without other preferences shows in the page."""
        prefs = TaskPrefs()
        try:
            with time_limit():
                prefs.set_steer(body.steer)
        except QueryError as e:
            raise HTTPException(400, str(e))
        return as_json(run_related(body, prefs))

    @app.get("/api/tasks/{name}/prefs")
    def get_prefs(name: str):
        return tasks.prefs(name).summary()

    @app.delete("/api/tasks/{name}/prefs")
    def forget_prefs(name: str):
        tasks.forget_prefs(name)
        return tasks.prefs(name).summary()

    @app.put("/api/tasks/{name}/steer")
    def steer(name: str, body: SteerRequest):
        try:
            with time_limit(), tasks.edit_prefs(name) as prefs:
                prefs.set_steer(body.text)
        except QueryError as e:
            raise HTTPException(400, str(e))
        return prefs.summary()

    @app.put("/api/tasks/{name}/settings")
    def settings(name: str, body: SettingsRequest):
        """A preset's settings, all or nothing: a bad regex changes none of them. What the task
        remembers about single passages stays as it is."""
        try:
            with time_limit(), tasks.edit_prefs(name) as prefs:
                prefs.set_steer(body.steer)
                prefs.ignored = [term for term in dict.fromkeys(body.ignored) if term.strip()]
                prefs.excluded_docs = list(dict.fromkeys(body.excluded_docs))
        except QueryError as e:
            raise HTTPException(400, str(e))
        return prefs.summary()

    @app.post("/api/tasks/{name}/terms")
    def terms(name: str, body: TermRequest):
        """Chip shortcuts: they edit the steer line (boost, unboost, exclude) or the ignored list."""
        with tasks.edit_prefs(name) as prefs:
            getattr(prefs, body.action)(body.term)
        return prefs.summary()

    @app.post("/api/tasks/{name}/feedback")
    def feedback(name: str, body: FeedbackRequest):
        if body.action not in ("restore", "unlike", "exclude_doc", "include_doc") and body.lane not in LANES:
            raise HTTPException(400, f"feedback needs the group the passage was shown in: {', '.join(LANES)}")
        with tasks.edit_prefs(name) as prefs:
            if body.action == "exclude_doc":
                prefs.exclude_doc(body.path)
            elif body.action == "include_doc":
                prefs.include_doc(body.path)
            elif body.action == "like":
                prefs.like(body.lane, body.path, body.text, body.start_line, body.end_line)
            elif body.action == "unlike":
                prefs.unlike(body.id)
            elif body.action == "restore" and body.id:
                prefs.restore_id(body.id)
            elif body.action == "restore":
                prefs.restore(body.path, body.text)
            elif body.action in ("dismiss", "avoid"):
                prefs.dismiss(body.lane, body.path, body.text, body.start_line, body.end_line,
                              avoid=body.action == "avoid")
            else:
                prefs.record(body.lane, added=True)
        return prefs.summary()

    @app.post("/api/llm/query")
    def synthesize(body: QueryRequest):
        """A query for the search box or the word rules, written by an LLM from what the person
        wants in English (see llm.py)."""
        try:
            if llm is None:
                raise Refused(503, "LLM help is off on this server: it has no DeepSeek key")
            return llm.synthesize(body.target, body.request, body.current)
        except Refused as e:
            headers = {"Retry-After": str(max(1, round(e.retry_after)))} if e.retry_after else None
            raise HTTPException(e.status, str(e), headers=headers) from None

    app.mount("/", PageFiles(version=page_version, directory=WEB, html=True), name="web")
    return app


def app_from_env() -> FastAPI:
    """The app `isis serve` runs, configured through ISIS_CORPUS (the folders, as a JSON list, or
    separated like PATH), ISIS_OSIRIS (those of the folders that are osiris archives, as a JSON
    object of folder to manifest, or each as folder and manifest separated by "|"; the manifest
    is empty if there is none), ISIS_TASKS, ISIS_MODELS (comma-separated, the default first; empty: keywords only),
    ISIS_OTHER_MODELS (comma-separated: the corpus's models that aren't loaded, which the page shows
    greyed out), ISIS_HOSTS (comma-separated Host names to answer, or "*"; default:
    this machine only), ISIS_PUBLIC ("1": serve the public, see public.py), ISIS_BASE_PATH (with
    ISIS_PUBLIC: where the site lives on its domain, such as /isis) and ISIS_BUILD_NATIVE ("1": build
    the Rust extension when it is missing or outdated, see native.start_build), ISIS_LLM_KEY_FILE
    (the file with the DeepSeek key for LLM help, see llm.load_key) and ISIS_URL (when
    set, the address to show at the bottom of the terminal, see terminal.py)."""
    from ..engine.semantic import Embedder

    configure_logging()
    if url := os.environ.get("ISIS_URL"):
        from .terminal import start

        start(url)
    corpus = os.environ["ISIS_CORPUS"]
    folders = [Path(d) for d in (json.loads(corpus) if corpus.startswith("[") else corpus.split(os.pathsep))]
    tasks_dir = Path(os.environ.get("ISIS_TASKS") or folders[0] / "_tasks")
    models = [m for m in os.environ.get("ISIS_MODELS", "").split(",") if m]
    archives = os.environ.get("ISIS_OSIRIS", "")
    osiris = json.loads(archives) if archives.startswith("{") else dict(
        entry.partition("|")[::2] for entry in archives.split(os.pathsep) if entry)
    engine = Engine(folders, embedder=[Embedder(m) for m in models], exclude=corpus_exclude(folders, tasks_dir),
                    osiris={folder: manifest or None for folder, manifest in osiris.items()})
    hosts = [h for h in os.environ.get("ISIS_HOSTS", ",".join(LOOPBACK)).split(",") if h]
    store = (Workspaces(tasks_dir, base_path=os.environ.get("ISIS_BASE_PATH", ""))
             if os.environ.get("ISIS_PUBLIC") == "1" else TaskStore(tasks_dir))
    key = load_key(os.environ.get("ISIS_LLM_KEY_FILE") or None)
    others = [m for m in os.environ.get("ISIS_OTHER_MODELS", "").split(",") if m and m not in models]
    return create_app(engine, store, allowed_hosts=hosts, build_native=os.environ.get("ISIS_BUILD_NATIVE") == "1",
                      llm=QuerySynth(key) if key else None, other_models=others)
