"""The web app: a JSON API over the engine and the task store, and the static page in web/."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import threading
import time
import zlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

from ..engine.index import MARKDOWN, Engine
from ..engine.params import configure, specs
from ..engine.related import RelatedConfig
from ..engine.search import QueryError, SearchConfig
from .tasks import Conflict, TaskError, TaskStore

log = logging.getLogger("isis")
WEB = Path(__file__).resolve().parents[2] / "web"
LOOPBACK = ("localhost", "127.0.0.1", "::1")


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


class NewTask(BaseModel):
    name: str


class SaveTask(BaseModel):
    content: str
    base_mtime: str | None = None


class RenameTask(BaseModel):
    new_name: str


class RelatedRequest(BaseModel):
    text: str
    cursor_line: int | None = None
    params: dict = {}  # overrides of RelatedConfig, as named in /api/params


class TermRequest(BaseModel):
    term: str  # boost and exclude: the word as shown on its chip; unboost, ignore, unignore: its stem
    action: Literal["boost", "unboost", "ignore", "unignore", "exclude"]


class SteerRequest(BaseModel):
    text: str  # clauses as in a search query: x prefer, +x require, -x never show


class FeedbackRequest(BaseModel):
    action: Literal["add", "dismiss", "avoid", "restore"]
    lane: str = ""  # where the passage was shown; restoring finds it from the dismissal
    path: str = ""
    text: str = ""
    id: str = ""  # restore: the dismissal's id from the preferences the page was given
    start_line: int | None = None
    end_line: int | None = None


def create_app(engine: Engine, tasks: TaskStore, *, background: bool = True, watch: bool = True,
               allowed_hosts: list[str] | None = None) -> FastAPI:
    """`allowed_hosts`: the Host headers the API answers (None: any). A page on another site can
    point its own domain at 127.0.0.1 (DNS rebinding); checking the Host header stops it from
    reading the corpus or overwriting tasks through the browser."""
    def refresh() -> None:
        try:
            stats = engine.refresh()
            if stats["changed"] or stats["removed"] or stats["embedded"]:
                log.info("index updated: %s", stats)
        except Exception:
            log.exception("index refresh failed")

    def watch_corpus(stop: threading.Event) -> None:
        from watchfiles import watch as watch_files

        tasks_dir = tasks.folder.resolve()

        def relevant(_, path: str) -> bool:
            p = Path(path)
            try:
                hidden = any(part.startswith(".") for part in p.relative_to(engine.root).parts)
            except ValueError:
                return False
            return p.suffix.lower() in MARKDOWN and tasks_dir not in p.parents and not hidden

        for _ in watch_files(engine.root, watch_filter=relevant, stop_event=stop, debounce=800):
            refresh()

    @asynccontextmanager
    async def lifespan(_):
        stop = threading.Event()
        if background:
            threading.Thread(target=refresh, daemon=True, name="isis-index").start()
        if watch:
            threading.Thread(target=watch_corpus, args=(stop,), daemon=True, name="isis-watch").start()
        yield
        stop.set()

    # No /docs or /redoc: FastAPI's pages for them load scripts from a public CDN.
    app = FastAPI(title="isis", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    if allowed_hosts is not None:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @app.exception_handler(TaskError)
    async def task_error(_, e: TaskError):
        body = {"detail": str(e)}
        if isinstance(e, Conflict):
            body["mtime"] = e.mtime
        return JSONResponse(body, status_code=e.status)

    # ---- corpus ----

    @app.get("/api/status")
    def status():
        return {
            **plain(engine.status),
            "corpus": str(engine.root),
            "model": getattr(engine.embedder, "model_id", None),
        }

    @app.get("/api/files")
    def files():
        return [{"path": p} for p in sorted(engine.snapshot.file_rows)]

    @app.get("/api/file")
    def file(path: str):
        try:
            content = engine.resolve(path).read_text(encoding="utf-8", errors="replace")
        except ValueError as e:
            raise HTTPException(400, str(e))
        except OSError:
            raise HTTPException(404, f"no such file: {path}")
        return {"path": path, "content": content}

    @app.get("/api/params")
    def params():
        """The parameters the page may tune, with defaults, ranges and help."""
        return {"search": specs(SearchConfig), "related": specs(RelatedConfig)}

    @app.get("/api/search")
    def search(q: str = "", mode: Literal["lexical", "hybrid"] = "lexical", limit: int = 50, params: str = ""):
        try:
            overrides = json.loads(params) if params else {}
        except ValueError:
            raise HTTPException(400, "params must be a JSON object")
        if not isinstance(overrides, dict):
            raise HTTPException(400, "params must be a JSON object")
        started = time.perf_counter()
        try:
            hits = engine.search(q, limit=min(limit, 200), mode=mode, config=configure(SearchConfig, overrides))
        except QueryError as e:
            raise HTTPException(400, str(e))
        return {"hits": plain(hits), "ms": round((time.perf_counter() - started) * 1000, 1)}

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

    @app.delete("/api/tasks/{name}")
    def delete_task(name: str):
        tasks.delete(name)
        return {"ok": True}

    @app.post("/api/tasks/{name}/related")
    def related(name: str, body: RelatedRequest):
        prefs = tasks.prefs(name)
        result = engine.related(
            body.text,
            cursor_line=body.cursor_line,
            prefs=prefs,
            config=configure(RelatedConfig, body.params),
            seed=zlib.crc32(name.encode()),
        )
        return {**plain(result), "prefs": prefs.summary()}

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
            with tasks.edit_prefs(name) as prefs:
                prefs.set_steer(body.text)
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
        if body.action != "restore" and not body.lane:
            raise HTTPException(400, "feedback needs the lane the passage was shown in")
        with tasks.edit_prefs(name) as prefs:
            if body.action == "restore" and body.id:
                prefs.restore_id(body.id)
            elif body.action == "restore":
                prefs.restore(body.path, body.text)
            elif body.action in ("dismiss", "avoid"):
                prefs.dismiss(body.lane, body.path, body.text, body.start_line, body.end_line,
                              avoid=body.action == "avoid")
            else:
                prefs.record(body.lane, added=True)
        return prefs.summary()

    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    return app


def app_from_env() -> FastAPI:
    """The app `isis serve` runs, configured through ISIS_CORPUS, ISIS_TASKS, ISIS_MODEL and
    ISIS_HOSTS (comma-separated Host names to answer, or "*"; default: this machine only)."""
    from ..engine.semantic import Embedder

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    corpus = Path(os.environ["ISIS_CORPUS"])
    tasks_dir = Path(os.environ.get("ISIS_TASKS") or corpus / "_tasks")
    model = os.environ.get("ISIS_MODEL", "")
    engine = Engine(corpus, embedder=Embedder(model) if model else None, exclude=(tasks_dir,))
    hosts = [h for h in os.environ.get("ISIS_HOSTS", ",".join(LOOPBACK)).split(",") if h]
    return create_app(engine, TaskStore(tasks_dir), allowed_hosts=hosts)
