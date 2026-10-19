"""The web app: a JSON API over the engine and the task store, and the static page in web/."""

from __future__ import annotations

import dataclasses
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
from pydantic import BaseModel

from ..engine.index import MARKDOWN, Engine
from .tasks import Conflict, TaskError, TaskStore

log = logging.getLogger("isis")
WEB = Path(__file__).resolve().parents[2] / "web"


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


class TermRequest(BaseModel):
    term: str
    action: Literal["pin", "unpin", "ban", "unban"]


class FeedbackRequest(BaseModel):
    action: Literal["add", "dismiss"]
    lane: str
    path: str = ""
    text: str = ""


def create_app(engine: Engine, tasks: TaskStore, *, background: bool = True, watch: bool = True) -> FastAPI:
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

    app = FastAPI(title="isis", lifespan=lifespan)

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

    @app.get("/api/search")
    def search(q: str = "", mode: Literal["lexical", "hybrid"] = "lexical", limit: int = 50):
        started = time.perf_counter()
        hits = engine.search(q, limit=min(limit, 200), mode=mode)
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
            body.text, cursor_line=body.cursor_line, prefs=prefs, seed=zlib.crc32(name.encode())
        )
        return {**plain(result), "prefs": prefs.to_dict()}

    @app.get("/api/tasks/{name}/prefs")
    def get_prefs(name: str):
        return tasks.prefs(name).to_dict()

    @app.post("/api/tasks/{name}/terms")
    def terms(name: str, body: TermRequest):
        prefs = tasks.prefs(name)
        pins, bans = set(prefs.pins), set(prefs.bans)
        if body.action == "pin":
            pins.add(body.term)
            bans.discard(body.term)
        elif body.action == "ban":
            bans.add(body.term)
            pins.discard(body.term)
        elif body.action == "unpin":
            pins.discard(body.term)
        else:
            bans.discard(body.term)
        prefs.pins, prefs.bans = sorted(pins), sorted(bans)
        tasks.save_prefs(name, prefs)
        return prefs.to_dict()

    @app.post("/api/tasks/{name}/feedback")
    def feedback(name: str, body: FeedbackRequest):
        prefs = tasks.prefs(name)
        prefs.record(body.lane, added=body.action == "add")
        if body.action == "dismiss" and body.text:
            prefs.dismissed.append({"path": body.path, "text": body.text})
        tasks.save_prefs(name, prefs)
        return prefs.to_dict()

    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    return app


def app_from_env() -> FastAPI:
    """The app `isis serve` runs, configured through ISIS_CORPUS, ISIS_TASKS and ISIS_MODEL."""
    from ..engine.semantic import Embedder

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    corpus = Path(os.environ["ISIS_CORPUS"])
    tasks_dir = Path(os.environ.get("ISIS_TASKS") or corpus / "_tasks")
    model = os.environ.get("ISIS_MODEL", "")
    engine = Engine(corpus, embedder=Embedder(model) if model else None, exclude=(tasks_dir,))
    return create_app(engine, TaskStore(tasks_dir))
