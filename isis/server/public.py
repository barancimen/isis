"""Public mode (`isis serve --public`): anyone may use the site, with no login. Each visitor gets
private tasks through an anonymous cookie, and limits keep one visitor from using up the server.
The corpus is read-only and open to all. Run it only behind a proxy that sets CF-Connecting-IP."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import cookie_parser

from .tasks import MAX_DISMISSED, TaskError, TaskStore

log = logging.getLogger(__name__)

PATTERN_SECONDS = 3.0  # matching time for all the regexes of one request
COOKIE = "isis_visitor"
COOKIE_DAYS = 365
TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
CLEAN_EVERY = 3600  # seconds
TOUCH_EVERY = 3600  # seconds between marks that a workspace is still in use
TOTALS_FOR = 30  # seconds that the total size of all workspaces is reused
STORES_KEPT = 2000
SLOT_CHECK = 0.05  # seconds between looks for a free slot
SECURITY_HEADERS = [
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    # 'unsafe-inline' for styles only: the editor adds its own style elements.
    ("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                                "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"),
]


@dataclass(frozen=True)
class Limits:
    max_tasks: int = 20
    max_task_bytes: int = 512 * 1024
    max_workspace_bytes: int = 2 * 1024 * 1024
    max_total_bytes: int = 1024 * 1024 * 1024  # all workspaces together
    max_workspaces: int = 5000
    idle_days: int = 30  # a workspace not used for this long is deleted
    new_workspaces_per_hour: int = 5  # per address, so a cleared cookie doesn't get round the caps
    max_liked: int = 200  # each one is embedded and scored on every Related refresh
    request_bytes: int = 1024 * 1024  # a saved task
    small_request_bytes: int = 16 * 1024  # anything else that is stored, such as a dismissed passage
    related_bytes: int = 128 * 1024  # the text of a task to refresh Related for: it may all be embedded
    query_bytes: int = 8 * 1024  # a URL's query string, which holds a search
    # What one address may do: (burst, requests per second after the burst).
    rates: dict = field(default_factory=lambda: {
        "search": (60, 2.0),
        "related": (30, 0.5),
        "write": (120, 2.0),
        "read": (300, 10.0),
    })
    related_bytes_per_token: int = 16 * 1024  # a long Related request counts as several
    # Seconds that one address's searches and Related refreshes may run: (burst, seconds per second
    # after it). Charged afterwards, so a request's real cost counts, whatever made it expensive.
    work: tuple[float, float] = (30.0, 0.25)
    busy_slots: int = 2  # searches and Related refreshes that run at once (the server has few CPUs)
    busy_waiters: int = 16  # requests that may queue for a slot; beyond that they are told to retry
    busy_wait: float = 3.0  # seconds a request queues before it is told to retry


class RateLimiter:
    """Token buckets, one per key: a burst, then a steady rate."""

    MAX_BUCKETS = 100_000

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, when)
        self.lock = threading.Lock()

    def _tokens(self, key: str, burst: float, per_second: float, now: float) -> float:
        if len(self.buckets) >= self.MAX_BUCKETS and key not in self.buckets:
            self.buckets.clear()  # forgetting everyone errs on the side of letting them in
        tokens, when = self.buckets.get(key, (burst, now))
        return min(burst, tokens + (now - when) * per_second)

    def charge(self, key: str, burst: float, per_second: float, amount: float) -> None:
        """Take `amount` tokens for work already done. The balance can go below zero, and
        retry_after with cost=0 then waits until it is paid off."""
        now = self.clock()
        with self.lock:
            self.buckets[key] = (self._tokens(key, burst, per_second, now) - amount, now)

    def retry_after(self, key: str, burst: float, per_second: float, cost: float = 1) -> float:
        """Take `cost` tokens and return 0, or return the seconds to wait if there are too few."""
        cost = min(cost, burst)
        now = self.clock()
        with self.lock:
            tokens = self._tokens(key, burst, per_second, now)
            if tokens < cost:
                self.buckets[key] = (tokens, now)
                return (cost - tokens) / per_second
            self.buckets[key] = (tokens - cost, now)
            return 0


def client_address(scope) -> str:
    """Who sent the request. An IPv6 user usually controls a whole /64, so that counts as one."""
    raw = Headers(scope=scope).get("cf-connecting-ip") or (scope.get("client") or ("unknown",))[0]
    try:
        address = ipaddress.ip_address(raw)
    except ValueError:
        return raw[:64]
    return str(ipaddress.ip_network(f"{address}/64", strict=False)) if address.version == 6 else str(address)


_visitor_store: ContextVar[VisitorStore] = ContextVar("isis_visitor_store")
_visitor_address: ContextVar[str] = ContextVar("isis_visitor_address", default="unknown")


class VisitorStore(TaskStore):
    """One visitor's tasks. The folder appears with the first task, so a visit that only searches
    leaves nothing on disk. Deleted tasks are removed at once, or they would escape the caps."""

    def __init__(self, folder: Path, workspaces: Workspaces):
        super().__init__(folder, make_folder=False)
        self.workspaces = workspaces
        self.limits = workspaces.limits

    def used(self) -> int:
        # The order of the list is bookkeeping, and small: it shouldn't cost anyone a save.
        return sum(p.stat().st_size for p in self.folder.rglob("*") if p.is_file() and p != self.order)

    def create(self, name: str) -> dict:
        self.clean(name)
        self.workspaces.check_room()
        if not self.folder.exists():
            self.workspaces.admit()
        elif sum(1 for _ in self.folder.glob("*.md")) >= self.limits.max_tasks:
            raise TaskError(f"you can have at most {self.limits.max_tasks} tasks here; delete one first", 403)
        return super().create(name)

    def write(self, name: str, content: str, base_mtime: str | None = None) -> str:
        size = len(content.encode("utf-8", "replace"))
        if size > self.limits.max_task_bytes:
            raise TaskError(f"this task is too long to save here (the limit is {self.limits.max_task_bytes // 1024} KB)", 413)
        self.workspaces.check_room()
        path = self._existing(name)
        if self.used() - path.stat().st_size + size > self.limits.max_workspace_bytes:
            raise TaskError(self._full(), 413)
        return super().write(name, content, base_mtime)

    def save_prefs(self, name: str, prefs) -> None:
        self.workspaces.check_room()
        if len(prefs.liked) > self.limits.max_liked:
            raise TaskError(f"you can ask for more like at most {self.limits.max_liked} passages per task", 403)
        prefs.dismissed = prefs.dismissed[-MAX_DISMISSED:]
        path = self._existing_meta_path(name)
        before = path.stat().st_size if path.exists() else 0
        size = len(json.dumps(prefs.to_dict(), indent=1))  # pure ASCII, so characters are bytes
        if self.used() - before + size > self.limits.max_workspace_bytes:
            raise TaskError(self._full(), 413)
        super().save_prefs(name, prefs)

    def _full(self) -> str:
        return f"your tasks are using all their space ({self.limits.max_workspace_bytes // (1024 * 1024)} MB); delete one first"

    def _to_trash(self, path: Path, name: str) -> Path:
        path.unlink()
        return path


class Workspaces:
    """All visitors' stores, under one folder. It stands in for a TaskStore in the app: anything
    else asked of it goes to the store of the visitor whose request is running."""

    def __init__(self, folder: str | Path, limits: Limits | None = None, limiter: RateLimiter | None = None,
                 base_path: str = ""):
        """`base_path`: where the site lives on its domain, such as /isis (empty: at the top)."""
        self.base_path = base_path.rstrip("/")
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.limits = limits or Limits()
        self.limiter = limiter or RateLimiter()
        self._stores: OrderedDict[str, VisitorStore] = OrderedDict()
        self._touched: dict[str, float] = {}
        self._lock = threading.Lock()
        self._totals = (0.0, 0, 0)  # when, bytes, workspaces

    def __getattr__(self, name: str):
        return getattr(_visitor_store.get(), name)

    def store(self, token: str) -> VisitorStore:
        # The folder is named by a hash, so a look at the disk or a log shows no working cookie.
        key = hashlib.sha256(token.encode()).hexdigest()[:32]
        with self._lock:
            store = self._stores.get(key)
            if store is None:
                store = self._stores[key] = VisitorStore(self.folder / key, self)
                while len(self._stores) > STORES_KEPT:
                    gone, _ = self._stores.popitem(last=False)
                    self._touched.pop(gone, None)
            else:
                self._stores.move_to_end(key)
            stale = time.time() - self._touched.get(key, 0) > TOUCH_EVERY
            if stale:
                self._touched[key] = time.time()
        if stale and store.folder.exists():
            os.utime(store.folder)  # it was used, so the cleaner leaves it
        return store

    def totals(self) -> tuple[int, int]:
        """Bytes and workspaces in use, from a scan at most a few seconds old."""
        with self._lock:
            when, size, count = self._totals
            if time.monotonic() - when < TOTALS_FOR and when:
                return size, count
            self._totals = (time.monotonic(), size, count)  # one request at a time does the scan
        size = count = 0
        for root, _, names in os.walk(self.folder):
            count += os.path.dirname(root) == str(self.folder)
            for name in names:
                try:
                    size += os.lstat(os.path.join(root, name)).st_size
                except OSError:  # deleted since the listing
                    pass
        with self._lock:
            self._totals = (time.monotonic(), size, count)
        return size, count

    def check_room(self) -> None:
        if self.totals()[0] >= self.limits.max_total_bytes:
            raise TaskError("this site has no space left for new work; your tasks are safe, but this change was not saved", 507)

    def admit(self) -> None:
        """A first task, which makes a new workspace."""
        if self.totals()[1] >= self.limits.max_workspaces:
            raise TaskError("this site has no room for new visitors' tasks at the moment", 507)
        per_hour = self.limits.new_workspaces_per_hour
        wait = self.limiter.retry_after(f"new:{_visitor_address.get()}", per_hour, per_hour / 3600)
        if wait:
            raise TaskError("too many new workspaces from this address; try again later", 429)

    def remove_idle(self, now: float | None = None) -> int:
        """Delete the workspaces not used for `idle_days`. Returns how many went."""
        cutoff = (time.time() if now is None else now) - self.limits.idle_days * 86400
        gone = 0
        for folder in self.folder.iterdir():
            try:
                if folder.is_dir() and folder.stat().st_mtime < cutoff:
                    shutil.rmtree(folder)
                    gone += 1
            except OSError as e:
                log.warning("cannot clean a workspace: %s", e)
        if gone:
            log.info("deleted %d idle workspace(s)", gone)
        return gone

    def keep_clean(self, stop: threading.Event) -> None:
        while True:
            try:
                self.remove_idle()
            except OSError as e:
                log.warning("cannot clean the workspaces: %s", e)
            if stop.wait(CLEAN_EVERY):
                return


def body_limit(limits: Limits, kind: str, method: str, path: str) -> int:
    if kind == "related":
        return limits.related_bytes
    if method == "PUT" and path.startswith("/api/tasks/") and not path.endswith(("/steer", "/settings")):
        return limits.request_bytes  # the text of a task
    return limits.small_request_bytes


def kind_of(method: str, path: str) -> str:
    if path == "/api/search":
        return "search"
    if method == "POST" and path.startswith("/api/tasks/") and path.endswith("/related"):
        return "related"
    if path.startswith("/api/tasks") and method in ("POST", "PUT", "DELETE", "PATCH"):
        return "write"
    return "read"


class PublicGuard:
    """Checks each request before the app sees it: its size, its address's rate, and a free slot
    if it is expensive. It also finds the visitor, and gives a new one a cookie."""

    def __init__(self, app, workspaces: Workspaces):
        self.app = app
        self.workspaces = workspaces
        self.limits = workspaces.limits
        self.base_path = workspaces.base_path
        self.running = 0
        self.waiting = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if self.base_path:
            if scope["path"] == self.base_path:  # the page's relative links need the final slash
                query = scope.get("query_string", b"").decode("latin-1")
                target = f"{self.base_path}/" + (f"?{query}" if query else "")
                return await RedirectResponse(target, 308)(scope, receive, send)
            if not scope["path"].startswith(f"{self.base_path}/"):
                return await JSONResponse({"detail": "not found"}, 404)(scope, receive, send)
            path = scope["path"][len(self.base_path):]
            scope = {**scope, "path": path, "raw_path": path.encode("utf-8", "replace")}
        limits, headers = self.limits, Headers(scope=scope)
        kind = kind_of(scope["method"], scope["path"])
        address = client_address(scope)
        max_body = body_limit(limits, kind, scope["method"], scope["path"])
        limiter = self.workspaces.limiter
        token, renew = "", False

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                out = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS:
                    out[name] = value
                if scope["path"].startswith("/api/"):  # answers differ per visitor: never keep them
                    out["Cache-Control"] = "no-store"
                    out.add_vary_header("Cookie")
                if renew:
                    out.append("Set-Cookie", f"{COOKIE}={token}; Path={self.base_path or '/'}; "
                                             f"Max-Age={COOKIE_DAYS * 86400}; HttpOnly; SameSite=Lax; Secure")
            await send(message)

        async def refuse(status: int, detail: str, retry_after: float | None = None):
            extra = {} if retry_after is None else {"Retry-After": str(max(1, round(retry_after)))}
            await JSONResponse({"detail": detail}, status, headers=extra)(scope, receive, send_with_headers)

        if len(scope.get("query_string", b"")) > limits.query_bytes:
            return await refuse(414, "this request is too long")
        length = headers.get("content-length", "")
        if not length.isdigit() and scope["method"] in ("POST", "PUT", "PATCH"):
            length = str(max_body)  # size not stated: count it as the largest
        if int(length or 0) > max_body:
            return await refuse(413, f"this request is too long (the limit is {max_body // 1024} KB)")

        burst, per_second = limits.rates[kind]
        cost = 1 + int(length or 0) // limits.related_bytes_per_token if kind == "related" else 1
        wait = limiter.retry_after(f"{kind}:{address}", burst, per_second, cost)
        expensive = kind in ("search", "related")
        if not wait and expensive:
            wait = limiter.retry_after(f"work:{address}", *limits.work, cost=0)
        if wait:
            return await refuse(429, "too many requests from this address; wait a moment and try again", wait)

        token = cookie_parser(headers.get("cookie", "")).get(COOKIE, "")
        renew = scope["path"] == "/"  # a visit to the page starts the cookie's time again
        if not TOKEN.fullmatch(token):
            token, renew = secrets.token_urlsafe(32), True
        _visitor_store.set(self.workspaces.store(token))
        _visitor_address.set(address)

        received = 0

        async def receive_some():  # a stated length can be false, so count what really arrives
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > max_body:
                raise HTTPException(413, f"this request is too long (the limit is {max_body // 1024} KB)")
            return message

        if not expensive:
            return await self.app(scope, receive_some, send_with_headers)
        if not await self._take_slot():
            return await refuse(503, "the server is busy; try again in a moment", 2)
        started = time.monotonic()
        try:
            await self.app(scope, receive_some, send_with_headers)
        finally:
            self.running -= 1
            limiter.charge(f"work:{address}", *limits.work, time.monotonic() - started)

    async def _take_slot(self) -> bool:
        """Wait a little for a place among the expensive requests. All of this runs on the one
        event loop, so the counts need no lock."""
        if self.waiting >= self.limits.busy_waiters:
            return False
        self.waiting += 1
        try:
            deadline = time.monotonic() + self.limits.busy_wait
            while self.running >= self.limits.busy_slots:
                if time.monotonic() >= deadline:
                    return False
                await asyncio.sleep(SLOT_CHECK)
            self.running += 1
            return True
        finally:
            self.waiting -= 1
