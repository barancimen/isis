"""Task documents: one markdown file each in the tasks folder, plus a small JSON file of
preferences (the steer line, ignored keywords, dismissed passages, feedback per lane) in `.meta/`.
Deleted tasks are moved to `.trash/`, not removed."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from ..engine.related import TaskPrefs

log = logging.getLogger("isis")
ILLEGAL = re.compile(r'[\\/:*?"<>|#^\[\]\x00-\x1f]')
MAX_DISMISSED = 500


class TaskError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class Conflict(TaskError):
    """The file changed on disk since the client loaded it."""

    def __init__(self, mtime: str):
        super().__init__("the task changed on disk since it was loaded", 409)
        self.mtime = mtime


def _mtime(path: Path) -> str:
    return str(path.stat().st_mtime_ns)  # a string: nanoseconds don't fit a JavaScript number


def _replace(path: Path, text: str) -> None:
    """Write `text` to `path` atomically: a temporary file of its own, flushed, then renamed."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class TaskStore:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.meta = self.folder / ".meta"
        self.trash = self.folder / ".trash"
        self._locks: dict[str, threading.RLock] = {}
        self._locks_lock = threading.Lock()

    def lock(self, name: str) -> threading.RLock:
        """One lock per task, held while its preferences are read, changed and saved."""
        with self._locks_lock:
            return self._locks.setdefault(self.clean(name), threading.RLock())

    @staticmethod
    def clean(name: str) -> str:
        name = ILLEGAL.sub("-", name).strip().lstrip(".").strip()
        if not name:
            raise TaskError("a task needs a name")
        return name[:120]

    def _path(self, name: str) -> Path:
        return self.folder / f"{self.clean(name)}.md"

    def _existing(self, name: str) -> Path:
        path = self._path(name)
        if not path.is_file():
            raise TaskError(f"no task named {name!r}", 404)
        return path

    def list(self) -> list[dict]:
        files = sorted(self.folder.glob("*.md"), key=lambda p: p.stem.lower())
        return [{"name": p.stem, "mtime": _mtime(p)} for p in files]

    def create(self, name: str) -> dict:
        self.folder.mkdir(parents=True, exist_ok=True)
        with self.lock(name):
            path = self._path(name)
            if path.exists():
                raise TaskError(f"a task named {path.stem!r} already exists", 409)
            path.write_text("", encoding="utf-8")
            return self.read(path.stem)

    def read(self, name: str) -> dict:
        with self.lock(name):
            path = self._existing(name)
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raise TaskError(f"{path.name} is not UTF-8 text; convert it to UTF-8 to open it here", 422)
            return {"name": path.stem, "content": content, "mtime": _mtime(path)}

    def write(self, name: str, content: str, base_mtime: str | None = None) -> str:
        """Save, unless the file changed on disk since `base_mtime` (then raise Conflict). The
        check and the write happen under the task's lock, so a save can't overwrite another
        save it hasn't seen, or bring back a task deleted meanwhile."""
        with self.lock(name):
            path = self._existing(name)
            before = _mtime(path)
            if base_mtime is not None and before != base_mtime:
                raise Conflict(before)
            _replace(path, content)
            if int(_mtime(path)) <= int(before):
                # File times move in ticks of a few ms: a save in the same tick would leave the
                # time unchanged, and a second writer's stale save would pass the check.
                os.utime(path, ns=(time.time_ns(), int(before) + 1))
            return _mtime(path)

    def rename(self, name: str, new_name: str) -> dict:
        first, second = sorted([self.clean(name), self.clean(new_name)])  # one order for A->B and B->A
        with self.lock(first), self.lock(second):
            path, target = self._existing(name), self._path(new_name)
            if target.exists() and target != path:
                raise TaskError(f"a task named {target.stem!r} already exists", 409)
            path.rename(target)
            meta = self._meta_path(path.stem)
            if meta.exists():
                meta.rename(self._meta_path(target.stem))
            return self.read(target.stem)

    def delete(self, name: str) -> None:
        with self.lock(name):
            path = self._existing(name)
            self.trash.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path.rename(self.trash / f"{path.stem}.{stamp}.md")
            meta = self._meta_path(path.stem)
            if meta.exists():
                meta.rename(self.trash / f"{path.stem}.{stamp}.json")

    # ---- preferences ----

    def _meta_path(self, name: str) -> Path:
        return self.meta / f"{name}.json"

    def prefs(self, name: str) -> TaskPrefs:
        path = self._meta_path(self._existing(name).stem)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return TaskPrefs()
        try:
            return TaskPrefs.from_dict(json.loads(text))
        except (ValueError, TypeError, AttributeError):  # saves are atomic, so the file itself is bad: keep it
            with self.lock(name):
                self.trash.mkdir(exist_ok=True)
                kept = self.trash / f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bad.json"
                try:
                    path.rename(kept)
                    log.warning("unreadable preferences for task %r moved to %s", path.stem, kept)
                except OSError:
                    pass
            return TaskPrefs()

    def save_prefs(self, name: str, prefs: TaskPrefs) -> None:
        self.meta.mkdir(exist_ok=True)
        prefs.dismissed = prefs.dismissed[-MAX_DISMISSED:]
        path = self._meta_path(self._existing(name).stem)
        _replace(path, json.dumps(prefs.to_dict(), indent=1))

    @contextmanager
    def edit_prefs(self, name: str):
        """Read, change and save a task's preferences, one change at a time."""
        with self.lock(name):
            prefs = self.prefs(name)
            yield prefs
            self.save_prefs(name, prefs)

    def forget_prefs(self, name: str) -> None:
        """Start the task's preferences over. The old file goes to .trash, like a deleted task."""
        with self.lock(name):
            path = self._meta_path(self._existing(name).stem)
            if path.exists():
                self.trash.mkdir(exist_ok=True)
                path.rename(self.trash / f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.prefs.json")
