"""Task documents: one markdown file each in the tasks folder, plus a small JSON file of
preferences (the steer line, ignored keywords, dismissed passages, feedback per lane) in `.meta/`.
Deleted tasks are moved to `.trash/`, not removed. `.order.json` keeps the order of the list."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path

from ..engine.related import TaskPrefs

log = logging.getLogger(__name__)
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


def _stamp() -> str:
    """A name for a copy in .trash: renaming onto an older copy from the same second would lose it."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000:06d}"


class TaskStore:
    def __init__(self, folder: str | Path, *, make_folder: bool = True):
        self.folder = Path(folder)
        if make_folder:
            self.folder.mkdir(parents=True, exist_ok=True)
        self.meta = self.folder / ".meta"
        self.trash = self.folder / ".trash"
        self.order = self.folder / ".order.json"
        self._lock = threading.RLock()  # held while a task or its preferences are read, changed and saved
        self._unnamed: set[str] = set()

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

    def _addressable(self, stem: str) -> bool:
        """Whether a file name is one the API can reach. A hand-made "C# notes.md" isn't."""
        try:
            return self.clean(stem) == stem
        except TaskError:
            return False

    def _saved_order(self) -> list[str]:
        try:
            names = json.loads(self.order.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [n for n in names if isinstance(n, str)] if isinstance(names, list) else []

    def _save_order(self, names: list[str]) -> None:
        try:
            if names:
                _replace(self.order, json.dumps(names, ensure_ascii=False))
            else:
                self.order.unlink(missing_ok=True)
        except OSError as e:  # the task itself is fine: only its place in the list is lost
            log.warning("cannot save the order of the tasks: %s", e)

    def _names(self) -> list[str]:
        return [task["name"] for task in self.list()]

    def list(self) -> list[dict]:
        """In the saved order. Files that it doesn't name, such as ones made by hand, follow by name."""
        place = {name: n for n, name in enumerate(self._saved_order())}
        out = []
        for p in sorted(self.folder.glob("*.md"), key=lambda p: (place.get(p.stem, len(place)), p.stem.lower())):
            try:
                if not self._addressable(p.stem):
                    if p.name not in self._unnamed:
                        self._unnamed.add(p.name)
                        log.warning("task file %s not listed: rename it without these characters: \\ / : * ? \" < > | # ^ [ ]", p.name)
                    continue
                out.append({"name": p.stem, "mtime": _mtime(p)})
            except OSError:  # deleted since the listing
                continue
        return out

    def create(self, name: str) -> dict:
        self.folder.mkdir(parents=True, exist_ok=True)
        with self._lock:
            path = self._path(name)
            if path.exists():
                raise TaskError(f"a task named {path.stem!r} already exists", 409)
            names = self._names()
            path.write_text("", encoding="utf-8")
            self._save_order([*names, path.stem])
            return self.read(path.stem)

    def read(self, name: str) -> dict:
        with self._lock:
            path = self._existing(name)
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raise TaskError(f"{path.name} is not UTF-8 text; convert it to UTF-8 to open it here", 422)
            return {"name": path.stem, "content": content, "mtime": _mtime(path)}

    def write(self, name: str, content: str, base_mtime: str | None = None) -> str:
        """Save, unless the file changed on disk since `base_mtime` (then raise Conflict). The
        check and the write happen under the lock, so a save can't overwrite another
        save it hasn't seen, or bring back a task deleted meanwhile."""
        with self._lock:
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
        with self._lock:
            path, target = self._existing(name), self._path(new_name)
            if target.exists() and target != path:
                raise TaskError(f"a task named {target.stem!r} already exists", 409)
            names = self._names()
            path.rename(target)
            self._save_order([target.stem if n == path.stem else n for n in names])
            meta = self._meta_path(path.stem)
            if meta.exists():
                meta.rename(self._meta_path(target.stem))
            return self.read(target.stem)

    def delete(self, name: str) -> None:
        with self._lock:
            path = self._existing(name)
            stamp = _stamp()
            self._to_trash(path, f"{path.stem}.{stamp}.md")
            meta = self._meta_path(path.stem)
            if meta.exists():
                self._to_trash(meta, f"{path.stem}.{stamp}.json")
            self._save_order(self._names())

    def _to_trash(self, path: Path, name: str) -> Path:
        self.trash.mkdir(exist_ok=True)
        kept = self.trash / name
        path.rename(kept)
        return kept

    # ---- preferences ----

    def _meta_path(self, name: str) -> Path:
        return self.meta / f"{name}.json"

    def _existing_meta_path(self, name: str) -> Path:
        return self._meta_path(self._existing(name).stem)

    def prefs(self, name: str) -> TaskPrefs:
        path = self._existing_meta_path(name)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return TaskPrefs()
        except UnicodeDecodeError:
            text = ""  # kept aside below, like bad JSON
        try:
            return TaskPrefs.from_dict(json.loads(text))
        except (ValueError, TypeError, AttributeError):  # saves are atomic, so the file itself is bad: keep it
            with self._lock, suppress(OSError):
                kept = self._to_trash(path, f"{path.stem}.{_stamp()}.bad.json")
                log.warning("unreadable preferences for task %r moved to %s", path.stem, kept)
            return TaskPrefs()

    def save_prefs(self, name: str, prefs: TaskPrefs) -> None:
        self.meta.mkdir(exist_ok=True)
        prefs.dismissed = prefs.dismissed[-MAX_DISMISSED:]
        _replace(self._existing_meta_path(name), json.dumps(prefs.to_dict(), indent=1))

    @contextmanager
    def edit_prefs(self, name: str):
        """Read, change and save a task's preferences, one change at a time."""
        with self._lock:
            prefs = self.prefs(name)
            yield prefs
            self.save_prefs(name, prefs)

    def forget_prefs(self, name: str) -> None:
        """Start the task's preferences over. The old file goes to .trash, like a deleted task."""
        with self._lock:
            path = self._existing_meta_path(name)
            if path.exists():
                self._to_trash(path, f"{path.stem}.{_stamp()}.prefs.json")
