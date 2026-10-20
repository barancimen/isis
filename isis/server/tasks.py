"""Task documents: one markdown file each in the tasks folder, plus a small JSON file of
preferences (pinned and banned terms, dismissed passages, feedback per lane) in `.meta/`.
Deleted tasks are moved to `.trash/`, not removed."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from ..engine.related import TaskPrefs

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


class TaskStore:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.meta = self.folder / ".meta"
        self.trash = self.folder / ".trash"

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
        path = self._path(name)
        if path.exists():
            raise TaskError(f"a task named {path.stem!r} already exists", 409)
        path.write_text("", encoding="utf-8")
        return self.read(path.stem)

    def read(self, name: str) -> dict:
        path = self._existing(name)
        return {"name": path.stem, "content": path.read_text(encoding="utf-8"), "mtime": _mtime(path)}

    def write(self, name: str, content: str, base_mtime: str | None = None) -> str:
        """Save, unless the file changed on disk since `base_mtime` (then raise Conflict)."""
        path = self._existing(name)
        if base_mtime is not None and _mtime(path) != base_mtime:
            raise Conflict(_mtime(path))
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
        return _mtime(path)

    def rename(self, name: str, new_name: str) -> dict:
        path, target = self._existing(name), self._path(new_name)
        if target.exists() and target != path:
            raise TaskError(f"a task named {target.stem!r} already exists", 409)
        path.rename(target)
        meta = self._meta_path(path.stem)
        if meta.exists():
            meta.rename(self._meta_path(target.stem))
        return self.read(target.stem)

    def delete(self, name: str) -> None:
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
            return TaskPrefs.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return TaskPrefs()

    def save_prefs(self, name: str, prefs: TaskPrefs) -> None:
        self.meta.mkdir(exist_ok=True)
        prefs.dismissed = prefs.dismissed[-MAX_DISMISSED:]
        path = self._meta_path(self._existing(name).stem)
        path.write_text(json.dumps(prefs.to_dict(), indent=1), encoding="utf-8")

    def forget_prefs(self, name: str) -> None:
        """Start the task's preferences over. The old file goes to .trash, like a deleted task."""
        path = self._meta_path(self._existing(name).stem)
        if path.exists():
            self.trash.mkdir(exist_ok=True)
            path.rename(self.trash / f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.prefs.json")
