"""`isis serve` builds the Rust extension when it is missing or outdated. Nothing is compiled
here: the build command is replaced by a fake."""

import os
import threading
from types import SimpleNamespace

import pytest

from isis import cli
from isis.engine import native


def source(tmp_path, *, mtime: float):
    """A native/ folder with the files the extension is built from."""
    folder = tmp_path / "native"
    (folder / "src").mkdir(parents=True)
    for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml", "src/lib.rs"):
        (folder / name).write_text("x")
        os.utime(folder / name, (mtime, mtime))
    return folder


def installed(tmp_path, *, mtime: float):
    folder = tmp_path / "site" / "isis_native"
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text("")
    built = folder / "isis_native.cpython-310-x86_64-linux-gnu.so"
    built.write_text("")
    os.utime(built, (mtime, mtime))
    return folder


def test_build_state(tmp_path):
    native_dir = source(tmp_path, mtime=1000)
    assert native.build_state(tmp_path / "elsewhere", tmp_path) is None  # no source to build from
    assert native.build_state(native_dir, tmp_path) == "missing"  # a folder without the library
    assert native.build_state(native_dir, installed(tmp_path, mtime=2000)) is None
    os.utime(native_dir / "src" / "lib.rs", (3000, 3000))  # a git pull or an edit after it was built
    assert native.build_state(native_dir, tmp_path / "site" / "isis_native") == "outdated"


@pytest.fixture
def no_build_yet(monkeypatch):
    """No extension loaded, Rust installed, a writable environment, and a network allowed."""
    monkeypatch.setattr(native, "isis_native", None)
    monkeypatch.setattr(native.shutil, "which", lambda name, path=None: f"/bin/{name}")
    monkeypatch.setattr(native.os, "access", lambda path, mode: True)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)


def finish_build():
    for thread in threading.enumerate():
        if thread.name == "isis-build-native":
            thread.join(10)


def never(*args, **kwargs):
    raise AssertionError("it shouldn't build")


def test_it_leaves_alone_what_it_cannot_build(no_build_yet, monkeypatch):
    monkeypatch.setattr(native, "build_state", lambda: None)
    assert native.start_build(run=never) == "current"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert native.start_build("missing", run=never) == "offline"
    monkeypatch.delenv("HF_HUB_OFFLINE")
    monkeypatch.setattr(native.os, "access", lambda path, mode: False)
    assert native.start_build("missing", run=never) == "read-only"  # the live server's user
    monkeypatch.setattr(native.shutil, "which", lambda name, path=None: None)
    assert native.start_build("missing", run=never) == "no cargo"


def test_a_missing_extension_is_used_once_built(no_build_yet, monkeypatch):
    commands = []

    def pip(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    built = object()
    monkeypatch.setattr(native.importlib, "import_module", lambda name: built)
    assert native.start_build("missing", run=pip) == "started"
    finish_build()
    assert commands and commands[0][-4:-1] == ["install", "--quiet", "--disable-pip-version-check"]
    assert commands[0][-1] == str(native.NATIVE_DIR)
    assert native.isis_native is built


def test_an_outdated_extension_waits_for_a_restart_and_a_failed_build_changes_nothing(no_build_yet, monkeypatch, caplog):
    monkeypatch.setattr(native.importlib, "import_module", never)
    caplog.set_level("INFO", logger="isis.engine.native")
    ok = lambda command, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr="")
    assert native.start_build("outdated", run=ok) == "started"
    finish_build()
    assert "restart isis" in caplog.text
    failed = lambda command, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="error: linker `cc` not found")
    assert native.start_build("missing", run=failed) == "started"
    finish_build()
    assert "linker `cc` not found" in caplog.text
    assert native.isis_native is None


def test_serve_builds_unless_told_not_to(monkeypatch, tmp_path):
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    for key in ("ISIS_CORPUS", "ISIS_MODELS", "ISIS_HOSTS", "ISIS_BUILD_NATIVE"):
        monkeypatch.setenv(key, "")  # put back as they were afterwards
    cli.main(["serve", str(tmp_path)])
    assert os.environ["ISIS_BUILD_NATIVE"] == "1"
    cli.main(["serve", str(tmp_path), "--no-build"])
    assert os.environ["ISIS_BUILD_NATIVE"] == "0"
