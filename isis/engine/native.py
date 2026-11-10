"""The optional Rust extension, isis_native (native/ in the repository; `pip install ./native`).

Two jobs of a search are loops over text in Python's regex engine: finding the words to highlight
in the lines of a result, and counting a phrase in every passage. The extension does both several
times faster, with the same answers (tests/test_native.py). Without it every function here answers
None, and the caller runs its Python code.

It never takes part in building the index: the words of the corpus are always split by text.py.

`isis serve` builds it in the background when it is missing or older than the files in native/
(see start_build), so a git pull that changes it needs no command of its own."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import threading
from collections.abc import Callable, Iterable
from pathlib import Path

from .text import STOPWORDS

log = logging.getLogger(__name__)
NATIVE_DIR = Path(__file__).resolve().parents[2] / "native"  # only in a checkout of the repository

try:
    import isis_native
except ImportError:  # not built
    isis_native = None

MAX_PHRASES = 256  # phrase patterns whose words are remembered
_phrases: dict[tuple[str, int], tuple[str, ...]] = {}
_lock = threading.Lock()


def line_spans(lines: list[str], terms: set[str]) -> list[list[tuple[int, int]]] | None:
    """term_spans (text.py) of each line, in one call."""
    if isis_native is None:
        return None
    wanted = terms - STOPWORDS  # the tokeniser drops them, so they are never highlighted
    return isis_native.term_spans(lines, wanted) if wanted else [[] for _ in lines]


def remember_phrase(pattern: re.Pattern, words: list[str]) -> None:
    """Note that `pattern` is these words with whitespace between them, in any case, so that
    phrase_counts can count it without the regex. A regex a user writes to the same pattern means
    the same, and is counted the same way."""
    if isis_native is None or not words:
        return
    with _lock:
        while len(_phrases) >= MAX_PHRASES:
            _phrases.pop(next(iter(_phrases)))
        _phrases[(pattern.pattern, pattern.flags)] = tuple(words)


def phrase_counts(pattern, texts: Iterable[str]) -> list[int] | None:
    """How often a pattern noted by remember_phrase matches in each text. None for any other
    pattern, and `texts` is then left untouched."""
    if isis_native is None:
        return None
    with _lock:
        words = _phrases.get((pattern.pattern, pattern.flags))
    if not words:
        return None
    try:
        return isis_native.count_phrase(list(texts), list(words))
    except (ValueError, UnicodeError):  # text it can't take, such as half of a surrogate pair
        return None


# ---- building it ----


def build_state(native_dir: Path = NATIVE_DIR, installed: Path | None = None) -> str | None:
    """"missing" if the extension isn't installed, "outdated" if a file it is built from changed
    after it was installed (a git pull, an edit), or None when it is current or there is nothing
    to build it from. `installed`: its folder, found by Python when not given."""
    if not (native_dir / "Cargo.toml").is_file():
        return None
    if installed is None:
        spec = importlib.util.find_spec("isis_native")
        if spec is None or not spec.origin:
            return "missing"
        installed = Path(spec.origin).parent
    built = [f.stat().st_mtime for f in installed.glob("isis_native*") if f.suffix in (".so", ".pyd")]
    if not built:
        return "missing"
    sources = [native_dir / name for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml")]
    sources += (native_dir / "src").rglob("*.rs")
    newest = max((f.stat().st_mtime for f in sources if f.is_file()), default=0.0)
    return "outdated" if newest > min(built) else None


def _search_path() -> str:
    """PATH, with the folder rustup installs into: a server started from a desktop or a service
    may not have it."""
    cargo_bin = Path.home() / ".cargo" / "bin"
    path = os.environ.get("PATH", "")
    return f"{path}{os.pathsep}{cargo_bin}" if cargo_bin.is_dir() else path


def start_build(state: str | None = None, run: Callable = subprocess.run) -> str:
    """Builds and installs the extension in a background thread when it is missing or outdated.
    Returns what it did: "current", "offline", "no cargo", "read-only" or "started". A missing
    extension is used as soon as it is built; an outdated one is already loaded, and the new one
    takes over at the next start. A build downloads maturin and the Rust crates, so it is skipped
    with HF_HUB_OFFLINE=1, which also keeps the models off the network."""
    state = build_state() if state is None else state
    if state is None:
        return "current"
    what = "isn't built" if state == "missing" else "is older than its source in native/"
    manual = f"{sys.executable} -m pip install {NATIVE_DIR}"
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        log.info("the Rust extension %s; not building it with HF_HUB_OFFLINE=1 (to build it: %s)", what, manual)
        return "offline"
    path = _search_path()
    if shutil.which("cargo", path=path) is None:
        log.info("the Rust extension %s, and Rust isn't installed: searches use the Python code "
                 "(install Rust with rustup to have it built, or start with --no-build to stop this message)", what)
        return "no cargo"
    if not os.access(sysconfig.get_path("platlib"), os.W_OK):
        log.info("the Rust extension %s, and this user can't install into %s (to build it: %s)",
                 what, sysconfig.get_path("platlib"), manual)
        return "read-only"

    def build() -> None:
        global isis_native
        command = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", str(NATIVE_DIR)]
        if shutil.which("nice"):  # searches keep the CPU while it compiles
            command = ["nice", "-n", "10", *command]
        try:
            done = run(command, capture_output=True, text=True, env={**os.environ, "PATH": path})
        except OSError as e:
            log.warning("could not build the Rust extension: %s (to build it: %s)", e, manual)
            return
        if done.returncode != 0:
            tail = "\n".join((done.stderr or done.stdout or "").strip().splitlines()[-15:])
            log.warning("could not build the Rust extension; searches use the Python code (to build it: %s; "
                        "--no-build stops isis from trying at each start)\n%s", manual, tail)
            return
        if state == "outdated":
            log.info("the Rust extension is rebuilt; restart isis to use the new one")
            return
        importlib.invalidate_caches()
        try:
            isis_native = importlib.import_module("isis_native")
        except ImportError as e:
            log.warning("the Rust extension was built but doesn't load: %s", e)
            return
        log.info("the Rust extension is built and in use")

    log.info("the Rust extension %s: building it in the background (a few minutes; searches use the Python code meanwhile)", what)
    threading.Thread(target=build, daemon=True, name="isis-build-native").start()
    return "started"
