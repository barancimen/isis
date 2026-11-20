"""The line at the bottom of the terminal while `isis serve` runs: the address, the memory isis
holds, and how much it has read from and written to disk since it started. Messages are printed
above it, and it is drawn again every few seconds. Only in a terminal, and only where Linux's
/proc tells these numbers; elsewhere just the address."""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading

EVERY = 5.0  # seconds between updates
GB = 1024**3

_lock = threading.Lock()  # the line and the messages, which other threads write too


def _fields(path: str, sep: str) -> dict[str, int]:
    out = {}
    with open(path) as f:
        for line in f:
            name, _, value = line.partition(sep)
            parts = value.split()
            if parts and parts[0].isdigit():
                out[name.strip()] = int(parts[0])
    return out


def _gb(n: float) -> str:
    return f"{n / GB:.1f} GB" if n >= GB else f"{n / GB:.2f} GB"


def usage() -> str:
    """Memory, disk read and disk written, as the line shows them. Memory is what isis holds itself
    (in RAM or swapped out), not the parts of its cache files the system keeps in RAM, which it can
    take back at any time. Empty where /proc doesn't tell them."""
    try:
        status = _fields("/proc/self/status", ":")
        io = _fields("/proc/self/io", ":")
    except OSError:
        return ""
    memory = (status.get("RssAnon", 0) + status.get("VmSwap", 0)) * 1024  # in kB there
    return (f"memory {_gb(memory)}, disk read {_gb(io.get('read_bytes', 0))}, "
            f"disk written {_gb(io.get('write_bytes', 0))}")


class StatusLine:
    def __init__(self, url: str, stream=None):
        self.url = url
        self.stream = stream or sys.stderr
        self.shown = False

    def text(self) -> str:
        line = f"isis: {self.url}"
        if stats := usage():
            line += f"   {stats}"
        try:
            width = os.get_terminal_size(self.stream.fileno()).columns
        except (OSError, ValueError):
            width = 0  # not known: as it is
        return line[:width - 1] if width > 1 else line  # one row, or redrawing it would leave the rest behind

    def draw(self) -> None:
        """Draw the line again. The caller holds _lock."""
        self.stream.write("\r\x1b[K" + self.text())
        self.stream.flush()
        self.shown = True

    def clear(self) -> None:
        if self.shown:
            self.stream.write("\r\x1b[K")
            self.shown = False

    def keep(self) -> None:
        """At exit: leave the last line where it is, with the shell prompt below it."""
        with _lock:
            if self.shown:
                self.stream.write("\n")
                self.stream.flush()
                self.shown = False


class _Handler(logging.StreamHandler):
    """Prints each message above the line."""

    def __init__(self, line: StatusLine):
        super().__init__(line.stream)
        self.line = line

    def emit(self, record: logging.LogRecord) -> None:
        with _lock:
            self.line.clear()
            super().emit(record)
            self.line.draw()


def start(url: str) -> StatusLine | None:
    """Show the line, and print messages above it, if isis is running in a terminal."""
    stream = sys.stderr
    if not stream.isatty():
        return None
    line = StatusLine(url, stream)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = _Handler(line)
    handler.setFormatter(logging.Formatter("isis: %(message)s"))
    root.addHandler(handler)
    logging.captureWarnings(True)  # so they too go above the line
    atexit.register(line.keep)

    def update() -> None:
        while not stop.wait(EVERY):
            with _lock:
                line.draw()

    stop = threading.Event()  # never set: the thread ends with the process
    with _lock:
        line.draw()
    threading.Thread(target=update, daemon=True, name="isis-status").start()
    return line
