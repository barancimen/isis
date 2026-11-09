"""The watch that refreshes the index. A stand-in for watchfiles drives most of this, because the
real one takes about a second per change; one test checks the real thing."""

import queue
import time

import pytest
from fastapi.testclient import TestClient

from isis.engine.index import Engine
from isis.engine.semantic import HashEmbedder
from isis.server.app import create_app
from isis.server.tasks import TaskStore


def wait_until(check, seconds=15, every=0.01):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(every)
    return False


class FakeWatch:
    """Records each watch started, and reports a change when told to."""

    def __init__(self):
        self.started = []
        self.changes = queue.Queue()

    def __call__(self, *roots, watch_filter, stop_event, debounce):
        self.started.append((roots, watch_filter))
        while not stop_event.is_set():
            try:
                self.changes.get(timeout=0.005)
            except queue.Empty:
                continue
            yield {("modified", "anything")}

    def change(self):
        self.changes.put(True)


@pytest.fixture
def fake_watch(monkeypatch):
    fake = FakeWatch()
    monkeypatch.setattr("watchfiles.watch", fake)
    return fake


def watching(engine, tasks):
    return TestClient(create_app(engine, TaskStore(tasks), background=False, watch=True))


def test_a_change_refreshes_the_index_and_a_new_link_restarts_the_watch(engine, corpus, tmp_path, fake_watch):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "b.md").write_text("Raccoonword, behind a link made while running.\n")
    with watching(engine, corpus / "_tasks"):
        assert wait_until(lambda: fake_watch.started)
        (corpus / "a.md").write_text("Quokkaword, just saved.\n")
        fake_watch.change()
        assert wait_until(lambda: engine.search("quokkaword"))
        assert len(fake_watch.started) == 1  # a plain change keeps the watch going

        (corpus / "new-link").symlink_to(outside)
        fake_watch.change()
        assert wait_until(lambda: len(fake_watch.started) == 2)  # a watch covers the links there when it starts
        assert engine.search("raccoonword")


def test_the_watch_covers_every_folder(tmp_path, fake_watch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    engine = Engine([a, b], embedder=HashEmbedder())
    with watching(engine, tmp_path / "tasks"):
        assert wait_until(lambda: fake_watch.started)
        roots, relevant = fake_watch.started[0]
    assert set(roots) == {a.resolve(), b.resolve()}
    assert relevant("added", str(b / "three.md"))


def test_the_watch_wakes_only_for_what_can_change_the_index(engine, corpus, tmp_path, fake_watch):
    (corpus / "_tasks").mkdir()
    with watching(engine, corpus / "_tasks"):
        assert wait_until(lambda: fake_watch.started)
        relevant = fake_watch.started[0][1]
    wanted = {
        "whales.md": True,  # markdown
        "ships": True,  # a folder, which may have been moved in with many files
        "gone-without-a-trace": True,  # deleted, so it may have been either
        "ships/.no-embed": True,  # hidden, but it changes what gets embedded
        "notes.txt": False,
        ".hidden/secret.md": False,
        ".isis/index.json": False,
        "_tasks/mine.md": False,
    }
    for path, expected in wanted.items():
        assert relevant("modified", str(corpus / path)) is expected, path
    assert not relevant("modified", str(tmp_path / "elsewhere.md"))


def test_a_real_watch_sees_a_file_in_a_linked_folder(engine, corpus, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (corpus / "link").symlink_to(outside)
    engine.refresh()

    def written_and_found():  # written again and again, because the watch takes a moment to start
        (outside / "a.md").write_text("Quokkaword, in a linked folder.\n")
        return engine.search("quokkaword")

    with watching(engine, corpus / "_tasks"):
        assert wait_until(written_and_found, every=0.25)


def test_the_watcher_survives_an_error_and_keeps_checking(engine, corpus, monkeypatch):
    calls = []

    def broken(*_, **__):
        calls.append(1)
        raise OSError("OS file watch limit reached")

    monkeypatch.setattr("watchfiles.watch", broken)
    monkeypatch.setattr("isis.server.app.WATCH_RETRY", 0.01)
    with TestClient(create_app(engine, TaskStore(corpus / "_tasks"), background=False)) as c:
        (corpus / "fresh.md").write_text("Stoatword arrived.\n")
        deadline = time.time() + 20
        while ("fresh.md" not in engine.files or len(calls) < 2) and time.time() < deadline:
            time.sleep(0.02)
        assert "fresh.md" in engine.files and len(calls) > 1
        assert c.get("/api/status").status_code == 200
