import threading
import time

import pytest
from fastapi.testclient import TestClient

from isis.server.app import create_app
from isis.server.tasks import TaskStore


@pytest.fixture
def client(engine, corpus):
    store = TaskStore(corpus / "_tasks")
    with TestClient(create_app(engine, store, background=False, watch=False)) as c:
        yield c


def test_status_files_and_file_content(client):
    status = client.get("/api/status").json()
    assert status["state"] == "ready" and status["files"] == 3
    assert [f["path"] for f in client.get("/api/files").json()] == ["finches.md", "ships/pequod.md", "whales.md"]
    assert client.get("/api/file", params={"path": "whales.md"}).json()["content"].startswith("# Whales")


def test_file_access_stays_inside_the_corpus(client):
    assert client.get("/api/file", params={"path": "../outside.md"}).status_code == 400
    assert client.get("/api/file", params={"path": "missing.md"}).status_code == 404


def test_search(client):
    body = client.get("/api/search", params={"q": "ambergris"}).json()
    (hit,) = body["hits"]
    assert hit["path"] == "whales.md" and hit["match"] == [4, 0, 9]
    assert any(line["marks"] for line in hit["snippet"])


def test_task_lifecycle(client, corpus):
    assert client.post("/api/tasks", json={"name": "Whale notes"}).json()["content"] == ""
    assert client.post("/api/tasks", json={"name": "Whale notes"}).status_code == 409
    assert client.post("/api/tasks", json={"name": "  "}).status_code == 400
    assert [t["name"] for t in client.get("/api/tasks").json()] == ["Whale notes"]

    task = client.get("/api/tasks/Whale notes").json()
    saved = client.put("/api/tasks/Whale notes", json={"content": "first", "base_mtime": task["mtime"]})
    assert saved.status_code == 200
    stale = client.put("/api/tasks/Whale notes", json={"content": "second", "base_mtime": task["mtime"]})
    assert stale.status_code == 409 and stale.json()["mtime"] == saved.json()["mtime"]
    assert (corpus / "_tasks" / "Whale notes.md").read_text() == "first"

    assert client.post("/api/tasks/Whale notes/rename", json={"new_name": "Cetology"}).json()["name"] == "Cetology"
    assert client.delete("/api/tasks/Cetology").json() == {"ok": True}
    assert client.get("/api/tasks").json() == []
    assert len(list((corpus / "_tasks" / ".trash").glob("Cetology.*.md"))) == 1


def test_names_cannot_escape_the_tasks_folder(client, corpus):
    name = client.post("/api/tasks", json={"name": "../../evil"}).json()["name"]
    assert (corpus / "_tasks" / f"{name}.md").exists() and "/" not in name


def test_related_terms_and_feedback(client):
    client.post("/api/tasks", json={"name": "t"})
    body = client.post("/api/tasks/t/related", json={"text": "The sperm whale and ambergris.", "cursor_line": 0}).json()
    assert [lane["name"] for lane in body["lanes"]] == ["core", "hidden", "bridges", "gaps"]
    assert "whale" in {c["term"] for c in body["terms"]}

    prefs = client.post("/api/tasks/t/terms", json={"term": "whale", "action": "ignore"}).json()
    assert prefs["ignored"] == ["whale"]
    prefs = client.post("/api/tasks/t/terms", json={"term": "Whale", "action": "boost"}).json()
    assert prefs["steer"] == "Whale" and prefs["ignored"] == []
    prefs = client.post("/api/tasks/t/terms", json={"term": "whale", "action": "unboost"}).json()
    assert prefs["steer"] == ""

    dismissal = {"action": "dismiss", "lane": "core", "path": "whales.md", "text": "x", "start_line": 4, "end_line": 6}
    prefs = client.post("/api/tasks/t/feedback", json=dismissal).json()
    assert prefs["feedback"] == {"core": [0, 1]}
    assert prefs["dismissed"] == [{"path": "whales.md", "text": "x", "lane": "core", "start_line": 4, "end_line": 6}]
    assert client.get("/api/tasks/t/prefs").json() == prefs
    assert client.post("/api/tasks/t/feedback", json={"action": "add", "path": "whales.md"}).status_code == 400


def test_dismissals_can_be_restored_and_feedback_forgotten(client, corpus):
    client.post("/api/tasks", json={"name": "t"})
    for text in ("x", "y"):
        client.post("/api/tasks/t/feedback", json={"action": "dismiss", "lane": "hidden", "path": "whales.md", "text": text})
    client.post("/api/tasks/t/feedback", json={"action": "add", "lane": "core", "path": "whales.md"})

    prefs = client.post("/api/tasks/t/feedback", json={"action": "restore", "path": "whales.md", "text": " X "}).json()
    assert [d["text"] for d in prefs["dismissed"]] == ["y"]
    assert prefs["feedback"] == {"hidden": [0, 1], "core": [1, 0]}

    client.put("/api/tasks/t/steer", json={"text": "whale"})
    assert client.delete("/api/tasks/t/prefs").json() == {"steer": "", "ignored": [], "dismissed": [], "feedback": {}}
    assert client.get("/api/tasks/t/prefs").json()["steer"] == ""
    assert len(list((corpus / "_tasks" / ".trash").glob("t.*.prefs.json"))) == 1


def test_tasks_inside_the_corpus_are_not_indexed(client, engine):
    client.post("/api/tasks", json={"name": "ambergris"})
    client.put("/api/tasks/ambergris", json={"content": "Ambergris ambergris ambergris."})
    engine.exclude.add((engine.root / "_tasks").resolve())
    engine.refresh()
    assert [h["path"] for h in client.get("/api/search", params={"q": "ambergris"}).json()["hits"]] == ["whales.md"]


def test_page_is_served(client):
    assert client.get("/").status_code == 200


def test_steer_line_and_avoided_passages(client):
    client.post("/api/tasks", json={"name": "t"})
    task = {"text": "The sperm whale and ambergris. Ahab hunted the white whale from the Pequod.", "params": {"z_weak": -1}}
    texts = lambda: [h["text"] for lane in client.post("/api/tasks/t/related", json=task).json()["lanes"] for h in lane["items"]]
    assert any("Ambergris" in t for t in texts())

    client.post("/api/tasks/t/terms", json={"term": "ambergri", "action": "ignore"})
    prefs = client.post("/api/tasks/t/terms", json={"term": "Ambergris", "action": "exclude"}).json()
    assert prefs["steer"] == "-Ambergris" and prefs["ignored"] == []  # excluding takes the word out of the ignored
    assert texts() and not any("ambergris" in t.lower() for t in texts())

    prefs = client.put("/api/tasks/t/steer", json={"text": "  +/[Ww]hal/   -/Pequod/ "}).json()
    assert prefs["steer"] == "+/[Ww]hal/ -/Pequod/"  # tidied
    assert texts() and all("hal" in t and "Pequod" not in t for t in texts())
    assert client.put("/api/tasks/t/steer", json={"text": "/[oops/"}).status_code == 400
    assert client.get("/api/tasks/t/prefs").json()["steer"] == "+/[Ww]hal/ -/Pequod/"  # unchanged
    client.put("/api/tasks/t/steer", json={"text": ""})

    first = next(t for t in texts() if "Ambergris" in t)
    body = {"action": "avoid", "lane": "core", "path": "whales.md", "text": first}
    prefs = client.post("/api/tasks/t/feedback", json=body).json()
    assert prefs["dismissed"][0]["avoid"] is True
    assert first not in texts()
    assert client.get("/api/search", params={"q": "/(oops/"}).status_code == 400


def test_preference_changes_do_not_overwrite_each_other(corpus):
    store = TaskStore(corpus / "_tasks")
    store.create("T")

    def click():
        with store.edit_prefs("T") as prefs:
            time.sleep(0.002)  # long enough for the threads to overlap without the lock
            prefs.record("core", added=True)

    threads = [threading.Thread(target=click) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.prefs("T").feedback == {"core": [20, 0]}
    assert not list((corpus / "_tasks" / ".meta").glob(".*.tmp"))


def test_unreadable_preferences_are_kept_aside(corpus):
    store = TaskStore(corpus / "_tasks")
    store.create("T")
    (corpus / "_tasks" / ".meta").mkdir()
    (corpus / "_tasks" / ".meta" / "T.json").write_text('{"steer": "whale', encoding="utf-8")
    assert store.prefs("T").steer == ""
    (kept,) = (corpus / "_tasks" / ".trash").glob("T.*.bad.json")
    assert kept.read_text(encoding="utf-8") == '{"steer": "whale'
