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

    prefs = client.post("/api/tasks/t/terms", json={"term": "whale", "action": "ban"}).json()
    assert prefs["bans"] == ["whale"]
    prefs = client.post("/api/tasks/t/terms", json={"term": "whale", "action": "pin"}).json()
    assert prefs["pins"] == ["whale"] and prefs["bans"] == []

    prefs = client.post(
        "/api/tasks/t/feedback", json={"action": "dismiss", "lane": "core", "path": "whales.md", "text": "x"}
    ).json()
    assert prefs["feedback"] == {"core": [0, 1]} and prefs["dismissed"] == [{"path": "whales.md", "text": "x"}]
    assert client.get("/api/tasks/t/prefs").json() == prefs


def test_tasks_inside_the_corpus_are_not_indexed(client, engine):
    client.post("/api/tasks", json={"name": "ambergris"})
    client.put("/api/tasks/ambergris", json={"content": "Ambergris ambergris ambergris."})
    engine.exclude.add((engine.root / "_tasks").resolve())
    engine.refresh()
    assert [h["path"] for h in client.get("/api/search", params={"q": "ambergris"}).json()["hits"]] == ["whales.md"]


def test_page_is_served(client):
    assert client.get("/").status_code == 200
