import json
import os
import threading
import time

import pytest
from conftest import write_corpus
from fastapi.testclient import TestClient

from isis.cli import allowed_hosts
from isis.engine.index import Engine
from isis.engine.semantic import HashEmbedder
from isis.server.app import PageVersion, create_app
from isis.server.tasks import TaskError, TaskStore


def test_status_files_and_file_content(client):
    status = client.get("/api/status").json()
    assert status["state"] == "ready" and status["files"] == 3 and status["index_bytes"] > 0
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


def test_new_tasks_go_at_the_end_of_the_list(client, corpus):
    names = lambda: [t["name"] for t in client.get("/api/tasks").json()]
    (corpus / "_tasks").mkdir(exist_ok=True)
    for name in ("b", "a"):
        (corpus / "_tasks" / f"{name}.md").write_text("")
    assert names() == ["a", "b"]  # files with no saved order go by name

    for name in ("z", "c"):
        client.post("/api/tasks", json={"name": name})
    assert names() == ["a", "b", "z", "c"]

    client.post("/api/tasks/b/rename", json={"new_name": "y"})
    assert names() == ["a", "y", "z", "c"]

    client.delete("/api/tasks/y")
    (corpus / "_tasks" / "by hand.md").write_text("")
    assert names() == ["a", "z", "c", "by hand"]

    client.post("/api/tasks", json={"name": "y"})  # a new task, not the old one back in its place
    assert names() == ["a", "z", "c", "by hand", "y"]

    (corpus / "_tasks" / ".order.json").write_text("not json")
    assert names() == sorted(names(), key=str.lower)


def test_tasks_can_be_moved(client):
    names = lambda tasks: [t["name"] for t in tasks]
    for name in ("a", "b", "c"):
        client.post("/api/tasks", json={"name": name})
    assert names(client.post("/api/tasks/c/move", json={"before": "a"}).json()) == ["c", "a", "b"]
    assert names(client.post("/api/tasks/c/move", json={"before": None}).json()) == ["a", "b", "c"]
    assert names(client.post("/api/tasks/a/move", json={"before": "c"}).json()) == ["b", "a", "c"]
    assert names(client.get("/api/tasks").json()) == ["b", "a", "c"]
    assert client.post("/api/tasks/nope/move", json={}).status_code == 404


def test_names_cannot_escape_the_tasks_folder(client, corpus):
    name = client.post("/api/tasks", json={"name": "../../evil"}).json()["name"]
    assert (corpus / "_tasks" / f"{name}.md").exists() and "/" not in name


def test_related_terms_and_feedback(client):
    client.post("/api/tasks", json={"name": "t"})
    body = client.post("/api/tasks/t/related", json={"text": "The sperm whale and ambergris."}).json()
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
    (entry,) = prefs["dismissed"]
    assert {k: v for k, v in entry.items() if k != "id"} == {"path": "whales.md", "text": "x", "lane": "core", "start_line": 4, "end_line": 6}
    assert client.get("/api/tasks/t/prefs").json() == prefs
    assert client.post("/api/tasks/t/feedback", json={"action": "add", "path": "whales.md"}).status_code == 400


def test_documents_can_be_excluded_and_included_again(client):
    client.post("/api/tasks", json={"name": "t"})
    body = {"action": "exclude_doc", "path": "whales.md"}  # no lane: it judges the document
    prefs = client.post("/api/tasks/t/feedback", json=body).json()
    assert prefs["excluded_docs"] == ["whales.md"] and prefs["feedback"] == {}
    assert client.get("/api/tasks/t/prefs").json()["excluded_docs"] == ["whales.md"]
    prefs = client.post("/api/tasks/t/feedback", json={"action": "include_doc", "path": "whales.md"}).json()
    assert prefs["excluded_docs"] == []


def test_dismissals_can_be_restored_and_feedback_forgotten(client, corpus):
    client.post("/api/tasks", json={"name": "t"})
    for text in ("x", "y"):
        client.post("/api/tasks/t/feedback", json={"action": "dismiss", "lane": "hidden", "path": "whales.md", "text": text})
    client.post("/api/tasks/t/feedback", json={"action": "add", "lane": "core", "path": "whales.md"})

    prefs = client.post("/api/tasks/t/feedback", json={"action": "restore", "path": "whales.md", "text": " X "}).json()
    assert [d["text"] for d in prefs["dismissed"]] == ["y"]
    assert prefs["feedback"] == {"hidden": [0, 1], "core": [1, 0]}
    prefs = client.post("/api/tasks/t/feedback", json={"action": "restore", "id": prefs["dismissed"][0]["id"]}).json()
    assert prefs["dismissed"] == [] and prefs["feedback"]["hidden"] == [0, 0]

    client.put("/api/tasks/t/steer", json={"text": "whale"})
    assert client.delete("/api/tasks/t/prefs").json() == {"steer": "", "ignored": [], "dismissed": [], "feedback": {}, "liked": [], "excluded_docs": []}
    assert client.get("/api/tasks/t/prefs").json()["steer"] == ""
    assert len(list((corpus / "_tasks" / ".trash").glob("t.*.prefs.json"))) == 1


def test_the_browser_must_check_the_page_is_current(client):
    for path in ("/", "/app.js", "/reflow.js", "/style.css"):
        assert client.get(path).headers["cache-control"] == "no-store", path


def test_steer_line_and_avoided_passages(client):
    client.post("/api/tasks", json={"name": "t"})
    task = {"text": "The sperm whale and ambergris. Ahab hunted the white whale from the Pequod.", "params": {"z_weak": -1}}
    texts = lambda: [h["text"] for lane in client.post("/api/tasks/t/related", json=task).json()["lanes"] for h in lane["items"]]
    assert any("Ambergris" in t for t in texts())

    client.post("/api/tasks/t/terms", json={"term": "ambergris", "action": "ignore"})
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


@pytest.mark.parametrize("damaged", [b'{"steer": "whale', b"\xff\xfe"], ids=["not json", "not utf-8"])
def test_unreadable_preferences_are_kept_aside(corpus, damaged):
    store = TaskStore(corpus / "_tasks")
    store.create("T")
    (corpus / "_tasks" / ".meta").mkdir()
    (corpus / "_tasks" / ".meta" / "T.json").write_bytes(damaged)
    assert store.prefs("T").steer == ""
    (kept,) = (corpus / "_tasks" / ".trash").glob("T.*.bad.json")
    assert kept.read_bytes() == damaged


def test_requests_for_other_hosts_are_refused(engine, corpus):
    app = create_app(engine, TaskStore(corpus / "_tasks"), background=False, watch=False, allowed_hosts=["localhost"])
    with TestClient(app, base_url="http://localhost:8000") as c:
        assert c.get("/api/status").status_code == 200
        assert c.get("/api/status", headers={"host": "attacker.example:8000"}).status_code == 400


def test_allowed_hosts_from_the_command_line():
    assert allowed_hosts("127.0.0.1", []) == ["localhost", "127.0.0.1", "::1"]
    assert allowed_hosts("0.0.0.0", []) == ["*"]
    assert allowed_hosts("0.0.0.0", ["notes.lan"]) == ["localhost", "127.0.0.1", "::1", "notes.lan"]
    assert allowed_hosts("192.168.1.5", []) == ["localhost", "127.0.0.1", "::1", "192.168.1.5"]


def test_dismissed_texts_are_sent_as_previews(client):
    client.post("/api/tasks", json={"name": "t"})
    long = "whale " * 500
    client.post("/api/tasks/t/feedback", json={"action": "dismiss", "lane": "core", "path": "whales.md", "text": long})
    body = client.post("/api/tasks/t/related", json={"text": "The sperm whale."}).json()
    assert len(body["prefs"]["dismissed"][0]["text"]) == 200


def test_saves_wait_for_each_other_and_never_bring_back_a_deleted_task(corpus):
    store = TaskStore(corpus / "_tasks")
    store.create("T")
    mtime = store.read("T")["mtime"]
    outcomes = []

    def save(text):
        try:
            store.write("T", text * 300_000, mtime)
            outcomes.append("saved")
        except Exception as e:
            outcomes.append(type(e).__name__)

    threads = [threading.Thread(target=save, args=(c,)) for c in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["Conflict", "saved"]  # the second save saw the first one's change
    assert not list((corpus / "_tasks").glob(".*.tmp"))

    store.delete("T")
    with pytest.raises(TaskError):
        store.write("T", "late save", None)
    assert not (corpus / "_tasks" / "T.md").exists()


def test_bad_preference_values_are_dropped_not_fatal(client, corpus):
    client.post("/api/tasks", json={"name": "t"})
    (corpus / "_tasks" / ".meta").mkdir(exist_ok=True)
    (corpus / "_tasks" / ".meta" / "t.json").write_text(
        '{"steer": 5, "ignored": "whale", "dismissed": ["x", {"text": "kept", "path": 3}], "feedback": {"core": 5, "gaps": [1, 2]}}')
    prefs = client.get("/api/tasks/t/prefs").json()
    assert (prefs["steer"], prefs["ignored"], prefs["feedback"]) == ("", [], {"gaps": [1, 2]})
    assert [d["text"] for d in prefs["dismissed"]] == ["kept"]
    assert client.post("/api/tasks/t/related", json={"text": "The sperm whale."}).status_code == 200


def test_a_task_that_is_not_utf8_gives_a_clear_error(client, corpus):
    client.post("/api/tasks", json={"name": "t"})
    (corpus / "_tasks" / "t.md").write_bytes("caf\xe9".encode("latin-1"))
    response = client.get("/api/tasks/t")
    assert response.status_code == 422 and "UTF-8" in response.json()["detail"]


def test_no_pages_that_load_from_a_cdn(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_each_panel_can_leave_out_a_folder(tmp_path):
    words = {"whale.md": "The whale and the harpoon.\n", "finch.md": "The finch and its beak.\n", "rock.md": "Granite and basalt.\n"}
    a, b = write_corpus(tmp_path / "a", words), write_corpus(tmp_path / "b", words)
    engine = Engine([a, b], embedder=HashEmbedder(), target_words=0)
    engine.refresh()
    with TestClient(create_app(engine, TaskStore(tmp_path / "tasks"), background=False, watch=False)) as c:
        assert c.get("/api/status").json()["folders"] == ["a", "b"]
        folders = lambda **p: {h["path"].split("/")[0] for h in c.get("/api/search", params={"q": "whale", **p}).json()["hits"]}
        assert folders() == {"a", "b"} and folders(skip='["b"]') == {"a"}
        assert c.get("/api/search", params={"q": "whale", "skip": "nope"}).status_code == 400
        c.post("/api/tasks", json={"name": "t"})
        body = {"text": "The whale and the harpoon.", "params": {"z_weak": -1}}
        shown = lambda **k: {h["path"].split("/")[0] for l in c.post("/api/tasks/t/related", json={**body, **k}).json()["lanes"] for h in l["items"]}
        assert shown() == {"a", "b"} and shown(skip_folders=["b"]) == {"a"}


def test_more_like_this_through_the_api(client):
    client.post("/api/tasks", json={"name": "t"})
    like = {"action": "like", "lane": "core", "path": "whales.md", "text": "x" * 300, "start_line": 2, "end_line": 2}
    prefs = client.post("/api/tasks/t/feedback", json=like).json()
    assert prefs["feedback"] == {"core": [1, 0]}
    (entry,) = prefs["liked"]
    assert len(entry["text"]) == 200 and entry["id"]  # a preview, as for dismissals
    assert client.post("/api/tasks/t/feedback", json={**like, "lane": ""}).status_code == 400
    assert client.post("/api/tasks/t/related", json={"text": "The sperm whale."}).json()["prefs"]["liked"]
    prefs = client.post("/api/tasks/t/feedback", json={"action": "unlike", "id": entry["id"]}).json()
    assert prefs["liked"] == [] and prefs["feedback"] == {"core": [0, 0]}


def test_only_tasks_that_can_be_opened_are_listed(client, corpus):
    client.post("/api/tasks", json={"name": "ok"})
    (corpus / "_tasks" / "C# notes.md").write_text("x")
    (corpus / "_tasks" / ".hidden.md").write_text("x")
    names = [t["name"] for t in client.get("/api/tasks").json()]
    assert "ok" in names and "C# notes" not in names and ".hidden" not in names
    assert all(client.get(f"/api/tasks/{n}").status_code == 200 for n in names)


def test_two_deletions_of_one_name_in_a_second_keep_both_copies(tmp_path, monkeypatch):
    monkeypatch.setattr("isis.server.tasks.time.strftime", lambda _: "same-second")
    store = TaskStore(tmp_path)
    for text in ("first", "second"):
        store.create("dup")
        store.write("dup", text)
        store.delete("dup")
    assert {p.read_text() for p in (tmp_path / ".trash").glob("dup.*.md")} == {"first", "second"}


def test_settings_are_restored_together(client):
    client.post("/api/tasks", json={"name": "t"}).raise_for_status()
    feedback = {"action": "like", "lane": "core", "path": "whales.md", "text": "Ambergris"}
    client.post("/api/tasks/t/feedback", json=feedback).raise_for_status()
    wanted = {"steer": "whale /sp.rm/ -squid", "ignored": ["ship", "ship"], "excluded_docs": ["finches.md"]}
    prefs = client.put("/api/tasks/t/settings", json=wanted).json()
    assert prefs["steer"] == "whale /sp.rm/ -squid" and prefs["ignored"] == ["ship"] and prefs["excluded_docs"] == ["finches.md"]
    assert len(prefs["liked"]) == 1  # what it remembers about single passages stays
    bad = client.put("/api/tasks/t/settings", json={"steer": "/(/", "ignored": [], "excluded_docs": []})
    assert bad.status_code == 400
    prefs = client.get("/api/tasks/t/prefs").json()
    assert prefs["steer"] == "whale /sp.rm/ -squid" and prefs["ignored"] == ["ship"]  # all or nothing
    assert client.put("/api/tasks/missing/settings", json={}).status_code == 404


def test_search_can_hide_documents(client):
    def paths(**params):
        return {hit["path"] for hit in client.get("/api/search", params={"q": "whale", **params}).json()["hits"]}

    assert paths() == {"whales.md", "ships/pequod.md"}
    assert paths(hide=json.dumps(["whales.md"])) == {"ships/pequod.md"}
    assert paths(hide=json.dumps(["whales.md", "ships/pequod.md"])) == set()
    assert paths(hide=json.dumps(["no/such/file.md"])) == {"whales.md", "ships/pequod.md"}
    assert client.get("/api/search", params={"q": "whale", "hide": "{}"}).status_code == 400
    assert client.get("/api/search", params={"q": "whale", "hide": "not json"}).status_code == 400


def test_the_page_carries_the_version_of_its_files(client):
    version = client.get("/api/status").json()["page_version"]
    for path, headers in (("/", {}), ("/index.html", {}), ("/", {"If-None-Match": client.get("/").headers.get("etag", "x")})):
        response = client.get(path, headers=headers)
        assert response.status_code == 200 and f'<meta name="page-version" content="{version}">' in response.text


def test_page_version_waits_for_the_files_to_settle(tmp_path):
    page = tmp_path / "app.js"
    page.write_text("one")
    os.utime(page, (1, 1))
    version = PageVersion(tmp_path, settle=60)
    first = version.current()

    page.write_text("two")  # just saved: more may be on the way
    assert version.current() == first

    os.utime(page, (2, 2))  # still for long enough
    assert version.current() != first


def test_related_for_any_text_stores_nothing(client, corpus):
    body = client.post("/api/related", json={"text": "The sperm whale dives for squid.", "steer": "+squid"}).json()
    items = [item for lane in body["lanes"] for item in lane["items"]]
    assert items and all("squid" in item["text"].lower() for item in items)
    assert "prefs" not in body
    r = client.post("/api/related", json={"text": "whales", "steer": "/(/"})
    assert r.status_code == 400 and "invalid pattern" in r.json()["detail"]
    assert not any((corpus / "_tasks").iterdir())


def test_related_does_not_depend_on_the_task_name(client):
    """The same text gives the same results in every task, as on the command line."""
    text = "The sperm whale dives for squid.\n\nThe finches of the Galapagos differ in their beaks.\n"
    answers = []
    for name in ("a", "Another name"):
        client.post("/api/tasks", json={"name": name})
        answers.append(client.post(f"/api/tasks/{name}/related", json={"text": text}).json()["lanes"])
    assert answers[0] == answers[1] == client.post("/api/related", json={"text": text}).json()["lanes"]
