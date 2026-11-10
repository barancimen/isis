"""`isis search` and `isis related`, with their requests sent to a test app instead of a server."""

import io
import json
import socket

import pytest
from fastapi.testclient import TestClient

from isis import client
from isis.engine.index import Engine
from isis.engine.semantic import HashEmbedder
from isis.server.app import create_app
from isis.server.tasks import TaskStore

from conftest import FILES, write_corpus

TASK = "The sperm whale dives for squid.\n"
WRAPPED = "# Squid\n\nThe giant squid lives in the deep\nsea, and the sperm whale hunts\nit in the dark.\n"


def serve(engine, tasks_dir, monkeypatch) -> TestClient:
    """Route the client's requests to an app over `engine`, as a running server would answer."""
    test_client = TestClient(create_app(engine, TaskStore(tasks_dir), background=False, watch=False))

    def fetch(server, method, path, query=None, body=None):
        r = test_client.request(method, path, params=query, json=body)
        if r.status_code >= 400:
            raise client.Failure(r.json().get("detail"))
        return r.json()

    monkeypatch.setattr(client, "fetch", fetch)
    return test_client


@pytest.fixture
def corpus(tmp_path):
    return write_corpus(tmp_path / "corpus", {**FILES, "squid.md": WRAPPED})


@pytest.fixture
def app(corpus, tmp_path, monkeypatch):
    engine = Engine(corpus, embedder=HashEmbedder(), index_dir=tmp_path / "index", target_words=0)
    engine.refresh()
    monkeypatch.chdir(tmp_path)
    return serve(engine, corpus / "_tasks", monkeypatch)


def run(capsys, *argv):
    code = client.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_one_line_per_result_with_the_whole_passage(app, capsys):
    code, out, err = run(capsys, "search", "squid")
    assert code == 0 and not err
    lines = out.splitlines()
    assert "[squid.md:3] The giant squid lives in the deep sea, and the sperm whale hunts it in the dark." in lines
    assert all(line.startswith("[") for line in lines)
    assert "…" not in out and "\x1b" not in out  # no ellipses, and no colors outside a terminal


def test_saving_in_three_formats_and_quiet(app, capsys, tmp_path):
    code, out, _ = run(capsys, "search", "squid", "-o", "hits.md", "-q")
    assert code == 0 and out == ""
    assert "the deep sea, and the sperm whale" in (tmp_path / "hits.md").read_text()
    assert "The giant **squid** lives" in (tmp_path / "hits.md").read_text()
    _, out, _ = run(capsys, "search", "squid", "-o", "hits.txt")
    assert (tmp_path / "hits.txt").read_text() == out
    run(capsys, "search", "squid", "-o", "hits.json")
    saved = json.loads((tmp_path / "hits.json").read_text())
    assert saved["hits"][0]["path"] in {"squid.md", "whales.md"}


def test_limit_files_and_json(app, capsys):
    _, out, _ = run(capsys, "search", "whale", "-n", "1")
    assert len(out.splitlines()) == 1
    _, out, _ = run(capsys, "search", "whale", "-l")
    assert len(out.splitlines()) == len(set(out.splitlines())) and all(not line.startswith("[") for line in out.splitlines())
    _, out, _ = run(capsys, "search", "whale", "--json")
    assert json.loads(out)["hits"]


def test_clauses_with_a_dash_are_clauses_not_options(app, capsys, tmp_path):
    _, out, _ = run(capsys, "search", "pequod", "--keywords")
    assert "[ships/pequod.md:3]" in out and "[ships/pequod.md:5]" in out  # 3 is "The Pequod was a whaling ship."
    _, out, _ = run(capsys, "search", "pequod", "-ship", "--keywords")
    assert "[ships/pequod.md:3]" not in out and "[ships/pequod.md:5]" in out
    run(capsys, "search", "whale", "-open")
    assert not (tmp_path / "pen").exists()  # not read as -o pen
    code, out, _ = run(capsys, "search", "zzzqqq", "--keywords")
    assert (code, out) == (1, "")  # nothing found, as in ripgrep


def test_settings_are_checked(app, capsys, tmp_path):
    assert run(capsys, "search", "whale", "-p", '{"k1": 2.0, "b": 0.5}')[0] == 0
    code, _, err = run(capsys, "search", "whale", "-p", '{"kay1": 2}')
    assert code == 2 and "no setting named kay1" in err and "rrf_k" in err
    code, _, err = run(capsys, "search", "whale", "-p", '{"k1": 9}')
    assert code == 2 and "k1 goes from 0.0 to 3.0" in err
    (tmp_path / "s.json").write_text('{"z_strong": 2.0, "query": "paragraphs"}')
    (tmp_path / "notes.md").write_text(TASK)
    assert run(capsys, "related", "notes.md", "-p", "@s.json")[0] in (0, 1)  # read from the file and accepted
    (tmp_path / "s.json").write_text('{"query": "sentences"}')
    code, _, err = run(capsys, "related", "notes.md", "-p", "@s.json")
    assert code == 2 and '"document" or "paragraphs"' in err
    assert run(capsys, "search", "whale", "-p", "[1]")[0] == 2


def test_related_from_a_file_or_stdin_with_word_rules(app, capsys, tmp_path, monkeypatch, corpus):
    (tmp_path / "notes.md").write_text(TASK)
    code, out, _ = run(capsys, "related", "notes.md")
    assert code == 0 and "Best results" in out.splitlines()
    _, out, _ = run(capsys, "related", "notes.md", "+squid", "-n", "1")
    results = [line for line in out.splitlines() if line.startswith("[")]
    assert len(results) == 1 and "squid" in results[0].lower()
    monkeypatch.setattr("sys.stdin", io.StringIO(TASK))
    assert run(capsys, "related", "-")[1] == run(capsys, "related", "notes.md")[1]
    code, _, err = run(capsys, "related", "notes.md", "/(/")
    assert code == 2 and "invalid pattern" in err
    assert not any((corpus / "_tasks").iterdir())  # nothing stored


def test_folders(tmp_path, monkeypatch, capsys):
    a = write_corpus(tmp_path / "notes", {"a.md": "# A\n\nThe whale sings.\n"})
    b = write_corpus(tmp_path / "books", {"b.md": "# B\n\nThe whale dives.\n"})
    engine = Engine([a, b], embedder=HashEmbedder(), index_dir=tmp_path / "index", target_words=0)
    engine.refresh()
    serve(engine, tmp_path / "tasks", monkeypatch)
    _, out, _ = run(capsys, "search", "whale", "--folder", "books")
    assert out.startswith("[books/b.md:3]") and "notes/" not in out
    code, _, err = run(capsys, "search", "whale", "--folder", "music")
    assert code == 2 and "notes, books" in err


def test_no_server(capsys):
    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    code, _, err = run(capsys, "search", "whale", "--server", f"http://127.0.0.1:{port}")
    assert code == 2 and "isis found no server" in err and "isis serve" in err


def test_an_older_server_is_named_as_such(capsys, tmp_path):
    """A server from before `isis related` answers its endpoint with the page's files: 405."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Old(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(405)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"detail": "Method Not Allowed"}')

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Old)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        (tmp_path / "notes.md").write_text(TASK)
        code, _, err = run(capsys, "related", str(tmp_path / "notes.md"), "--server", f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
    assert code == 2 and "older than this command" in err
