"""Archives made by osiris: each document's kind, from the manifest and its frontmatter, and leaving
sources out of Search and Related."""

import json
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from isis import cli, client
from isis.engine.index import Engine
from isis.engine.related import RelatedConfig
from isis.engine.osiris import archive_folders, kind_of, lane_of_source
from isis.engine.semantic import HashEmbedder
from isis.server.app import create_app
from isis.server.tasks import TaskStore

from conftest import legacy_cache, write_corpus

# doc id -> (lane in the manifest, source in the frontmatter, text)
DOCS = {
    "aa01": ("youtube", "youtube:transcript:human", "The whale sings a long song in the deep sea."),
    "aa02": ("youtube", "youtube:transcript:auto:scry", "The whale song was recorded near the reef."),
    "bb01": ("hackernews", "hackernews", "A thread about whale tracking with satellites."),
    "bb02": ("reddit", "html:wayback", "A reddit thread about whale watching, fetched from the Wayback Machine."),
    "cc01": ("generic", "pdf:wayback", "A paper on whale migration across the ocean."),
    "cc02": ("generic", "html", "A blog post about whale watching tours."),
}


def doc(source: str, text: str) -> str:
    return f'---\nurl: "https://example.org/{source}"\nsource: "{source}"\n---\n\n{text}\n'


def make_archive(root, docs=DOCS, manifest=True):
    md = root / "md"
    for did, (_, source, text) in docs.items():
        write_corpus(md, {f"{did[:2]}/{did}.md": doc(source, text)})
    if manifest:
        write_manifest(root, docs)
    return root


def write_manifest(root, docs):
    with sqlite3.connect(root / "manifest.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS urls (url TEXT PRIMARY KEY, lane TEXT NOT NULL, doc_id TEXT, status TEXT)")
        db.execute("DELETE FROM urls")
        db.executemany("INSERT INTO urls VALUES (?, ?, ?, 'ok')", [(f"https://x/{d}", lane, d) for d, (lane, _, _) in docs.items()])
        db.execute("INSERT INTO urls VALUES ('https://x/skipped', 'skip', NULL, 'skipped')")
    db.close()


@pytest.fixture
def served(tmp_path):
    notes = write_corpus(tmp_path / "notes", {"n.md": "# Notes\n\nMy own notes on the whale.\n"})
    md, manifest = archive_folders(make_archive(tmp_path / "data"))
    engine = Engine([notes, md], embedder=HashEmbedder(), target_words=0, osiris={md: manifest})
    engine.refresh()
    return engine


def sources_of(engine):
    return {path.rsplit("/", 1)[-1]: source for path, source in engine.snapshot.file_sources.items()}


def test_kinds():
    assert kind_of("youtube", "youtube:transcript:auto:scry") == "youtube:auto"
    assert kind_of("youtube", "youtube:metadata") == "youtube:nocaptions"
    assert kind_of("generic", "pdf:wayback") == "generic:pdf"
    assert kind_of("generic", "html:render") == "generic:web"
    assert kind_of("generic", "doi:openalex") == "generic:doi"
    assert kind_of("localfile", "file:pdf:ocr") == "localfile:pdf"
    assert kind_of("arxiv", "arxiv_abstract") == "arxiv:abstract" and kind_of("arxiv", "pdf:ocr") == "arxiv:full"
    assert kind_of("github", "github:gist") == "github:gist"
    assert kind_of("hackernews", "html:wayback") == "hackernews"  # one kind: whatever the source
    assert kind_of("youtube", "html:render") == "youtube:other"
    assert kind_of("newlane", "x") == "newlane"
    assert lane_of_source("tweet:twitterapi") == "twitter" and lane_of_source("html") == "generic"
    assert lane_of_source("file:epub") == "localfile" and lane_of_source("html_file") == "singlefile"


def test_archive_folders(tmp_path):
    data = make_archive(tmp_path / "data")
    assert archive_folders(data) == archive_folders(data / "md") == (data / "md", data / "manifest.sqlite")
    bare = make_archive(tmp_path / "bare", manifest=False)
    assert archive_folders(bare) == (bare / "md", None)


def test_each_document_has_a_source(served):
    assert sources_of(served) == {
        "aa01.md": "md/youtube:human", "aa02.md": "md/youtube:auto", "bb01.md": "md/hackernews",
        "bb02.md": "md/reddit",  # the manifest knows the lane of a page fetched the Wayback way
        "cc01.md": "md/generic:pdf", "cc02.md": "md/generic:web",
    }
    assert "notes/n.md" not in served.snapshot.file_sources
    # Each source's passages are together, so leaving some out costs little.
    s = served.snapshot
    for rows in s.source_rows().values():
        assert np.all(np.diff(np.sort(rows)) == 1)
    status = {x["id"]: (x["lane"], x["lane_name"], x["name"], x["documents"]) for x in served.sources()}
    assert status["md/youtube:auto"] == ("md/youtube", "YouTube", "Videos with automatic captions", 1)
    assert status["md/hackernews"] == ("md/hackernews", "Hacker News", "Hacker News", 1)
    assert [x["id"] for x in served.sources()] == ["md/generic:web", "md/generic:pdf", "md/youtube:human",
                                                    "md/youtube:auto", "md/reddit", "md/hackernews"]


def test_sources_can_be_left_out(served):
    found = lambda mode, skip: {h.path for h in served.search("whale", mode=mode, skip_folders=skip)}
    for mode in ("lexical", "hybrid"):
        everything = found(mode, [])
        assert len(everything) == 7
        assert found(mode, ["md/youtube"]) == everything - {"md/aa/aa01.md", "md/aa/aa02.md"}
        assert found(mode, ["md/youtube:auto"]) == everything - {"md/aa/aa02.md"}
        assert found(mode, ["md/youtube:auto", "notes"]) == everything - {"md/aa/aa02.md", "notes/n.md"}
        assert found(mode, ["md/nothing", "nope/youtube"]) == everything
    only_youtube = ["notes", "md/generic", "md/reddit", "md/hackernews"]
    assert {h.path for h in served.search("whale", skip_folders=only_youtube)} == {"md/aa/aa01.md", "md/aa/aa02.md"}
    result = served.related("The whale sings its song.", config=RelatedConfig(z_weak=-1), skip_folders=only_youtube)
    shown = {h.path for lane in result.lanes for h in lane.items}
    assert shown and shown <= {"md/aa/aa01.md", "md/aa/aa02.md"}


def test_without_a_manifest_the_source_decides(tmp_path):
    md, manifest = archive_folders(make_archive(tmp_path / "data", manifest=False))
    engine = Engine(md, osiris={md: manifest})
    engine.refresh()
    assert sources_of(engine)["bb02.md"] == "/generic:web"  # served alone, the folder has no name
    assert {h.path for h in engine.search("whale", skip_folders=["/generic"])} == {"aa/aa01.md", "aa/aa02.md", "bb/bb01.md"}


def test_a_changed_manifest_relabels_documents(tmp_path):
    data = make_archive(tmp_path / "data")
    md, manifest = archive_folders(data)
    engine = Engine(md, osiris={md: manifest})
    engine.refresh()
    write_manifest(data, {**DOCS, "bb02": ("generic", "html:wayback", "")})
    engine.refresh()
    assert sources_of(engine)["bb02.md"] == "/generic:web"


def test_sources_are_cached_and_read_for_an_older_cache(tmp_path):
    md, manifest = archive_folders(make_archive(tmp_path / "data"))
    Engine(md, osiris={md: manifest}).refresh()
    cache, = legacy_cache(md / ".isis")
    meta = json.loads(cache.read_text())
    assert {f["path"]: f["source"] for f in meta["files"]}["aa/aa01.md"] == "youtube:transcript:human"
    again = Engine(md, osiris={md: manifest})
    assert sources_of(again)["aa01.md"] == "/youtube:human"  # straight from the cache
    for f in meta["files"]:
        del f["source"]
    cache.write_text(json.dumps(meta))
    older = Engine(md, osiris={md: manifest})
    assert sources_of(older)["aa02.md"] == "/youtube:auto"  # read from the files' frontmatter
    older.refresh()
    legacy_cache(md / ".isis")
    assert all(f.get("source") for f in json.loads(cache.read_text())["files"])  # and kept from now on


def test_status_and_search_through_the_api(served, tmp_path):
    with TestClient(create_app(served, TaskStore(tmp_path / "tasks"), background=False, watch=False)) as c:
        status = c.get("/api/status").json()
        assert status["folders"] == ["notes", "md"]
        assert {s["id"] for s in status["sources"]} == set(served.snapshot.file_sources.values())
        hits = c.get("/api/search", params={"q": "whale", "skip": json.dumps(["notes", "md/generic", "md/youtube"])}).json()["hits"]
        assert {h["path"] for h in hits} == {"md/bb/bb01.md", "md/bb/bb02.md"}


def test_the_client_names_sources(served, tmp_path, monkeypatch, capsys):
    test_client = TestClient(create_app(served, TaskStore(tmp_path / "tasks"), background=False, watch=False))

    def fetch(server, method, path, query=None, body=None):
        r = test_client.request(method, path, params=query, json=body)
        if r.status_code >= 400:
            raise client.Failure(r.json().get("detail"))
        return r.json()

    monkeypatch.setattr(client, "fetch", fetch)
    skip = lambda folders=(), sources=(): set(client.skipped_folders("", list(folders), list(sources)))
    assert skip(sources=["youtube"]) == {"notes", "md/generic:web", "md/generic:pdf", "md/reddit", "md/hackernews"}
    assert skip(sources=["youtube:human", "md/hackernews"]) == {"notes", "md/generic:web", "md/generic:pdf", "md/reddit",
                                                                "md/youtube:auto"}
    assert skip(folders=["notes"], sources=["reddit"]) == {"md/generic:web", "md/generic:pdf", "md/youtube:human",
                                                          "md/youtube:auto", "md/hackernews"}
    assert skip(folders=["md"], sources=["reddit"]) == {"notes"}  # the whole archive is asked for
    with pytest.raises(client.Failure, match="no source named vimeo. The sources are generic, generic:pdf"):
        skip(sources=["vimeo"])
    assert client.main(["search", "whale", "--source", "hackernews", "-l"]) == 0
    assert capsys.readouterr().out.split() == ["md/bb/bb01.md"]


def test_the_command_line_takes_an_archive(tmp_path, monkeypatch):
    data = make_archive(tmp_path / "data")
    notes = write_corpus(tmp_path / "notes", {"n.md": "Notes.\n"})
    seen = {}
    monkeypatch.setattr(cli, "cmd_index", lambda args: seen.update(corpus=args.corpus, osiris=args.osiris))
    cli.main(["index", str(notes), "--osiris", str(data)])
    assert cli.folders_of(seen["corpus"], seen["osiris"]) == [notes, data / "md"]
    cli.main(["index", str(data), str(notes), "--osiris", str(data)])  # named first, it stays first
    assert cli.folders_of(seen["corpus"], seen["osiris"]) == [data / "md", notes]
    assert cli.archives_of([str(data / "md")]) == {data / "md": data / "manifest.sqlite"}
