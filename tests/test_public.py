"""Public mode: private tasks per visitor, limits, and regexes that can't hang the server."""

import asyncio
import re
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from isis import cli
from isis.engine.search import QueryError, compile_pattern, pattern_time_limit
from isis.server.app import create_app
from isis.server.public import COOKIE, Limits, PublicGuard, RateLimiter, Workspaces, client_address
from isis.server.tasks import TaskStore

pytest.importorskip("regex")
ROOMY = {"search": (1000, 100.0), "related": (1000, 100.0), "write": (1000, 100.0), "read": (1000, 100.0)}
HTTPS = "https://testserver"  # the cookie is Secure, so a plain http client would not send it back


@pytest.fixture
def make_client(engine, tmp_path):
    made = []

    def make(limits=None, *, address="203.0.113.7", share=None):
        workspaces = share or Workspaces(tmp_path / "public", limits or Limits(rates=ROOMY))
        client = TestClient(create_app(engine, workspaces, background=False, watch=False), base_url=HTTPS,
                            headers={"CF-Connecting-IP": address})
        client.__enter__()
        made.append(client)
        client.workspaces = workspaces
        return client

    yield make
    for client in made:
        client.__exit__(None, None, None)


def visitor(client, share, address="203.0.113.8"):
    return TestClient(client.app, base_url=HTTPS, headers={"CF-Connecting-IP": address})


def test_visitors_have_separate_private_tasks(make_client):
    alice = make_client()
    bob = visitor(alice, alice.workspaces)
    alice.post("/api/tasks", json={"name": "mine"}).raise_for_status()
    alice.put("/api/tasks/mine", json={"content": "secret"}).raise_for_status()
    assert [t["name"] for t in alice.get("/api/tasks").json()] == ["mine"]
    assert bob.get("/api/tasks").json() == []
    assert bob.get("/api/tasks/mine").status_code == 404
    assert bob.put("/api/tasks/mine", json={"content": "vandal"}).status_code == 404
    assert bob.delete("/api/tasks/mine").status_code == 404
    assert alice.get("/api/tasks/mine").json()["content"] == "secret"


def test_the_cookie_is_private_and_kept(make_client):
    client = make_client()
    first = client.get("/api/tasks").headers["set-cookie"]
    assert first.startswith(f"{COOKIE}=") and "HttpOnly" in first and "SameSite=lax" in first.replace("Lax", "lax")
    assert "Secure" in first
    assert "set-cookie" not in client.get("/api/tasks").headers
    spoofed = TestClient(client.app, headers={"X-Forwarded-Proto": "http"}).get("/api/status")
    assert "Secure" in spoofed.headers["set-cookie"]  # whatever a request claims


def test_a_made_up_cookie_is_replaced(make_client):
    client = make_client()
    response = client.get("/api/tasks", headers={"Cookie": f"{COOKIE}=../../etc"})
    assert response.status_code == 200 and "set-cookie" in response.headers
    assert ".." not in response.headers["set-cookie"].split(";")[0]


def test_searching_leaves_nothing_on_disk(make_client):
    client = make_client()
    client.get("/api/tasks")
    client.get("/api/search", params={"q": "ambergris"}).raise_for_status()
    assert list(client.workspaces.folder.iterdir()) == []


def test_the_task_folder_does_not_show_the_cookie(make_client):
    client = make_client()
    client.post("/api/tasks", json={"name": "x"}).raise_for_status()
    token = client.cookies[COOKIE]
    (folder,) = client.workspaces.folder.iterdir()
    assert folder.name != token and token not in folder.name


def test_corpus_files_are_limited_to_what_is_indexed(make_client, corpus):
    client = make_client()
    (corpus / "_tasks").mkdir()
    (corpus / "_tasks" / "owner.md").write_text("private")
    assert client.get("/api/file", params={"path": "whales.md"}).status_code == 200
    assert client.get("/api/file", params={"path": "_tasks/owner.md"}).status_code == 404
    assert client.get("/api/file", params={"path": ".hidden/secret.md"}).status_code == 404


def test_there_is_no_way_to_write_to_the_corpus(make_client):
    client = make_client()
    assert client.put("/api/file", params={"path": "whales.md"}, json={"content": "x"}).status_code in (404, 405)
    assert client.post("/api/file", params={"path": "whales.md"}, json={"content": "x"}).status_code in (404, 405)


def test_answers_about_a_visitor_are_never_kept(make_client):
    client = make_client(Limits(rates={**ROOMY, "search": (1, 0.01)}))
    for response in (client.get("/api/tasks"), client.get("/api/search", params={"q": "whale"}),
                     client.get("/api/search", params={"q": "whale"})):  # the last is refused
        assert response.headers["cache-control"] == "no-store" and "cookie" in response.headers["vary"].lower()
    assert client.get("/app.js").headers["cache-control"] == "no-store"


def test_security_headers_are_on_every_answer(make_client):
    client = make_client(Limits(rates={**ROOMY, "search": (1, 0.01)}))
    client.get("/api/search", params={"q": "whale"})
    for response in (client.get("/"), client.get("/api/status"), client.get("/api/search", params={"q": "whale"})):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_a_stored_passage_cannot_overshoot_the_space(make_client):
    client = make_client(Limits(rates=ROOMY, max_workspace_bytes=20_000))
    client.post("/api/tasks", json={"name": "t"}).raise_for_status()
    passage = {"action": "dismiss", "lane": "best", "path": "whales.md"}
    assert client.post("/api/tasks/t/feedback", json={**passage, "text": "x" * 15_000}).status_code == 200
    assert client.post("/api/tasks/t/feedback", json={**passage, "text": "y" * 15_000}).status_code == 413
    (folder,) = client.workspaces.folder.iterdir()
    assert sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) <= 20_000


def test_only_a_saved_task_may_be_a_large_request(make_client):
    client = make_client(Limits(rates=ROOMY, small_request_bytes=500))
    client.post("/api/tasks", json={"name": "t"}).raise_for_status()
    large = {"action": "dismiss", "lane": "best", "path": "whales.md", "text": "x" * 600}
    assert client.post("/api/tasks/t/feedback", json=large).status_code == 413
    assert client.put("/api/tasks/t/steer", json={"text": "x" * 600}).status_code == 413
    assert client.put("/api/tasks/t/settings", json={"steer": "x" * 600}).status_code == 413
    assert client.post("/api/tasks", json={"name": "x" * 600}).status_code == 413
    assert client.put("/api/tasks/t", json={"content": "x" * 600}).status_code == 200


def test_likes_are_capped(make_client):
    client = make_client(Limits(rates=ROOMY, max_liked=2))
    client.post("/api/tasks", json={"name": "t"}).raise_for_status()
    codes = [client.post("/api/tasks/t/feedback", json={"action": "like", "lane": "best", "path": "whales.md", "text": f"passage {i}"}).status_code
             for i in range(3)]
    assert codes == [200, 200, 403]


def test_task_caps(make_client):
    limits = Limits(rates=ROOMY, max_tasks=2, max_task_bytes=100, max_workspace_bytes=150)
    client = make_client(limits)
    client.post("/api/tasks", json={"name": "a"}).raise_for_status()
    client.post("/api/tasks", json={"name": "b"}).raise_for_status()
    assert client.post("/api/tasks", json={"name": "c"}).status_code == 403
    assert client.put("/api/tasks/a", json={"content": "x" * 101}).status_code == 413
    client.put("/api/tasks/a", json={"content": "x" * 100}).raise_for_status()
    assert client.put("/api/tasks/b", json={"content": "x" * 60}).status_code == 413
    client.put("/api/tasks/b", json={"content": "x" * 50}).raise_for_status()
    client.delete("/api/tasks/a").raise_for_status()
    client.post("/api/tasks", json={"name": "c"}).raise_for_status()


def test_deleted_tasks_do_not_pile_up(make_client):
    client = make_client(Limits(rates=ROOMY, max_workspace_bytes=1000))
    for i in range(5):
        client.post("/api/tasks", json={"name": "t"}).raise_for_status()
        client.put("/api/tasks/t", json={"content": "x" * 900}).raise_for_status()
        client.delete("/api/tasks/t").raise_for_status()
    (folder,) = client.workspaces.folder.iterdir()
    assert [p for p in folder.rglob("*") if p.is_file()] == []


def test_the_total_cap_refuses_saves(make_client):
    client = make_client(Limits(rates=ROOMY, max_total_bytes=10))
    client.post("/api/tasks", json={"name": "a"}).raise_for_status()
    client.put("/api/tasks/a", json={"content": "x" * 50}).raise_for_status()
    client.workspaces._totals = (0.0, 0, 0)  # the scan is reused for a while
    response = client.put("/api/tasks/a", json={"content": "y"})
    assert response.status_code == 507
    assert client.get("/api/tasks/a").json()["content"] == "x" * 50


def test_clearing_cookies_does_not_get_round_the_workspace_limit(make_client):
    client = make_client(Limits(rates=ROOMY, new_workspaces_per_hour=2))
    for i in range(3):
        client.cookies.clear()
        response = client.post("/api/tasks", json={"name": "t"})
        assert response.status_code == (429 if i == 2 else 200)
    elsewhere = TestClient(client.app, base_url=HTTPS, headers={"CF-Connecting-IP": "198.51.100.1"})
    elsewhere.post("/api/tasks", json={"name": "t"}).raise_for_status()


def test_idle_workspaces_go_and_used_ones_stay(make_client):
    client = make_client()
    client.post("/api/tasks", json={"name": "t"}).raise_for_status()
    workspaces = client.workspaces
    (folder,) = workspaces.folder.iterdir()
    assert workspaces.remove_idle() == 0
    assert workspaces.remove_idle(now=time.time() + 31 * 86400) == 1
    assert not folder.exists()
    assert client.get("/api/tasks").json() == []
    client.post("/api/tasks", json={"name": "again"}).raise_for_status()  # the same cookie starts over


def test_rate_limits_answer_429_with_retry_after(make_client):
    limits = Limits(rates={**ROOMY, "search": (3, 0.01)})
    client = make_client(limits)
    codes = [client.get("/api/search", params={"q": "whale"}) for _ in range(4)]
    assert [r.status_code for r in codes] == [200, 200, 200, 429]
    assert int(codes[-1].headers["retry-after"]) >= 1
    other = TestClient(client.app, headers={"CF-Connecting-IP": "198.51.100.1"})
    assert other.get("/api/search", params={"q": "whale"}).status_code == 200  # each address has its own
    assert client.get("/api/tasks").status_code == 200  # and each kind of request


def test_a_long_related_request_costs_more(make_client):
    limits = Limits(rates={**ROOMY, "related": (10, 0.01)}, related_bytes_per_token=1024)
    client = make_client(limits)
    client.post("/api/tasks", json={"name": "t"}).raise_for_status()
    assert client.post("/api/tasks/t/related", json={"text": "w " * 3000}).status_code == 200  # costs 6 of 10
    assert client.post("/api/tasks/t/related", json={"text": "w " * 3000}).status_code == 429
    assert client.post("/api/tasks/t/related", json={"text": "w"}).status_code == 200  # a short one still fits


def test_request_sizes(make_client):
    client = make_client(Limits(rates=ROOMY, related_bytes=100, request_bytes=2000, query_bytes=50))
    client.post("/api/tasks", json={"name": "t"}).raise_for_status()
    assert client.post("/api/tasks/t/related", json={"text": "x" * 200}).status_code == 413
    assert client.put("/api/tasks/t", json={"content": "x" * 3000}).status_code == 413
    assert client.get("/api/search", params={"q": "x" * 60}).status_code == 414

    def lying():  # no Content-Length: counted as the largest, then counted as it arrives
        yield b'{"content": "'
        yield b"x" * 3000
        yield b'"}'

    assert client.put("/api/tasks/t", content=lying(), headers={"Content-Type": "application/json"}).status_code == 413


def test_the_address_comes_from_the_proxy():
    scope = {"type": "http", "client": ("127.0.0.1", 1), "headers": [(b"cf-connecting-ip", b"203.0.113.9")]}
    assert client_address(scope) == "203.0.113.9"
    assert client_address({**scope, "headers": []}) == "127.0.0.1"
    ipv6 = [(b"cf-connecting-ip", b"2001:db8:1:2:3:4:5:6")]
    other = [(b"cf-connecting-ip", b"2001:db8:1:2:ffff::1")]
    assert client_address({**scope, "headers": ipv6}) == client_address({**scope, "headers": other})


def test_expensive_work_is_charged_by_the_time_it_took(tmp_path):
    workspaces = Workspaces(tmp_path / "public", Limits(rates=ROOMY, work=(0.1, 0.01)))
    ran = []

    async def slow(scope, receive, send):
        ran.append(scope["path"])
        await asyncio.sleep(0.2)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    async def main():
        transport = httpx.ASGITransport(app=PublicGuard(slow, workspaces))
        async with httpx.AsyncClient(transport=transport, base_url="https://t") as client:
            first = await client.get("/api/search", headers={"CF-Connecting-IP": "203.0.113.1"})
            second = await client.get("/api/search", headers={"CF-Connecting-IP": "203.0.113.1"})
            cheap = await client.get("/api/status", headers={"CF-Connecting-IP": "203.0.113.1"})
            other = await client.get("/api/search", headers={"CF-Connecting-IP": "203.0.113.2"})
            return first, second, cheap, other

    first, second, cheap, other = asyncio.run(main())
    assert (first.status_code, second.status_code, cheap.status_code, other.status_code) == (200, 429, 200, 200)
    assert int(second.headers["retry-after"]) >= 5  # the 0.1 s it overspent, at 0.01 s per second
    assert len(ran) == 3


def test_rate_limiter_refills():
    now = [0.0]
    limiter = RateLimiter(clock=lambda: now[0])
    assert limiter.retry_after("k", 2, 1) == 0 and limiter.retry_after("k", 2, 1) == 0
    assert limiter.retry_after("k", 2, 1) == 1
    now[0] = 1.0
    assert limiter.retry_after("k", 2, 1) == 0


def test_busy_limit_runs_a_few_and_turns_the_rest_away(tmp_path):
    limits = Limits(rates=ROOMY, busy_slots=1, busy_wait=0.15, busy_waiters=2)
    workspaces = Workspaces(tmp_path / "public", limits)
    running, peak = [0], [0]

    async def slow(scope, receive, send):
        running[0] += 1
        peak[0] = max(peak[0], running[0])
        await asyncio.sleep(0.4)
        running[0] -= 1
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    async def main():
        transport = httpx.ASGITransport(app=PublicGuard(slow, workspaces))
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await asyncio.gather(*(client.get("/api/search") for _ in range(5)))

    responses = asyncio.run(main())
    assert peak[0] == 1
    assert sorted(r.status_code for r in responses) == [200, 503, 503, 503, 503]
    assert all(r.json()["detail"].startswith("the server is busy") for r in responses if r.status_code == 503)

    async def again():  # the slot came back
        transport = httpx.ASGITransport(app=PublicGuard(slow, workspaces))
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.get("/api/search")

    assert asyncio.run(again()).status_code == 200


# ---- under a path ----

def test_the_site_can_live_under_a_path(engine, tmp_path):
    workspaces = Workspaces(tmp_path / "public", Limits(rates=ROOMY), base_path="/isis")
    with TestClient(create_app(engine, workspaces, background=False, watch=False), base_url=HTTPS, follow_redirects=False) as client:
        redirect = client.get("/isis", params={"a": "1"})
        assert redirect.status_code == 308 and redirect.headers["location"] == "/isis/?a=1"
        page = client.get("/isis/")
        assert page.status_code == 200 and "<title>isis</title>" in page.text
        assert "Path=/isis;" in page.headers["set-cookie"]
        assert client.get("/isis/app.js").status_code == 200
        assert client.get("/isis/api/search", params={"q": "ambergris"}).json()["hits"]
        assert client.post("/isis/api/tasks", json={"name": "t"}).status_code == 200
        for elsewhere in ("/", "/api/status", "/isisx/", "/other/isis/"):
            assert client.get(elsewhere).status_code == 404


def test_the_page_uses_relative_addresses():
    from isis.server.app import WEB

    source = (WEB / "app.js").read_text(encoding="utf-8")
    assert "/api/" not in re.sub(r"//.*", "", source)


@pytest.mark.parametrize("flags", [["--base-path", "/isis"], ["--public", "--base-path", "isis/"]])
def test_a_base_path_needs_public_mode_and_a_clean_form(corpus, tmp_path, flags):
    with pytest.raises(SystemExit) as e:
        cli.main(["serve", str(corpus), "--allow-host", "x.example", "--tasks", str(tmp_path / "out"), *flags])
    assert "--base-path" in str(e.value)


# ---- regexes ----

def test_without_a_limit_patterns_stay_plain_re():
    assert isinstance(compile_pattern("whale"), re.Pattern)


def test_a_slow_pattern_stops_in_time():
    text = "a" * 3000
    with pattern_time_limit(0.2):
        pattern = compile_pattern(r"(a+)+$b")
        started = time.monotonic()
        with pytest.raises(QueryError, match="took too long"):
            list(pattern.finditer(text))
    assert time.monotonic() - started < 2


def test_all_patterns_of_a_request_share_the_time():
    text = "a" * 3000
    with pattern_time_limit(0.2):
        first, second = compile_pattern(r"(a+)+$b"), compile_pattern(r"(a*)*\d")
        started = time.monotonic()
        with pytest.raises(QueryError):
            list(first.finditer(text))
        with pytest.raises(QueryError):
            list(second.finditer(text))
    assert time.monotonic() - started < 2


@pytest.mark.parametrize("pattern", [r"(x{65535}){65535}", r"((a{50}){50}){50}", r"a{3000}", "x" * 301])
def test_patterns_costly_to_compile_are_refused(pattern):
    with pattern_time_limit(1), pytest.raises(QueryError, match="too"):
        compile_pattern(pattern)


@pytest.mark.parametrize("pattern,text,expected", [
    (r"whale", "The WHALE and the whale", ["WHALE", "whale"]),
    (r"Whale", "The WHALE and the Whale", ["Whale"]),
    (r"(?<=sperm )whale\b", "sperm whale whaler", ["whale"]),
    (r"(\w)\1", "the ship sees all", ["ee", "ll"]),
    (r"(?i)ahab|ishmael", "Ahab met ISHMAEL", ["Ahab", "ISHMAEL"]),
    (r"\b\w{5}\b", "ships sail on white seas", ["ships", "white"]),
    (r"^#+ ", "# one", ["# "]),
])
def test_timed_patterns_agree_with_plain_ones(pattern, text, expected):
    plain = [m.group() for m in compile_pattern(pattern).finditer(text)]
    with pattern_time_limit(1):
        timed = [m.group() for m in compile_pattern(pattern).finditer(text)]
    assert plain == timed == expected


@pytest.mark.parametrize("pattern", [
    r"(?r)x", r"\p{L}+", r"(a)(?1)", r"(?:foo){e<=1}", r"(?:foo){e}", r"(?:foo){2i+2d+2s<=4}", r"(?:foo){s<=2,e<=3}", r"(?:foo){1<=e<=3}",
])
def test_regex_only_syntax_stays_invalid(pattern):
    with pattern_time_limit(1), pytest.raises(QueryError, match="invalid pattern"):
        compile_pattern(pattern)


def test_search_and_related_report_a_slow_pattern(make_client, monkeypatch):
    client = make_client()
    client.post("/api/tasks", json={"name": "t"}).raise_for_status()
    client.put("/api/tasks/t/steer", json={"text": "+/whale/"}).raise_for_status()
    monkeypatch.setattr("isis.server.app.PATTERN_SECONDS", 0.0)  # no time at all
    other = make_client(share=client.workspaces)
    other.cookies = client.cookies
    response = other.get("/api/search", params={"q": "+/ambergris/"})
    assert response.status_code == 400 and "took too long" in response.json()["detail"]
    response = other.post("/api/tasks/t/related", json={"text": "The white whale."})
    assert response.status_code == 400 and "took too long" in response.json()["detail"]


def test_regex_search_works_in_public_mode(make_client):
    client = make_client()
    body = client.get("/api/search", params={"q": r"+/ambergris/"}).json()
    assert [hit["path"] for hit in body["hits"]] == ["whales.md"]
    assert any(line["marks"] for line in body["hits"][0]["snippet"])


def test_a_steer_line_with_a_costly_pattern_is_refused(make_client):
    client = make_client()
    client.post("/api/tasks", json={"name": "t"}).raise_for_status()
    assert client.put("/api/tasks/t/steer", json={"text": r"/(x{65535}){65535}/"}).status_code == 400


def test_local_mode_has_none_of_this(engine, corpus):
    store = TaskStore(corpus / "_tasks")
    with TestClient(create_app(engine, store, background=False, watch=False)) as client:
        response = client.get("/api/status")
        assert "set-cookie" not in response.headers
        for _ in range(3):
            client.post("/api/tasks", json={"name": f"t{_}"}).raise_for_status()
        client.put("/api/tasks/t0", json={"content": "x" * (2 * 1024 * 1024)}).raise_for_status()
        client.delete("/api/tasks/t0").raise_for_status()
        assert any((corpus / "_tasks" / ".trash").iterdir())  # still kept
        assert client.get("/api/file", params={"path": "_tasks/t1.md"}).status_code == 200


# ---- the command line ----

@pytest.mark.parametrize("flags,message", [
    (["--host", "0.0.0.0", "--allow-host", "x.example", "--tasks", "{outside}"], "loopback"),
    (["--tasks", "{outside}"], "--allow-host"),
    (["--allow-host", "x.example"], "--tasks"),
    (["--allow-host", "x.example", "--tasks", "{inside}"], "outside the corpus"),
])
def test_public_mode_refuses_unsafe_settings(corpus, tmp_path, flags, message):
    flags = [f.format(outside=tmp_path / "out", inside=corpus / "_tasks") for f in flags]
    with pytest.raises(SystemExit) as e:
        cli.main(["serve", str(corpus), "--public", *flags])
    assert message in str(e.value)
