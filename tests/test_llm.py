"""LLM help for query synthesis, with a stand-in for DeepSeek: no network."""

import json

import pytest
from fastapi.testclient import TestClient

from isis.server.app import create_app
from isis.server.llm import (EXAMPLES, MAX_ATTEMPTS, Budget, QuerySynth, Refused, Unusable, Upstream, check_query, guide,
                             instructions, query_in)
from isis.server.public import kind_of
from isis.server.tasks import TaskStore


def answers(*texts):
    """A stand-in for DeepSeek that gives these answers in turn, and keeps what it was sent."""
    sent = []

    def complete(messages, timeout):
        sent.append([dict(m) for m in messages])
        reply = texts[min(len(sent), len(texts)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    complete.sent = sent
    return complete


def as_json(query):
    return json.dumps({"query": query})


@pytest.mark.parametrize("target", ["search", "related"])
def test_the_examples_the_model_is_shown_are_valid(target):
    for _, query in EXAMPLES[target]:
        assert check_query(query, target) == query


@pytest.mark.parametrize("query, reason", [
    ("", "empty"),
    ("whale\nship", "one line"),
    ("x" * 301, "limit is 300"),
    ('"white whale', "phrase isn't closed"),
    ('whale ""', "phrase is empty"),
    ("whale /harpoon", "isn't closed"),
    ("whale /a/b", "isn't closed"),
    ("whale + ship", "stands alone"),
    ("whale site:", "incomplete"),
    ("whale doc:", "incomplete"),
    ("whale /(/", "invalid pattern"),
    ("-ship -boat", "only of - rules"),
])
def test_queries_the_box_cant_take(query, reason):
    with pytest.raises(Unusable, match=reason):
        check_query(query, "search")


def test_word_rules_may_only_remove():
    assert check_query("-ship -boat", "related") == "-ship -boat"


def test_the_answer_is_a_json_object():
    assert query_in('```json\n{"query": "whale*"}\n```', "search") == "whale*"
    with pytest.raises(Unusable, match="JSON object"):
        query_in("whale*", "search")
    with pytest.raises(Unusable, match='"query" string'):
        query_in('{"q": "whale*"}', "search")


def test_the_model_gets_the_page_guide():
    text = guide("search")
    assert text.startswith("## The basics") and "`whale*`" in text
    assert "Type your query specifications" not in text  # the guide only, not the box above it
    assert guide("related").startswith("## What Related does")
    assert '{"query": "' in instructions("related") and guide("related") in instructions("related")


def test_a_wrong_answer_is_sent_back_with_the_reason():
    complete = answers("not json", as_json('"white whale'), as_json('"white whale" harpoon*'))
    result = QuerySynth("key", complete=complete).synthesize("search", "white whales and harpoons", "old query")
    assert result == {"query": '"white whale" harpoon*', "attempts": 3}
    first, last = complete.sent[0], complete.sent[-1]
    assert first[0]["role"] == "system" and "The search box now holds: old query" in first[1]["content"]
    assert [m["role"] for m in last] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert "phrase isn't closed" in last[-1]["content"]


def test_it_gives_up_after_the_last_attempt():
    complete = answers(as_json("/(/"))
    with pytest.raises(Refused, match=f"failed after {MAX_ATTEMPTS} attempts.*invalid pattern") as e:
        QuerySynth("key", complete=complete).synthesize("search", "anything")
    assert e.value.status == 502 and len(complete.sent) == MAX_ATTEMPTS


def test_deepseek_failures():
    flaky = answers(Upstream("DeepSeek answered HTTP 503", retry=True), as_json("whale"))
    assert QuerySynth("key", complete=flaky).synthesize("search", "whales")["attempts"] == 2
    refused = answers(Upstream("DeepSeek refused the request (HTTP 402)", retry=False), as_json("whale"))
    with pytest.raises(Refused, match="HTTP 402"):
        QuerySynth("key", complete=refused).synthesize("search", "whales")
    assert len(refused.sent) == 1  # not tried again


def test_the_budget_counts_every_call():
    now = [1000.0]
    budget = Budget(per_hour=3, per_day=5, clock=lambda: now[0])
    synth = QuerySynth("key", budget=budget, complete=answers("no", "no", as_json("whale")))
    assert synth.synthesize("search", "whales")["attempts"] == 3
    with pytest.raises(Refused, match="used as much as it may be") as e:
        synth.synthesize("search", "whales")
    assert e.value.status == 429 and 0 < e.value.retry_after <= 3600
    now[0] += 3601
    QuerySynth("key", budget=budget, complete=answers(as_json("whale"))).synthesize("search", "whales")
    QuerySynth("key", budget=budget, complete=answers(as_json("whale"))).synthesize("search", "whales")
    with pytest.raises(Refused) as e:  # five in the day
        QuerySynth("key", budget=budget, complete=answers(as_json("whale"))).synthesize("search", "whales")
    assert e.value.retry_after > 3600


def test_requests_it_refuses():
    synth = QuerySynth("key", complete=answers(as_json("whale")))
    for target, request, status in [("search", "   ", 400), ("search", "x" * 601, 400), ("files", "whales", 400)]:
        with pytest.raises(Refused) as e:
            synth.synthesize(target, request)
        assert e.value.status == status
    with pytest.raises(Refused) as e:
        QuerySynth(None).synthesize("search", "whales")
    assert e.value.status == 503


def test_through_the_api(engine, corpus):
    complete = answers(as_json("whale* -ship*"))
    with TestClient(create_app(engine, TaskStore(corpus / "_tasks"), background=False, watch=False,
                               llm=QuerySynth("key", complete=complete))) as c:
        assert c.get("/api/status").json()["llm"] is True
        r = c.post("/api/llm/query", json={"target": "search", "request": "whales, not ships", "current": "whale"})
        assert r.json() == {"query": "whale* -ship*", "attempts": 1}
        assert c.post("/api/llm/query", json={"target": "nope", "request": "x"}).status_code == 422
    with TestClient(create_app(engine, TaskStore(corpus / "_tasks"), background=False, watch=False)) as c:
        assert c.get("/api/status").json()["llm"] is False
        r = c.post("/api/llm/query", json={"target": "search", "request": "whales"})
        assert r.status_code == 503 and "no DeepSeek key" in r.json()["detail"]


def test_public_mode_limits_it_on_its_own():
    assert kind_of("POST", "/api/llm/query") == "llm"
