"""Checks, in headless Chromium, of mistakes the page once made, each named by what must hold:
the first task keeps the search on show, a failed save is tried again, a highlight after an emoji
is on its word, Related never acts on another task, and so on.

    pip install playwright && playwright install chromium
    isis serve sample_corpus --tasks /tmp/isis-check-tasks --port 8765     # an EMPTY tasks folder
    python scripts/browser_regressions.py http://127.0.0.1:8765 sample_corpus /tmp/isis-check-tasks [check ...]

It makes and deletes tasks freely, so it refuses to run on a server that has any. It writes one
file into the corpus for a moment (e2e-emoji.md). The searches expect the sample corpus, served
with a meaning model.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

URL = sys.argv[1].rstrip("/")
CORPUS = Path(sys.argv[2])
TASKS = Path(sys.argv[3])
ONLY = sys.argv[4:]
results = []


def api(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(URL + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as r:
        return json.loads(r.read())


def clear_tasks():
    for task in api("GET", "/api/tasks"):
        api("DELETE", "/api/tasks/" + urllib.parse.quote(task["name"]))


def check(name):
    def wrap(fn):
        fn.check_name = name
        return fn
    return wrap


def task_text(page):
    return page.evaluate("window.isis.taskView.state.doc.toString()")


def ready(page):
    page.goto(URL)
    page.wait_for_selector("#statusbar[data-state=ready]", timeout=30000)


@check("first task keeps the open document and the search")
def first_task(page, answers):
    clear_tasks()
    ready(page)
    page.click(".side-tabs button[data-panel=search]")
    page.fill("#search-input", '"white whale"')
    page.wait_for_selector("#search-results .hit", timeout=20000)
    page.locator("#search-results .hit .hit-body").first.click()
    page.wait_for_selector("#reader-editor .cm-hit-range", timeout=10000)
    path = page.locator("#reader-path").inner_text()
    answers.append("first")
    page.locator("#search-results .hit button.add").first.click()
    expect(page.locator(".task-tab.active")).to_contain_text("first")
    page.wait_for_timeout(500)
    assert page.locator("#reader-path").inner_text() == path, page.locator("#reader-path").inner_text()
    assert page.input_value("#search-input") == '"white whale"', page.input_value("#search-input")
    assert page.locator("#search-results .hit").count() > 0
    assert "whale" in task_text(page).lower()
    # a second task starts with an empty reader and search, as before
    answers.append("second")
    page.click("#new-task")
    expect(page.locator(".task-tab.active")).to_contain_text("second")
    page.wait_for_timeout(300)
    assert page.locator("#reader-path").inner_text() == "No document open"
    assert page.input_value("#search-input") == ""
    page.locator(".task-tab", has_text="first").locator(".name").click()
    expect(page.locator("#reader-path")).to_have_text(path)
    assert page.input_value("#search-input") == '"white whale"'


@check("ending a rename by clicking into the task keeps the focus there")
def rename_blur(page, answers):
    clear_tasks()
    api("POST", "/api/tasks", {"name": "old name"})
    ready(page)
    page.locator(".task-tab .name").first.dblclick()
    page.fill(".task-tab input.rename", "new name")
    page.click("#task-editor .cm-content")
    page.keyboard.type("typed after the rename")
    page.wait_for_timeout(700)
    expect(page.locator(".task-tab.active")).to_contain_text("new name")
    assert page.evaluate("document.activeElement.closest('#task-editor') !== null"), page.evaluate("document.activeElement.outerHTML.slice(0, 80)")
    assert task_text(page) == "typed after the rename", task_text(page)
    # Enter still gives the focus to the tab, under its new name
    page.locator(".task-tab .name").first.dblclick()
    page.fill(".task-tab input.rename", "third name")
    page.keyboard.press("Enter")
    page.wait_for_timeout(500)
    assert page.evaluate("document.activeElement.textContent") == "third name", page.evaluate("document.activeElement.outerHTML.slice(0, 80)")
    expect(page.locator("#save-state")).to_have_attribute("data-state", "saved", timeout=8000)
    assert (TASKS / "third name.md").read_text() == "typed after the rename"


@check("a save that fails is tried again by itself")
def save_retry(page, answers):
    clear_tasks()
    api("POST", "/api/tasks", {"name": "retry"})
    ready(page)
    failing = {"on": True}
    page.route("**/api/tasks/retry", lambda route: route.abort() if failing["on"] and route.request.method == "PUT" else route.continue_())
    page.click("#task-editor .cm-content")
    page.keyboard.type("words that must not be lost")
    expect(page.locator("#save-state")).to_have_attribute("data-state", "failed", timeout=5000)
    flash = page.locator("#flash").inner_text()
    assert "can't be reached" in flash, flash
    failing["on"] = False
    expect(page.locator("#save-state")).to_have_attribute("data-state", "saved", timeout=8000)  # with no key pressed
    assert (TASKS / "retry.md").read_text() == "words that must not be lost"


@check("the changed-on-disk banner follows its task, and a rename doesn't switch the check off")
def conflict(page, answers):
    clear_tasks()
    api("POST", "/api/tasks", {"name": "one"})
    api("POST", "/api/tasks", {"name": "two"})
    ready(page)
    page.locator(".task-tab", has_text="one").locator(".name").click()
    page.click("#task-editor .cm-content")
    page.keyboard.type("mine")
    expect(page.locator("#save-state")).to_have_attribute("data-state", "saved", timeout=8000)
    time.sleep(0.05)
    (TASKS / "one.md").write_text("changed elsewhere")
    page.keyboard.type(" more")
    expect(page.locator("#conflict")).to_be_visible(timeout=5000)
    page.locator(".task-tab", has_text="two").locator(".name").click()
    expect(page.locator("#conflict")).to_be_hidden()
    expect(page.locator("#save-state")).to_have_attribute("data-state", "saved")
    page.locator(".task-tab", has_text="one").locator(".name").click()
    expect(page.locator("#conflict")).to_be_visible()
    expect(page.locator("#save-state")).to_have_attribute("data-state", "unsaved")
    page.click("#conflict-keep")
    expect(page.locator("#conflict")).to_be_hidden(timeout=5000)
    expect(page.locator("#save-state")).to_have_attribute("data-state", "saved", timeout=8000)
    assert (TASKS / "one.md").read_text() == "mine more"
    # changed elsewhere, then renamed here: the next edit must still ask
    time.sleep(0.05)
    (TASKS / "one.md").write_text("changed elsewhere again")
    page.locator(".task-tab", has_text="one").locator(".name").dblclick()
    page.fill(".task-tab input.rename", "uno")
    page.keyboard.press("Enter")
    expect(page.locator(".task-tab.active")).to_contain_text("uno")
    page.click("#task-editor .cm-content")
    page.keyboard.type("!")
    expect(page.locator("#conflict")).to_be_visible(timeout=5000)
    assert (TASKS / "uno.md").read_text() == "changed elsewhere again"
    page.click("#conflict-load")
    expect(page.locator("#conflict")).to_be_hidden(timeout=5000)
    assert task_text(page) == "changed elsewhere again"


@check("Enter in a numbered list leaves other lists and prose alone")
def list_enter(page, answers):
    clear_tasks()
    api("POST", "/api/tasks", {"name": "lists"})
    ready(page)
    text = "1. one\n2. two\n\nThe company began in\n1987. It grew quickly.\n\n1. other\n2. list\n\n- [ ] box"
    page.evaluate("""(text) => { const v = window.isis.taskView;
        v.dispatch({changes: {from: 0, to: v.state.doc.length, insert: text}, selection: {anchor: 6}}); v.focus(); }""", text)
    page.keyboard.press("Enter")
    page.keyboard.type("new")
    assert task_text(page) == "1. one\n2. new\n3. two\n\nThe company began in\n1987. It grew quickly.\n\n1. other\n2. list\n\n- [ ] box", task_text(page)
    end = len(task_text(page))
    page.evaluate("(at) => window.isis.taskView.dispatch({selection: {anchor: at}})", end)
    page.keyboard.press("Enter")
    page.keyboard.type("next")
    assert task_text(page).endswith("- [ ] box\n- [ ] next"), task_text(page)[-40:]


@check("an error always says something")
def error_text(page, answers):
    clear_tasks()
    ready(page)
    page.click(".side-tabs button[data-panel=search]")
    page.route("**/api/search*", lambda route: route.fulfill(status=502, content_type="text/html", body="<html>Bad gateway</html>"))
    page.fill("#search-input", "whale")
    expect(page.locator("#search-status")).to_contain_text("502", timeout=5000)  # a proxy's page has no message of its own
    page.unroute("**/api/search*")
    page.route("**/api/search*", lambda route: route.fulfill(status=422, content_type="application/json",
                                                               body=json.dumps({"detail": [{"msg": "Input should be a valid integer", "loc": ["query", "limit"]}]})))
    page.fill("#search-input", "whales")
    expect(page.locator("#search-status")).to_contain_text("Input should be a valid integer", timeout=5000)
    page.unroute("**/api/search*")
    page.route("**/api/search*", lambda route: route.abort())
    page.fill("#search-input", "whale ship")
    expect(page.locator("#search-status")).to_contain_text("can't be reached", timeout=5000)
    page.unroute("**/api/search*")
    # elsewhere an error is a message at the bottom
    answers.append("named")
    page.route("**/api/tasks", lambda route: route.fulfill(status=504, content_type="text/html", body="") if route.request.method == "POST" else route.continue_())
    page.click("#new-task")
    expect(page.locator("#flash")).to_contain_text("504", timeout=5000)
    page.unroute("**/api/tasks")


@check("a highlight after an emoji is on the word itself")
def emoji_marks(page, answers):
    sample = CORPUS / "e2e-emoji.md"
    sample.write_text("# Emoji\n\nThe \U0001F40B whale and the \U0001F4F7Bellwether counties, twice \U0001F40B\U0001F40B the bellwether.\n")
    try:
        for _ in range(60):  # until the index has it
            if any(h["path"] == sample.name for h in api("GET", "/api/search?q=bellwether")["hits"]):
                break
            time.sleep(2)
        clear_tasks()
        ready(page)
        page.click(".side-tabs button[data-panel=search]")
        page.fill("#search-input", "bellwether +counties")
        card = page.locator("#search-results .hit", has_text="counties").first
        expect(card).to_be_visible(timeout=20000)
        marks = card.locator("mark").all_inner_texts()
        assert marks == ["Bellwether", "counties", "bellwether"], marks
        card.locator(".hit-body").click()
        expect(page.locator("#reader-editor .cm-hit-match").first).to_be_visible(timeout=10000)
        in_reader = page.locator("#reader-editor .cm-hit-match").all_inner_texts()
        assert in_reader == ["Bellwether", "counties", "bellwether"], in_reader
    finally:
        sample.unlink()


@check("a search the server couldn't run says so in place, and waits out a short wait")
def search_errors(page, answers):
    clear_tasks()
    ready(page)
    page.click(".side-tabs button[data-panel=search]")
    busy = {"left": 1}

    def handle(route):
        if busy["left"]:
            busy["left"] -= 1
            return route.fulfill(status=503, content_type="application/json", headers={"Retry-After": "1"},
                                 body=json.dumps({"detail": "the server is busy. Try again in a moment"}))
        return route.continue_()

    page.route("**/api/search*", handle)
    page.fill("#search-input", "whale")
    expect(page.locator("#search-status")).to_contain_text("Searching again in 1 s", timeout=5000)
    expect(page.locator("#search-results .hit").first).to_be_visible(timeout=10000)  # by itself
    page.unroute("**/api/search*")
    page.route("**/api/search*", lambda route: route.fulfill(status=429, content_type="application/json", headers={"Retry-After": "120"},
                                                               body=json.dumps({"detail": "too many requests from this address. Wait a moment and try again"})))
    page.fill("#search-input", "whale ship")
    expect(page.locator("#search-status")).to_have_text("⚠ Too many requests from this address. Wait a moment and try again", timeout=5000)
    page.wait_for_timeout(3000)
    expect(page.locator("#search-status")).to_contain_text("Too many requests")  # still there, not a passing flash
    assert page.locator("#search-results .hit").count() == 0
    page.unroute("**/api/search*")


TEXT_A = "The sperm whale dives deep to hunt giant squid, and the whalers chase it with harpoons across the ocean.\n"
TEXT_B = "Natural selection preserves the variations that help a finch survive on the islands of the Galapagos.\n"


@check("Related never shows one task's keywords and rules while another is open")
def related_switch(page, answers):
    clear_tasks()
    for name, text in (("A", TEXT_A), ("B", TEXT_B)):
        api("POST", "/api/tasks", {"name": name})
        api("PUT", f"/api/tasks/{name}", {"content": text})
    api("PUT", "/api/tasks/A/steer", {"text": "harpoon -ship"})
    ready(page)
    page.locator(".task-tab", has_text="A").locator(".name").click()
    page.click(".side-tabs button[data-panel=related]")
    expect(page.locator("#related-terms .chip").first).to_be_visible(timeout=30000)
    expect(page.locator("#steer-input")).to_have_value("harpoon -ship")
    page.route("**/api/tasks/B/related", lambda route: route.fulfill(status=503, content_type="application/json",
                                                                       body=json.dumps({"detail": "the server is busy. Try again in a moment"})))
    page.locator(".task-tab", has_text="B").locator(".name").click()
    expect(page.locator("#related-status")).to_contain_text("The server is busy", timeout=10000)
    assert page.locator("#related-terms .chip").count() == 0
    assert page.locator("#related-lanes .hit").count() == 0
    expect(page.locator("#related-steer")).to_be_hidden()
    expect(page.locator("#related-refresh")).to_be_enabled()
    page.unroute("**/api/tasks/B/related")
    page.click("#related-refresh")
    expect(page.locator("#related-terms .chip").first).to_be_visible(timeout=30000)
    expect(page.locator("#steer-input")).to_have_value("")
    assert api("GET", "/api/tasks/B/prefs")["steer"] == "" and api("GET", "/api/tasks/A/prefs")["steer"] == "harpoon -ship"


@check("a suggestion hidden at once comes back when the server refuses")
def hide_refused(page, answers):
    clear_tasks()
    api("POST", "/api/tasks", {"name": "A"})
    api("PUT", "/api/tasks/A", {"content": TEXT_A})
    ready(page)
    page.click(".side-tabs button[data-panel=related]")
    expect(page.locator("#related-lanes .hit").first).to_be_visible(timeout=30000)
    before = page.locator("#related-lanes .hit").count()
    first = page.locator("#related-lanes .hit").first.get_attribute("data-key")
    page.route("**/api/tasks/A/feedback", lambda route: route.fulfill(status=413, content_type="application/json",
                                                                        body=json.dumps({"detail": "your tasks are using all their space"})))
    page.locator("#related-lanes .hit .hit-menu-btn").first.click()
    page.locator(".hit-menu .menu-item", has_text="Hide this passage").click()
    expect(page.locator("#flash")).to_contain_text("using all their space", timeout=5000)
    assert page.locator("#related-lanes .hit").count() == before
    assert page.locator("#related-lanes .hit").first.get_attribute("data-key") == first  # back in its place
    page.unroute("**/api/tasks/A/feedback")
    page.locator("#related-lanes .hit .hit-menu-btn").first.click()
    page.locator(".hit-menu .menu-item", has_text="Hide this passage").click()
    expect(page.locator("#related-lanes .hit")).to_have_count(before - 1, timeout=5000)


@check("sliders: one heading per group, and a default that the default button restores")
def sliders(page, answers):
    clear_tasks()
    api("POST", "/api/tasks", {"name": "A"})
    ready(page)
    page.click(".side-tabs button[data-panel=search]")
    page.locator("#search-tune > summary").click()
    groups = page.locator("#search-tune .tune-group").all_inner_texts()
    assert len(groups) == len(set(groups)) == 2, groups
    page.click(".side-tabs button[data-panel=related]")
    page.locator("#related-tune > summary").click()
    row = page.locator("#related-tune [data-name=z_strong]")
    assert row.locator("output").inner_text() == "3.25", row.locator("output").inner_text()
    row.locator("input").focus()
    page.keyboard.press("ArrowRight")
    assert row.locator("output").inner_text() == "3.30"
    expect(page.locator("#related-tune .tune-count")).to_have_attribute("data-changed", "1")
    row.locator(".tune-default").click()
    assert row.locator("output").inner_text() == "3.25"
    expect(page.locator("#related-tune .tune-count")).to_have_attribute("data-changed", "0")
    # a built-in preset whose values lie between the old steps
    page.select_option("#related-preset", label="Close matches")
    assert row.locator("output").inner_text() == "3.75", row.locator("output").inner_text()
    assert page.evaluate("window.isis.state.params.related.z_strong") == 3.75
    page.locator("#related-tune .tune-reset").click()


@check("a result's title opens its document even when a Files filter hides it")
def title_clears_filter(page, answers):
    clear_tasks()
    ready(page)
    page.click(".side-tabs button[data-panel=files]")
    page.fill("#file-filter", "zzz-nothing")
    page.click(".side-tabs button[data-panel=search]")
    page.fill("#search-input", '"white whale"')
    page.wait_for_selector("#search-results .hit", timeout=20000)
    page.locator("#search-results .hit .hit-title").first.click()
    expect(page.locator("#reader-path")).not_to_have_text("No document open", timeout=10000)
    page.click(".side-tabs button[data-panel=files]")
    assert page.input_value("#file-filter") == ""
    expect(page.locator("#file-list .file.current")).to_be_visible()


CHECKS = [first_task, rename_blur, save_retry, conflict, list_enter, error_text,
          emoji_marks, search_errors, related_switch, hide_refused, sliders, title_clears_filter]


def main():
    if api("GET", "/api/tasks"):
        sys.exit("this server has tasks, and these checks delete every task: start one with an empty --tasks folder")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for fn in CHECKS:
            if ONLY and fn.__name__ not in ONLY:
                continue
            context = browser.new_context(viewport={"width": 1400, "height": 900})
            page = context.new_page()
            errors, answers = [], []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("dialog", lambda d: d.accept(answers.pop(0)) if d.type == "prompt" else d.accept())
            try:
                fn(page, answers)
                assert not errors, errors
                results.append(("ok", fn.check_name, ""))
            except Exception as e:  # noqa: BLE001
                results.append(("FAIL", fn.check_name, str(e).strip().splitlines()[0][:300] + (f" | page errors: {errors}" if errors else "")))
            context.close()
        browser.close()
    clear_tasks()
    for status, name, detail in results:
        print(f"{status:4} {name}" + (f"\n       {detail}" if detail else ""))
    sys.exit(any(status == "FAIL" for status, _, _ in results))


if __name__ == "__main__":
    main()
