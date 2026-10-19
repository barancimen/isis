"""Drive the running web app in headless Chromium and check the main workflows end to end.

    pip install playwright && playwright install chromium
    isis serve sample_corpus --port 8765            # in another terminal
    python scripts/browser_check.py http://127.0.0.1:8765 sample_corpus [screenshot_dir]

It creates two tasks named "e2e-…" in the corpus's _tasks folder and deletes them at the end
(deleted tasks go to _tasks/.trash). The search steps expect the sample corpus.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

URL, CORPUS = sys.argv[1].rstrip("/"), Path(sys.argv[2])
SHOTS = Path(sys.argv[3]) if len(sys.argv) > 3 else None
TASKS = CORPUS / "_tasks"
STAMP = time.strftime("%H%M%S")
WHALES, ISLANDS = f"e2e-whales-{STAMP}", f"e2e-islands-{STAMP}"


def step(n, message):
    print(f"{n:>2}. {message}", flush=True)


def shot(page, name):
    if SHOTS:
        SHOTS.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SHOTS / name))


def main() -> None:
    errors: list[str] = []
    answers: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.on("console", lambda m: m.type == "error" and "409" not in m.text and errors.append(m.text))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("dialog", lambda d: d.accept(answers.pop(0)) if d.type == "prompt" else d.accept())
        task_text = lambda: page.evaluate("window.isis.taskView.state.doc.toString()")
        saved = lambda: expect(page.locator("#save-state")).to_have_text("Saved", timeout=15000)

        page.goto(URL)
        expect(page.locator("#index-status")).to_contain_text("passages", timeout=20000)

        answers.append(WHALES)
        page.click("#new-task")
        expect(page.locator(".task-tab.active")).to_contain_text(WHALES)
        page.click("#task-editor .cm-content")
        page.keyboard.type("# Whales\nThe sperm whale dives deep to hunt giant squid.\n\n# Selection\nNatural selection preserves useful variations.")
        saved()
        assert (TASKS / f"{WHALES}.md").read_text().startswith("# Whales")
        step(1, "created a task, typed in it, autosaved to disk")

        page.click(".side-tabs button[data-panel=search]")
        page.fill("#search-input", '"white whale"')
        expect(page.locator("#search-results .hit").first).to_be_visible(timeout=10000)
        location = page.locator("#search-results .hit .hit-loc").first.inner_text()
        page.locator("#search-results .hit .hit-body").first.click()
        expect(page.locator("#reader-path")).to_have_text(location.rsplit(":", 1)[0])
        expect(page.locator("#reader-editor .cm-hit-match").first).to_be_visible()
        step(2, f"search hit opened at its match: {location}")
        shot(page, "search.png")

        page.evaluate("""() => {
            const v = window.isis.taskView, doc = v.state.doc.toString();
            v.dispatch({selection: {anchor: doc.indexOf("squid.") + "squid.".length}});
        }""")
        selected = page.evaluate("""() => {
            const r = window.isis.reader, line = r.state.doc.lineAt(r.state.selection.main.head);
            r.dispatch({selection: {anchor: line.from, head: line.to}}); r.focus();
            return r.state.sliceDoc(line.from, line.to).trim();
        }""")
        page.keyboard.press("Control+Enter")
        doc = task_text()
        assert doc.index("squid.") < doc.index(selected) < doc.index("# Selection"), doc
        saved()
        step(3, "reader selection inserted at the task cursor (Ctrl+Enter)")

        page.click(".side-tabs button[data-panel=related]")
        expect(page.locator("#related-lanes .lane")).to_have_count(4, timeout=30000)
        expect(page.locator("#related-lanes .hit").first).to_be_visible(timeout=30000)
        counts = page.evaluate("""() => [...document.querySelectorAll('#related-lanes .lane h3')].map(h => h.innerText.replace(/\\n/g, ' '))""")
        step(4, f"related lanes: {counts}")
        shot(page, "related.png")

        before = len(task_text())
        page.locator("#related-lanes .hit .add").first.click()
        assert len(task_text()) > before
        saved()
        meta = json.loads((TASKS / ".meta" / f"{WHALES}.json").read_text())
        step(5, f"+ inserted a related passage and recorded feedback {meta['feedback']}")

        page.locator("#related-terms .chip .chip-x").first.click()
        expect(page.locator("#related-terms .chip.banned")).to_have_count(1, timeout=30000)
        page.locator("#related-lanes .hit button[title^='Not useful']").first.click()
        expect(page.locator("#related-status")).not_to_have_text("Updating…", timeout=30000)
        meta = json.loads((TASKS / ".meta" / f"{WHALES}.json").read_text())
        assert meta["bans"] and meta["dismissed"]
        step(6, f"banned a term {meta['bans']} and dismissed a passage")

        (TASKS / f"{WHALES}.md").write_text("edited elsewhere\n")
        page.click("#task-editor .cm-content")
        page.keyboard.press("Control+End")
        page.keyboard.type(" More.")
        expect(page.locator("#conflict")).to_be_visible(timeout=10000)
        page.click("#conflict-keep")
        saved()
        assert (TASKS / f"{WHALES}.md").read_text().endswith("More.")
        step(7, "a change on disk was detected; 'Keep mine' overwrote it")

        cursor = page.evaluate("window.isis.taskView.state.selection.main.head")
        answers.append(ISLANDS)
        page.click("#new-task")
        expect(page.locator(".task-tab.active")).to_contain_text(ISLANDS)
        page.locator(".task-tab .name", has_text=WHALES).click()
        expect(page.locator(".task-tab.active")).to_contain_text(WHALES)
        assert task_text().endswith("More.")
        assert page.evaluate("window.isis.taskView.state.selection.main.head") == cursor
        step(8, "switched tasks and back; text and cursor kept")

        page.click(".side-tabs button[data-panel=files]")
        first = page.evaluate("window.isis.state.files[0].path")
        page.fill("#file-filter", first)
        expect(page.locator("#file-list .file").first).to_have_attribute("data-path", first)
        page.locator("#file-list .file").first.click()
        expect(page.locator("#reader-path")).to_have_text(first)
        step(9, f"files panel filtered and opened {first}")
        shot(page, "files.png")
        browser.close()

    for name in (WHALES, ISLANDS):
        request = urllib.request.Request(f"{URL}/api/tasks/{urllib.parse.quote(name)}", method="DELETE")
        urllib.request.urlopen(request).read()
    print("browser errors:", errors or "none")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
