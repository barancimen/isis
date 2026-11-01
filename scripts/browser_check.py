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


STEPS = [0]


def opens_as_shown(page, card) -> str:
    """Click a result card; the reader must highlight exactly the passage text the card shows
    (its dimmed context excluded), and every word the card marks. Returns the passage's start."""
    squash = lambda t: re.sub(r"\s+", "", t)
    shown = card.evaluate("c => [...c.querySelectorAll('.hit-body .line:not(.ctx)')].map(l => l.textContent).join(' ')")
    marked = card.locator(".hit-body .line:not(.ctx) mark").count()
    path = card.locator(".hit-loc").inner_text().rsplit(":", 1)[0]
    card.locator(".hit-body").click()
    expect(page.locator("#reader-path")).to_have_text(path)  # the highlight is set as the file appears
    expect(page.locator("#reader-editor .cm-hit-range").first).to_be_visible()
    highlighted = page.evaluate("() => [...document.querySelectorAll('#reader-editor .cm-hit-range')].map(e => e.textContent).join(' ')")
    assert squash(highlighted) == squash(shown), f"reader: ...{highlighted[-80:]!r}\ncard:   ...{shown[-80:]!r}"
    assert page.locator("#reader-editor .cm-hit-match").count() == marked, (page.locator("#reader-editor .cm-hit-match").count(), marked)
    return shown[:40]


def refresh_related(page):
    """Press Refresh, and wait for the new suggestions."""
    expect(page.locator("#related-refresh")).to_be_enabled(timeout=30000)
    with page.expect_response(lambda r: r.url.endswith("/related"), timeout=30000):
        page.click("#related-refresh")
    expect(page.locator("#related-status")).not_to_have_text("Updating…", timeout=30000)
    page.wait_for_timeout(300)  # the response is rendered


def step(message):
    STEPS[0] += 1
    print(f"{STEPS[0]:>2}. {message}", flush=True)


def set_slider(page, selector, value):
    page.eval_on_selector(selector, "(el, v) => { el.value = v; el.dispatchEvent(new Event('input', {bubbles: true})); }", str(value))


def shot(page, name):
    if SHOTS:
        SHOTS.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SHOTS / name))


def main() -> None:
    errors: list[str] = []
    answers: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(ignore_default_args=["--hide-scrollbars"])
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.on("console", lambda m: m.type == "error" and not re.search(r"\b(409|400)\b", m.text) and errors.append(m.text))
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
        step("created a task, typed in it, autosaved to disk")

        page.click(".side-tabs button[data-panel=search]")
        page.fill("#search-input", '"white whale"')
        expect(page.locator("#search-results .hit").first).to_be_visible(timeout=10000)
        location = page.locator("#search-results .hit .hit-loc").first.inner_text()
        opens_as_shown(page, page.locator("#search-results .hit").first)
        expect(page.locator("#search-results .hit").first).to_have_class(re.compile("opened"))  # stays marked
        expect(page.locator("#reader-path")).to_have_text(location.rsplit(":", 1)[0])
        page.fill("#search-input", "")  # or the old results could be swapped out under the clicks below
        expect(page.locator("#search-results .hit")).to_have_count(0)
        page.fill("#search-input", "whale ship")  # long passages that start and end inside paragraphs
        expect(page.locator("#search-results .hit").nth(5)).to_be_visible(timeout=10000)
        for i in range(6):
            opens_as_shown(page, page.locator("#search-results .hit").nth(i))
        step(f"search hits open at their match, highlighted from the passage's first to its last word: {location}")
        shot(page, "search.png")

        searches: list[str] = []
        page.on("request", lambda r: "/api/search" in r.url and searches.append(urllib.parse.unquote(r.url)))
        page.locator("#search-tune > summary").click()
        set_slider(page, "#search-tune [data-name=keyword_weight] input", 0)
        expect(page.locator("#search-tune .tune-count")).to_have_text(" · 1 changed")
        expect(page.locator("#search-results .hit .hit-why").first).to_have_text(re.compile(r"meaning #1$"), timeout=15000)
        assert any('"keyword_weight":0' in u for u in searches), searches
        assert page.locator("#search-tune .tune-python").inner_text() == "SearchConfig(keyword_weight=0)"
        page.locator("#search-tune button", has_text="Reset to defaults").click()
        expect(page.locator("#search-tune .tune-count")).to_have_text("")
        step("search parameters: keyword weight 0 ranks by meaning alone; reset restores the defaults")

        slider = lambda name: page.locator(f"#search-tune [data-name={name}] input")
        expect(page.locator("#search-meaning")).to_have_attribute("aria-pressed", "true")
        page.click("#search-meaning")
        expect(page.locator("#search-meaning")).to_have_attribute("aria-pressed", "false")
        expect(slider("meaning_weight")).to_have_value("0")
        expect(page.locator("#search-results .hit .hit-why")).to_have_count(0, timeout=15000)  # no meaning ranks
        set_slider(page, "#search-tune [data-name=meaning_weight] input", 2)  # the slider moves the button too
        expect(page.locator("#search-meaning")).to_have_attribute("aria-pressed", "true")
        page.click("#search-meaning")
        expect(slider("meaning_weight")).to_have_value("0")
        page.click("#search-meaning")  # back to the value it had
        expect(slider("meaning_weight")).to_have_value("2")
        set_slider(page, "#search-tune [data-name=k1] input", 2)
        expect(page.locator("#search-tune .tune-count")).to_have_text(" · 2 changed")
        page.locator("#search-tune [data-name=meaning_weight] .tune-default").click()
        expect(slider("meaning_weight")).to_have_value("1")
        expect(page.locator("#search-tune .tune-count")).to_have_text(" · 1 changed")
        expect(page.locator("#search-tune [data-name=meaning_weight] .tune-default")).to_be_hidden()
        page.locator("#search-tune button", has_text="Reset to defaults").click()
        expect(slider("k1")).to_have_value("1.2")
        expect(page.locator("#search-tune .tune-count")).to_have_text("")
        step("Meaning button and Meaning weight slider move together; default buttons restore one slider or all")

        passage_lines = """() => [...document.querySelectorAll('#search-results .hit-body .line:not(.ctx)')].map(l => l.innerText)"""
        page.fill("#search-input", "")
        expect(page.locator("#search-results .hit")).to_have_count(0)
        page.fill("#search-input", "whale -/Ahab/ -harpoon")
        expect(page.locator("#search-results .hit").first).to_be_visible(timeout=15000)
        lines = page.evaluate(passage_lines)
        assert lines and not any("Ahab" in l or "harpoon" in l.lower() for l in lines), lines
        page.fill("#search-input", "")
        expect(page.locator("#search-results .hit")).to_have_count(0)
        page.fill("#search-input", "/white whale/")
        expect(page.locator("#search-results .hit mark").first).to_have_text(re.compile("white whale", re.I), timeout=15000)
        page.fill("#search-input", "/(oops/")
        expect(page.locator("#search-status")).to_contain_text("invalid pattern", timeout=15000)
        step("search: exclusions hold with Meaning on, /regex/ matches are highlighted, a bad pattern is reported")

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
        step("reader selection inserted at the task cursor (Ctrl+Enter)")

        page.click(".side-tabs button[data-panel=related]")
        expect(page.locator("#related-lanes .lane")).to_have_count(4, timeout=30000)
        expect(page.locator("#related-lanes .hit").first).to_be_visible(timeout=30000)
        counts = page.evaluate("""() => [...document.querySelectorAll('#related-lanes .lane h3')].map(h => h.innerText.replace(/\\n/g, ' '))""")
        for card in page.locator("#related-lanes .hit").all()[:6]:
            opens_as_shown(page, card)
        last = page.locator("#related-lanes .hit").nth(5)
        expect(last).to_have_class(re.compile("opened"))
        expect(page.locator("#related-lanes .hit.opened")).to_have_count(1)  # only the last one clicked
        step(f"related lanes: {counts}; suggestions open highlighted exactly as their cards show them")
        shot(page, "related.png")

        page.locator("#related-tune > summary").click()
        set_slider(page, "#related-tune [data-name=total_slots] input", 6)
        expect(page.locator("#related-tune .tune-count")).to_have_text(" · 1 changed")
        refresh_related(page)
        page.wait_for_function("() => document.querySelectorAll('#related-lanes .hit').length <= 6", timeout=30000)
        shot(page, "related-tuned.png")
        page.locator("#related-tune [data-name=total_slots] .tune-default").click()  # one slider back
        expect(page.locator("#related-tune [data-name=total_slots] input")).to_have_value("24")
        expect(page.locator("#related-tune .tune-count")).to_have_text("")
        refresh_related(page)
        page.wait_for_function("() => document.querySelectorAll('#related-lanes .hit').length > 6", timeout=30000)
        set_slider(page, "#related-tune [data-name=query] select", "paragraphs")  # the dropdown too
        expect(page.locator("#related-tune .tune-count")).to_have_text(" · 1 changed")
        page.locator("#related-tune [data-name=query] .tune-default").click()
        expect(page.locator("#related-tune [data-name=query] select")).to_have_value("document")
        expect(page.locator("#related-tune [data-name=query] .tune-default")).to_be_hidden()
        set_slider(page, "#related-tune [data-name=total_slots] input", 6)  # and all of them
        page.locator("#related-tune button", has_text="Reset to defaults").click()
        expect(page.locator("#related-tune .tune-count")).to_have_text("")
        refresh_related(page)
        page.wait_for_function("() => document.querySelectorAll('#related-lanes .hit').length > 6", timeout=30000)
        page.locator("#related-tune > summary").click()
        step("related parameters: 'Results' at 6 shows at most 6; default buttons restore one setting or all")

        refreshes: list[str] = []
        page.on("request", lambda r: "/related" in r.url and refreshes.append(r.url))
        page.evaluate("""() => {
            const v = window.isis.taskView;
            v.dispatch({selection: {anchor: 0, head: v.state.doc.length}});
            v.dispatch({selection: {anchor: 2}});
        }""")
        page.wait_for_timeout(2500)
        assert not refreshes, "selecting text or moving the cursor refreshed the related panel"
        expect(page.locator("#related-refresh")).to_be_disabled()
        step("selecting text and moving the cursor leave the related panel alone")

        shown = lambda: page.locator("#related-lanes .hit .hit-loc").all_inner_texts()
        cards = shown()
        page.click("#task-editor .cm-content")
        page.keyboard.press("Control+End")
        page.keyboard.type(" An edit.")
        expect(page.locator("#related-refresh")).to_be_enabled()
        page.wait_for_timeout(3000)  # longer than any pause in typing
        assert not refreshes and shown() == cards, "an edit changed the suggestions by itself"
        refresh_related(page)
        assert len(refreshes) == 1
        expect(page.locator("#related-refresh")).to_be_disabled()
        step("an edit only arms Refresh: the suggestions stay until it is pressed, and then it is off again")

        before = len(task_text())
        page.locator("#related-lanes .hit .add").first.click()
        assert len(task_text()) > before
        saved()
        meta = json.loads((TASKS / ".meta" / f"{WHALES}.json").read_text())
        step(f"+ inserted a related passage and recorded feedback {meta['feedback']}")

        page.locator("#related-terms .chip .chip-x").first.click()
        expect(page.locator("#related-terms .chip.ignored")).to_have_count(1, timeout=30000)
        with page.expect_response(lambda r: r.url.endswith("/feedback"), timeout=30000):
            page.locator("#related-lanes .hit button[title^='Not useful']").first.click()
        page.wait_for_timeout(200)
        meta = json.loads((TASKS / ".meta" / f"{WHALES}.json").read_text())
        assert meta["ignored"] and meta["dismissed"]
        step(f"ignored a keyword {meta['ignored']} and dismissed a passage")

        drawer = page.locator("#related-feedback")
        drawer.locator("summary").click()
        expect(drawer.locator(".tune-count")).to_contain_text("1 dismissed", timeout=30000)
        expect(drawer.locator(".feedback-item")).to_have_count(1)
        shot(page, "feedback.png")
        drawer.locator(".feedback-item button", has_text="Restore").click()
        expect(drawer.locator(".tune-count")).not_to_contain_text("dismissed", timeout=30000)
        assert json.loads((TASKS / ".meta" / f"{WHALES}.json").read_text())["dismissed"] == []
        drawer.locator("button", has_text="Forget all").click()
        expect(drawer.locator(".tune-count")).to_have_text("", timeout=30000)
        expect(page.locator("#related-terms .chip.ignored")).to_have_count(0)
        assert not (TASKS / ".meta" / f"{WHALES}.json").exists()
        drawer.locator("summary").click()
        step("feedback drawer: restored the dismissed passage, then forgot all of the task's feedback")

        related_lines = """() => [...document.querySelectorAll('#related-lanes .hit-body .line:not(.ctx)')].map(l => l.innerText)"""
        refresh_related(page)
        assert any("squid" in l.lower() for l in page.evaluate(related_lines))
        page.fill("#steer-input", "  -squid ")
        page.press("#steer-input", "Enter")
        expect(page.locator("#steer-input")).to_have_value("-squid", timeout=30000)
        assert any("squid" in l.lower() for l in page.evaluate(related_lines))  # not until Refresh
        refresh_related(page)
        lines = page.evaluate(related_lines)
        assert lines and not any("squid" in l.lower() for l in lines), lines
        page.fill("#steer-input", "-squid +/[Ww]hal/")
        page.press("#steer-input", "Enter")
        refresh_related(page)
        hits = page.evaluate("""() => [...document.querySelectorAll('#related-lanes .hit')].map(h =>
            [...h.querySelectorAll('.hit-body .line:not(.ctx)')].map(l => l.innerText).join(' '))""")
        assert hits and all(re.search("[Ww]hal", t) for t in hits), hits
        page.fill("#steer-input", "/(oops/")
        page.press("#steer-input", "Enter")
        expect(page.locator("#steer-error")).to_contain_text("invalid pattern", timeout=30000)
        page.press("#steer-input", "Escape")
        expect(page.locator("#steer-input")).to_have_value("-squid +/[Ww]hal/")
        page.locator("#related-lanes .hit button.avoid").first.click()
        expect(page.locator("#related-feedback .tune-count")).to_contain_text("1 avoided", timeout=30000)
        meta = json.loads((TASKS / ".meta" / f"{WHALES}.json").read_text())
        assert meta["steer"] == "-squid +/[Ww]hal/" and meta["dismissed"][0]["avoid"] is True
        shot(page, "steer.png")
        step("related steer line: -squid excludes, +/regex/ requires, a bad pattern is reported; ⊘ avoided a passage")

        card = page.locator("#related-lanes .hit").first
        title = card.locator(".hit-loc").inner_text()
        card.locator("button.like").click()
        marked = page.locator("#related-lanes .hit", has=page.locator("button.like[aria-pressed=true]"))
        expect(marked).to_have_count(1, timeout=30000)
        assert marked.locator(".hit-loc").inner_text() == title  # it stays on show
        expect(page.locator("#related-feedback .tune-count")).to_contain_text("1 more like this")
        meta = json.loads((TASKS / ".meta" / f"{WHALES}.json").read_text())
        assert len(meta["liked"]) == 1 and meta["liked"][0]["path"] == title.rsplit(":", 1)[0]
        assert task_text().count(meta["liked"][0]["text"][:40]) == 0  # not added to the document
        if not page.locator("#related-feedback").evaluate("e => e.open"):
            page.locator("#related-feedback > summary").click()
        page.locator("#related-feedback .feedback-item button", has_text="Stop").click()
        expect(marked).to_have_count(0, timeout=30000)
        step("↑ more like this: marks a suggestion, stays on show, stays out of the document; the drawer undoes it")

        (TASKS / f"{WHALES}.md").write_text("edited elsewhere\n")
        page.click("#task-editor .cm-content")
        page.keyboard.press("Control+End")
        page.keyboard.type(" More.")
        expect(page.locator("#conflict")).to_be_visible(timeout=10000)
        page.click("#conflict-keep")
        saved()
        assert (TASKS / f"{WHALES}.md").read_text().endswith("More.")
        step("a change on disk was detected; 'Keep mine' overwrote it")

        cursor = page.evaluate("window.isis.taskView.state.selection.main.head")
        answers.append(ISLANDS)
        page.click("#new-task")
        expect(page.locator(".task-tab.active")).to_contain_text(ISLANDS)
        page.locator(".task-tab .name", has_text=WHALES).click()
        expect(page.locator(".task-tab.active")).to_contain_text(WHALES)
        assert task_text().endswith("More.")
        assert page.evaluate("window.isis.taskView.state.selection.main.head") == cursor
        step("switched tasks and back; text and cursor kept")

        side_width = lambda: page.evaluate("document.querySelector('#side').getBoundingClientRect().width")
        before = side_width()
        box = page.locator(".resizer[data-resize=side]").bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + 200
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + 60, y, steps=4)
        page.mouse.move(x + 120, y, steps=4)
        page.mouse.up()
        widened = side_width()
        assert abs(widened - before - 120) < 6, (before, widened)
        page.reload()
        expect(page.locator("#index-status")).to_contain_text("passages", timeout=20000)
        assert abs(side_width() - widened) < 2, "the width was not kept"
        page.locator(".resizer[data-resize=side]").dblclick()
        assert abs(side_width() - before) < 2
        step(f"dragged the side panel from {before:.0f} to {widened:.0f}px; kept after reload; double-click reset it")

        expect(page.locator("#index-size")).to_have_text(re.compile(r"^[\d,]+ documents · index [\d.]+ [KMG]B$"))
        step(f"status bar: {page.locator('#index-size').inner_text()}")

        page.reload()
        page.click(".side-tabs button[data-panel=files]")
        expect(page.locator("#file-list .dir").first).to_be_visible(timeout=10000)
        expect(page.locator("#file-list .file")).to_have_count(0)  # folders start collapsed
        folder = page.locator("#file-list .dir").first
        folder.click()
        expect(page.locator("#file-list .file").first).to_be_visible()
        expect(page.locator("#file-list .dir").first).to_have_attribute("aria-expanded", "true")
        page.locator("#file-list .dir").first.click()
        expect(page.locator("#file-list .file")).to_have_count(0)
        target = page.evaluate("window.isis.state.files[window.isis.state.files.length - 1].path")
        page.evaluate("(path) => window.isis.openFile(path)", target)
        expect(page.locator("#file-list .file.current")).to_have_attribute("data-path", target)  # revealed
        step("files panel: folders start collapsed and toggle; the file in the reader is revealed")

        first = page.evaluate("window.isis.state.files[0].path")
        page.fill("#file-filter", first)
        expect(page.locator("#file-list .file").first).to_have_attribute("data-path", first)
        page.locator("#file-list .file").first.click()
        expect(page.locator("#reader-path")).to_have_text(first)
        step(f"files panel filtered and opened {first}")
        shot(page, "files.png")

        # The scrollbar follows the mouse, even over text not yet on screen (long chapters).
        page.evaluate("window.isis.openFile('voyage-of-the-beagle/017-chapter-xvii-galapagos-archipelago.md')")
        expect(page.locator("#reader-path")).to_contain_text("017-chapter")
        measure = """() => { const e = document.querySelector('#reader-editor .cm-scroller'); const r = e.getBoundingClientRect();
            return [e.scrollTop, e.scrollHeight, e.clientHeight, r.right, r.top]; }"""
        def thumb(top, height, client):  # as styled: no arrow buttons, at least 40 px tall
            size = max(client * client / height, 40)
            return top / max(height - client, 1) * (client - size), size
        top, height, client, right, y0 = page.evaluate(measure)
        start, size = thumb(top, height, client)
        page.mouse.move(right - 6, y0 + start + size / 2)
        page.mouse.down()
        for i in range(1, 11):
            page.mouse.move(right - 6, y0 + start + size / 2 + 30 * i, steps=2)
            page.wait_for_timeout(80)
        page.wait_for_timeout(200)
        now = page.evaluate(measure)
        moved = thumb(now[0], now[1], now[2])[0] - start
        page.mouse.up()
        assert 285 <= moved <= 315, f"the mouse moved 300 px, the scrollbar {moved:.0f} px"
        step(f"reader scrollbar follows a first drag: mouse 300 px, thumb {moved:.0f} px")

        sample = CORPUS / "e2e-envelopes.md"
        sample.write_text("---\ntitle: A survey\ntags: [whales, oil]\n---\n\nA hard wrapped\nparagraph.\n\n---\nnote: in the middle\nstatus: draft\n---\n\n"
                          "---\nA **prose** section that\nis hard wrapped.\n---\n\nA real heading\n---\n\nBefore a rule.\n\n---\n\nAfter it.\n")
        try:
            page.evaluate("(p) => window.isis.openFile(p)", sample.name)
            editor = page.locator("#reader-editor")
            line = lambda text: editor.locator(".cm-line", has_text=text)
            weight = lambda locator: int(locator.evaluate("e => getComputedStyle(e).fontWeight"))
            expect(line("title: A survey")).to_have_class(re.compile("cm-env-meta"))
            expect(line("note: in the middle")).to_have_class(re.compile("cm-env-meta"))
            assert editor.locator(".cm-env-key").all_inner_texts() == ["title:", "tags:", "note:", "status:"]
            assert line("status: draft").inner_text().strip() == "status: draft"  # one line per key
            assert weight(line("[whales, oil]").locator("span", has_text="whales").last) < 600  # data: no heading, no link
            prose = line("section that is hard wrapped.")  # flows, in a card, not a heading
            expect(prose).to_have_class(re.compile("cm-env"))
            assert "cm-env-meta" not in prose.get_attribute("class")
            assert weight(prose.locator("span", has_text="prose").first) >= 600  # **prose** is bold
            assert prose.evaluate("e => getComputedStyle(e).fontSize") == line("Before a rule.").evaluate("e => getComputedStyle(e).fontSize")
            assert weight(line("A real heading").locator("span").first) >= 600  # still a heading
            assert editor.locator(".cm-rule").count() == 1
            step("--- sections: metadata and prose cards, a real heading and a rule each render as themselves")
        finally:
            sample.unlink()
        browser.close()

    for name in (WHALES, ISLANDS):
        request = urllib.request.Request(f"{URL}/api/tasks/{urllib.parse.quote(name)}", method="DELETE")
        urllib.request.urlopen(request).read()
    print("browser errors:", errors or "none")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
