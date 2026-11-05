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
    expect(page.locator("#related-status")).not_to_have_attribute("aria-busy", "true", timeout=30000)
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
        saved = lambda: expect(page.locator("#save-state")).to_have_attribute("data-state", "saved", timeout=15000)

        page.goto(URL)
        expect(page.locator("#statusbar")).to_have_attribute("data-state", "ready", timeout=20000)

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
        expect(page.locator("#search-tune .tune-count")).to_have_attribute("data-changed", "1")
        expect(page.locator("#search-results .hit").first).to_have_attribute("data-meaning-rank", "1", timeout=15000)
        assert any('"keyword_weight":0' in u for u in searches), searches
        page.click("#search-tune .tune-reset")
        expect(page.locator("#search-tune .tune-count")).to_have_attribute("data-changed", "0")
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
        expect(page.locator("#search-tune .tune-count")).to_have_attribute("data-changed", "2")
        page.locator("#search-tune [data-name=meaning_weight] .tune-default").click()
        expect(slider("meaning_weight")).to_have_value("1")
        expect(page.locator("#search-tune .tune-count")).to_have_attribute("data-changed", "1")
        expect(page.locator("#search-tune [data-name=meaning_weight] .tune-default")).to_be_hidden()
        page.click("#search-tune .tune-reset")
        expect(slider("k1")).to_have_value("1.2")
        expect(page.locator("#search-tune .tune-count")).to_have_attribute("data-changed", "0")
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
        expect(page.locator("#search-status")).to_contain_text("⚠", timeout=15000)
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
        expect(page.locator("#related-tune .tune-count")).to_have_attribute("data-changed", "1")
        refresh_related(page)
        page.wait_for_function("() => document.querySelectorAll('#related-lanes .hit').length <= 6", timeout=30000)
        shot(page, "related-tuned.png")
        page.locator("#related-tune [data-name=total_slots] .tune-default").click()  # one slider back
        expect(page.locator("#related-tune [data-name=total_slots] input")).to_have_value("24")
        expect(page.locator("#related-tune .tune-count")).to_have_attribute("data-changed", "0")
        refresh_related(page)
        page.wait_for_function("() => document.querySelectorAll('#related-lanes .hit').length > 6", timeout=30000)
        set_slider(page, "#related-tune [data-name=query] select", "paragraphs")  # the dropdown too
        expect(page.locator("#related-tune .tune-count")).to_have_attribute("data-changed", "1")
        page.locator("#related-tune [data-name=query] .tune-default").click()
        expect(page.locator("#related-tune [data-name=query] select")).to_have_value("document")
        expect(page.locator("#related-tune [data-name=query] .tune-default")).to_be_hidden()
        set_slider(page, "#related-tune [data-name=total_slots] input", 6)  # and all of them
        page.click("#related-tune .tune-reset")
        expect(page.locator("#related-tune .tune-count")).to_have_attribute("data-changed", "0")
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
            page.locator("#related-lanes .hit button.dismiss").first.click()
        page.wait_for_timeout(200)
        meta = json.loads((TASKS / ".meta" / f"{WHALES}.json").read_text())
        assert meta["ignored"] and meta["dismissed"]
        step(f"ignored a keyword {meta['ignored']} and dismissed a passage")

        drawer = page.locator("#related-feedback")
        drawer.locator("summary").click()
        expect(drawer.locator(".feedback-item[data-kind=removed]")).to_have_count(1, timeout=30000)
        expect(drawer.locator(".feedback-item")).to_have_count(1)
        shot(page, "feedback.png")
        drawer.locator(".chip.ignored .chip-x").click()
        expect(page.locator("#related-terms .chip.ignored")).to_have_count(0, timeout=30000)
        expect(drawer.locator(".chip.ignored")).to_have_count(0, timeout=30000)
        assert json.loads((TASKS / ".meta" / f"{WHALES}.json").read_text())["ignored"] == []
        step("the drawer's × searches for an ignored keyword again")
        page.locator("#related-terms .chip .chip-x").first.click()
        expect(page.locator("#related-terms .chip.ignored")).to_have_count(1, timeout=30000)
        drawer.locator(".feedback-item[data-kind=removed] .undo").click()
        expect(drawer.locator(".feedback-item[data-kind=removed]")).to_have_count(0, timeout=30000)
        assert json.loads((TASKS / ".meta" / f"{WHALES}.json").read_text())["dismissed"] == []
        drawer.locator(".tune-reset").click()
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
        expect(page.locator("#steer-error")).to_contain_text("⚠", timeout=30000)
        page.press("#steer-input", "Escape")
        expect(page.locator("#steer-input")).to_have_value("-squid +/[Ww]hal/")
        page.locator("#related-lanes .hit button.avoid").first.click()
        expect(page.locator("#related-feedback .feedback-item[data-kind=removed] .feedback-tag")).to_have_count(1, timeout=30000)
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
        expect(page.locator("#related-feedback .feedback-item[data-kind=liked]")).to_have_count(1)
        meta = json.loads((TASKS / ".meta" / f"{WHALES}.json").read_text())
        assert len(meta["liked"]) == 1 and meta["liked"][0]["path"] == title.rsplit(":", 1)[0]
        assert task_text().count(meta["liked"][0]["text"][:40]) == 0  # not added to the document
        if not page.locator("#related-feedback").evaluate("e => e.open"):
            page.locator("#related-feedback > summary").click()
        page.locator("#related-feedback .feedback-item[data-kind=liked] .undo").click()
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
        expect(page.locator("#statusbar")).to_have_attribute("data-state", "ready", timeout=20000)
        assert abs(side_width() - widened) < 2, "the width was not kept"
        page.locator(".resizer[data-resize=side]").dblclick()
        assert abs(side_width() - before) < 2
        step(f"dragged the side panel from {before:.0f} to {widened:.0f}px; kept after reload; double-click reset it")

        expect(page.locator("#index-size")).to_have_text(re.compile(r"\d"))
        step(f"status bar: {page.locator('#index-size').inner_text()}")

        # Another browser: here a reload keeps the open folders, and the task's document opens its own.
        fresh = browser.new_page(viewport={"width": 1600, "height": 1000})
        fresh.goto(URL)
        fresh.click(".side-tabs button[data-panel=files]")
        expect(fresh.locator("#file-list .dir").first).to_be_visible(timeout=10000)
        expect(fresh.locator("#file-list .file")).to_have_count(0)  # folders start collapsed
        folder = fresh.locator("#file-list .dir").first
        folder.click()
        expect(fresh.locator("#file-list .file").first).to_be_visible()
        expect(fresh.locator("#file-list .dir").first).to_have_attribute("aria-expanded", "true")
        fresh.locator("#file-list .dir").first.click()
        expect(fresh.locator("#file-list .file")).to_have_count(0)
        target = fresh.evaluate("window.isis.state.files[window.isis.state.files.length - 1].path")
        fresh.evaluate("(path) => window.isis.openFile(path)", target)
        expect(fresh.locator("#file-list .file.current")).to_have_attribute("data-path", target)  # revealed
        fresh.close()
        step("files panel: folders start collapsed and toggle; the file in the reader is revealed")

        page.click(".side-tabs button[data-panel=files]")
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
            page.wait_for_selector("#reader .cm-env-meta:not(.cm-env-edge)")
            highlight = page.evaluate("getComputedStyle(document.querySelector('#reader .cm-env-meta:not(.cm-env-edge)'), '::selection').backgroundColor")
            assert highlight != "rgba(0, 0, 0, 0)", highlight  # a selection in the card must show
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

        sample = CORPUS / "e2e-lists.md"
        sample.write_text("- top\n  - sub\n    - subsub\n- hard wrapped\n  and more\n\n1. one\n   - under one\n\n    code, not a list\n")
        try:
            page.evaluate("(p) => window.isis.openFile(p)", sample.name)
            items = page.locator("#reader-editor .cm-li")
            expect(items.first).to_be_visible()
            pad = lambda text: page.locator("#reader-editor .cm-line", has_text=text).first.evaluate("e => parseFloat(getComputedStyle(e).paddingLeft)")
            top, sub, subsub = pad("top"), pad("sub"), pad("subsub")
            assert sub - top >= 18 and subsub - sub >= 18, (top, sub, subsub)  # each level clearly further in
            assert page.locator("#reader-editor .cm-li-marker").all_inner_texts() == ["•", "◦", "▪", "•", "1.", "◦"]
            wrapped = page.locator("#reader-editor .cm-line", has_text="hard wrapped").first.inner_text()
            assert wrapped.replace("•", "").split() == ["hard", "wrapped", "and", "more"], wrapped  # it flows on
            assert (CORPUS / sample.name).read_text().startswith("- top\n  - sub")  # the file is unchanged
            step("lists: bullets per level, each clearly indented; hard-wrapped items flow; numbers stay")

            page.evaluate("""() => { const v = window.isis.taskView;
                v.dispatch({changes: {from: 0, to: v.state.doc.length, insert: '- one\\n\\n  - nested item\\n\\n- third'}}); v.focus(); }""")
            spot = """() => { const r = document.createRange(); const line = [...document.querySelectorAll('#task-editor .cm-line')].find((l) => l.textContent.includes('nested item'));
                const mark = line.querySelector('.cm-li-marker').getBoundingClientRect();
                const walker = document.createTreeWalker(line, NodeFilter.SHOW_TEXT); let node;
                while ((node = walker.nextNode())) if (node.textContent.includes('nested')) break;
                r.setStart(node, node.textContent.indexOf('nested')); r.setEnd(node, node.textContent.indexOf('nested') + 1);
                return { marker: line.querySelector('.cm-li-marker').textContent, centre: mark.left + mark.width / 2, text: r.getBoundingClientRect().left }; }"""
            page.evaluate("() => { const v = window.isis.taskView; v.dispatch({selection: {anchor: v.state.doc.line(5).to}}); }")
            page.wait_for_timeout(150)
            off = page.evaluate(spot)
            page.evaluate("() => { const v = window.isis.taskView; v.dispatch({selection: {anchor: v.state.doc.line(3).to}}); }")
            page.wait_for_timeout(150)
            on = page.evaluate(spot)
            assert (off["marker"], on["marker"]) == ("◦", "-"), (off, on)
            assert abs(off["centre"] - on["centre"]) < 0.5 and abs(off["text"] - on["text"]) < 0.5, (off, on)  # nothing moves
            page.keyboard.press("Enter")
            page.keyboard.type("next")
            assert task_text().split("\n")[2:4] == ["  - nested item", "  - next"], task_text()  # no blank line between
            step("in a task the marker on the cursor's line takes the bullet's exact place; Enter goes straight to the next item")
        finally:
            sample.unlink()

        def open_from_files(path):
            page.click(".side-tabs button[data-panel=files]")
            page.fill("#file-filter", path)
            page.click(f'#file-list .file[data-path="{path}"]')
            expect(page.locator("#reader-path")).to_have_text(path)
            page.wait_for_timeout(200)

        def top_of_reader():
            return page.evaluate("""() => {
                const view = window.isis.reader, box = view.scrollDOM.getBoundingClientRect();
                return view.posAtCoords({ x: view.contentDOM.getBoundingClientRect().left + 1, y: box.top + 1 }, false);
            }""")

        def scroll_reader(share):
            page.evaluate("""(share) => {
                const box = window.isis.reader.scrollDOM;
                box.scrollTop = (box.scrollHeight - box.clientHeight) * share;
            }""", share)
            page.wait_for_timeout(500)  # the position is saved shortly after scrolling stops
            return top_of_reader()

        longest = page.evaluate("""async () => {  // leaving out this script's own files, just deleted
            const paths = (await (await fetch("api/files")).json()).map((f) => f.path).filter((p) => !p.startsWith("e2e-")).slice(0, 40);
            const sized = await Promise.all(paths.map(async (path) =>
                [(await (await fetch(`api/file?path=${encodeURIComponent(path)}`)).json()).content.length, path]));
            return sized.sort((a, b) => b[0] - a[0]).slice(0, 2).map((s) => s[1]);
        }""")
        first, second = longest
        page.evaluate("localStorage.removeItem('isis.reader-scroll'); window.isis.state.readerScroll = {}")
        open_from_files(first)
        page.evaluate("document.querySelector('#reader .cm-scroller').scrollTop = 0")
        first_at = scroll_reader(0.6)
        assert first_at > 0
        open_from_files(second)
        assert page.evaluate("window.isis.reader.scrollDOM.scrollTop") == 0  # never opened: at the top
        second_at = scroll_reader(0.3)
        open_from_files(first)
        assert abs(top_of_reader() - first_at) < 200, (top_of_reader(), first_at)
        open_from_files(second)
        assert abs(top_of_reader() - second_at) < 200, (top_of_reader(), second_at)
        scroll_reader(0)  # back at the top is remembered too
        open_from_files(first)
        page.set_viewport_size({"width": 1100, "height": 800})  # another width moves the same text
        page.wait_for_timeout(500)
        scrolled_at = scroll_reader(0.5)
        open_from_files(second)
        assert page.evaluate("window.isis.reader.scrollDOM.scrollTop") == 0
        page.set_viewport_size({"width": 1500, "height": 800})
        page.wait_for_timeout(500)
        open_from_files(first)
        assert abs(top_of_reader() - scrolled_at) < 300, (top_of_reader(), scrolled_at)
        page.reload()
        expect(page.locator("#new-task")).to_be_visible()
        page.wait_for_timeout(500)
        open_from_files(first)
        assert abs(top_of_reader() - scrolled_at) < 300, (top_of_reader(), scrolled_at)  # also after a reload
        open_from_files(second)
        page.evaluate("(path) => window.isis.openFile(path, { line: 5 })", first)  # a result goes to its match
        page.wait_for_timeout(300)
        assert top_of_reader() < scrolled_at / 2, (top_of_reader(), scrolled_at)
        step("Files opens a document where it was last read, or at the top the first time; a result still opens at its match")

        reader_box, task_box = "#reader .find-panel input", "#task .find-panel input"
        page.evaluate("(path) => window.isis.openFile(path)", first)
        page.wait_for_timeout(300)
        page.click("#reader .cm-content")
        page.keyboard.press("Control+f")
        expect(page.locator(reader_box)).to_be_focused()
        expect(page.locator(task_box)).to_have_count(0)  # only the pane that was clicked
        page.keyboard.type("the")
        expect(page.locator("#reader .find-count")).to_have_text(re.compile(r"\d\D+\d"), timeout=5000)
        match_total = lambda: re.findall(r"[\d,]+", page.locator("#reader .find-count").inner_text())[-1]
        total = match_total()
        scrolled = page.evaluate("window.isis.reader.scrollDOM.scrollTop")
        for _ in range(200):
            page.keyboard.press("Enter")
        assert match_total() == total
        assert page.evaluate("window.isis.reader.scrollDOM.scrollTop") > scrolled + 50  # the scrollbar follows
        chosen = page.evaluate("""() => {
            const s = window.isis.reader.state, m = s.selection.main, box = window.isis.reader.scrollDOM.getBoundingClientRect();
            const at = window.isis.reader.coordsAtPos(m.from);
            return { word: s.sliceDoc(m.from, m.to), visible: Boolean(at) && at.top >= box.top && at.bottom <= box.bottom };
        }""")
        assert chosen["word"].lower() == "the" and chosen["visible"], chosen
        before = page.locator("#reader .find-count").inner_text()
        page.keyboard.press("Shift+Enter")
        assert page.locator("#reader .find-count").inner_text() != before
        page.click("#task .cm-content")
        page.keyboard.press("Control+f")
        expect(page.locator(task_box)).to_be_focused()
        expect(page.locator(reader_box)).to_be_visible()  # each pane keeps its own
        page.keyboard.type("zzzqqq")
        expect(page.locator("#task .find-count")).to_have_text(re.compile(r"^\D+$"))
        assert page.locator(reader_box).input_value() == "the"
        page.keyboard.press("Escape")
        expect(page.locator(task_box)).to_have_count(0)
        expect(page.locator(reader_box)).to_be_visible()
        page.evaluate("""() => {  // two words on either side of a line break are found
            const s = window.isis.taskView.state; window.isis.taskView.dispatch({ changes: { from: 0, to: s.doc.length, insert: "alpha\\nbeta and alpha beta\\n" } });
        }""")
        page.click("#task .cm-content")
        page.keyboard.press("Control+f")
        page.keyboard.press("Control+a")
        page.keyboard.type("alpha beta")
        expect(page.locator("#task .find-count")).to_have_text(re.compile(r"\D2$"))
        page.keyboard.press("Escape")
        page.click("#reader .cm-content")
        page.keyboard.press("Escape")
        expect(page.locator(reader_box)).to_have_count(0)
        step("Ctrl+F finds in the pane last clicked, each with its own box; Enter steps through the matches and the scrollbar follows")

        page.click(".side-tabs button[data-panel=related]")
        for text in ("The white whale and the sea, and Ahab.\\n", "The white whale, the sea, and the ship.\\n"):
            page.evaluate("""(text) => {
                const s = window.isis.taskView.state;
                window.isis.taskView.dispatch({ changes: { from: 0, to: s.doc.length, insert: text } });
            }""", text.replace("\\n", "\n"))
            expect(page.locator("#related-refresh")).to_be_enabled(timeout=10000)
            page.evaluate("document.querySelector('#panel-related').scrollTop = 0")
            page.click("#related-refresh")
            expect(page.locator("#related-lanes .hit").first).to_be_visible(timeout=20000)
            expect(page.locator("#related-status")).not_to_have_attribute("aria-busy", "true", timeout=30000)
            page.wait_for_timeout(200)
            gap = page.evaluate("""() => {
                const heading = document.querySelector('#related-lanes .lane h3');
                return { text: heading.textContent, gap: heading.getBoundingClientRect().top - document.querySelector('#panel-related').getBoundingClientRect().top };
            }""")
            assert gap["text"].startswith("Best results") and abs(gap["gap"]) < 1, gap
        step("Refresh brings the top of Best results to the top of the panel")

        def save_preset(kind, name):
            answers.append(name)
            page.click(f"#{kind}-presets .preset-save")
            expect(page.locator(f"#{kind}-preset")).to_have_value(name)

        page.click(".side-tabs button[data-panel=search]")
        page.evaluate("localStorage.removeItem('isis.presets'); window.isis.state.presets = { search: [], related: [] }; window.isis.renderPresets('search'); window.isis.renderPresets('related')")
        page.fill("#search-input", "white whale")
        expect(page.locator("#search-results .hit").first).to_be_visible(timeout=10000)
        expect(page.locator("#search-hidden")).to_be_hidden()
        hidden_path = page.locator("#search-results .hit").first.get_attribute("data-path")
        page.locator("#search-results .hit").first.locator(".exclude-doc").click()
        expect(page.locator("#search-hidden")).to_be_visible()
        page.wait_for_timeout(800)  # the search runs again
        expect(page.locator(f'#search-results .hit[data-path="{hidden_path}"]')).to_have_count(0)
        expect(page.locator("#search-results .hit").first).to_be_visible(timeout=10000)
        page.reload()  # this browser remembers it
        expect(page.locator("#new-task")).to_be_visible()
        page.click(".side-tabs button[data-panel=search]")
        page.fill("#search-input", "white whale")
        expect(page.locator("#search-results .hit").first).to_be_visible(timeout=10000)
        expect(page.locator(f'#search-results .hit[data-path="{hidden_path}"]')).to_have_count(0)
        page.click("#search-hidden > summary")
        page.locator("#search-hidden .undo").click()
        expect(page.locator("#search-hidden")).to_be_hidden()
        expect(page.locator(f'#search-results .hit[data-path="{hidden_path}"]').first).to_be_visible(timeout=10000)
        step("Search can hide a whole document, remembers it after a reload, and can bring it back")

        page.locator("#search-results .hit").first.locator(".exclude-doc").click()
        expect(page.locator("#search-hidden")).to_be_visible()
        page.fill("#search-input", "whale +/sp.rm/")
        page.click("#search-tune > summary")
        set_slider(page, "#search-tune [data-name=k1] input", 2.5)
        page.click("#search-meaning")
        fits = page.evaluate("""() => {  // the button that moved sliders show stays inside its row
            const row = document.querySelector('#search-tune .tune-row.changed'), button = row.querySelector('.tune-default');
            return { inside: button.getBoundingClientRect().right <= row.getBoundingClientRect().right + 0.5, whole: button.scrollWidth <= button.clientWidth };
        }""")
        assert fits == {"inside": True, "whole": True}, fits
        save_preset("search", "Regex by keywords")
        page.click("#search-tune .tune-reset")
        page.locator("#search-hidden .tune-reset").click()
        expect(page.locator("#search-hidden")).to_be_hidden()
        page.fill("#search-input", "something else")
        expect(page.locator("#search-preset option").first).to_contain_text("Regex by keywords")
        expect(page.locator("#search-preset option").first).to_have_attribute("value", "")  # marked as changed, not chosen
        kept = page.evaluate("window.isis.state.presets.search[0].settings")  # changes reach a preset only through Save
        assert kept["query"] == "whale +/sp.rm/" and kept["params"]["k1"] == 2.5, kept
        page.select_option("#search-preset", "Regex by keywords")
        expect(page.locator("#search-input")).to_have_value("whale +/sp.rm/")
        expect(page.locator("#search-hidden")).to_be_visible()  # the preset's hidden document
        expect(page.locator("#search-tune [data-name=k1] input")).to_have_value("2.5")
        expect(page.locator("#search-meaning")).to_have_attribute("aria-pressed", "false")
        expect(page.locator("#search-results .hit").first).to_be_visible(timeout=10000)
        page.locator("#search-hidden .tune-reset").click()
        page.click("#search-tune .tune-reset")
        page.fill("#search-input", "")
        page.click("#search-tune > summary")

        page.click(".side-tabs button[data-panel=related]")
        page.fill("#steer-input", "whale /sh.p/ -squid")
        page.press("#steer-input", "Enter")
        expect(page.locator("#steer-input")).to_have_attribute("data-saved", "whale /sh.p/ -squid", timeout=5000)
        page.click("#related-tune > summary")
        set_slider(page, "#related-tune [data-name=total_slots] input", 10)
        save_preset("related", "Short list")
        page.fill("#steer-input", "other")
        page.press("#steer-input", "Enter")
        expect(page.locator("#steer-input")).to_have_attribute("data-saved", "other", timeout=5000)
        page.click("#related-tune .tune-reset")
        page.focus("#steer-input")  # the box shows what is restored, also with the cursor in it
        page.select_option("#related-preset", "Short list")
        expect(page.locator("#steer-input")).to_have_value("whale /sh.p/ -squid", timeout=5000)
        expect(page.locator("#related-tune [data-name=total_slots] input")).to_have_value("10")
        page.reload()  # presets stay in this browser
        expect(page.locator("#new-task")).to_be_visible()
        page.click(".side-tabs button[data-panel=related]")
        expect(page.locator("#related-preset option", has_text="Short list")).to_have_count(1)
        page.click("#related-presets .preset-delete")
        expect(page.locator("#related-preset option", has_text="Short list")).to_have_count(0)
        expect(page.locator("#steer-input")).to_have_value("whale /sh.p/ -squid")  # deleting changes no settings
        page.evaluate("localStorage.removeItem('isis.params')")
        step("presets save, restore and delete the settings of Search and Related, including regexes")
        browser.close()

    for name in (WHALES, ISLANDS):
        request = urllib.request.Request(f"{URL}/api/tasks/{urllib.parse.quote(name)}", method="DELETE")
        urllib.request.urlopen(request).read()
    print("browser errors:", errors or "none")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
