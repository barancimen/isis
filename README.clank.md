# isis

A vibe-coded local web app for working through a corpus of markdown files:

- **Read** any file, and **collect** selections into *task documents*: one markdown file per task, switched with vertical tabs.
- **Insert at the cursor.** Each selection goes in at the task document's cursor, as its own paragraph, even while you are editing that document. The task saves itself to disk.
- **Search** the whole corpus with BM25, optionally fused with embeddings. Each hit shows its context and location, and clicking it opens the file at the match.
- **Related** treats the task document itself as the query. It suggests passages in four lanes: core matches, hidden connections (same idea, other words), bridges between your themes, and gaps in your coverage. Embeddings come from small CPU models: [mdbr-leaf-ir](https://huggingface.co/MongoDB/mdbr-leaf-ir) by default, or [bge-micro-v2](https://huggingface.co/TaylorAI/bge-micro-v2), chosen in the status bar.

The search engine is plain Python (`isis/engine`), so you can take it apart in a notebook. The server is a thin FastAPI layer, and the page is plain JavaScript with CodeMirror.

## Quick start

```sh
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools   # editable installs need setuptools >= 64
.venv/bin/pip install -e ".[dev]"            # add ",notebook" for Jupyter
.venv/bin/python scripts/fetch_sample_corpus.py   # optional: three public-domain books as a corpus
.venv/bin/isis serve sample_corpus           # then open http://127.0.0.1:8000
```

Point `isis serve` at any folder of `.md` files, or at several: `isis serve dir1 dir2`. Subfolders are scanned at any depth, and symlinked folders and files are followed, so a folder from elsewhere can join the corpus through a link (a link loop ends, and anything reachable twice is indexed once). Hidden files and folders are skipped. The terminal reports what happens: how many files are new, changed or removed, embedding progress, and links it follows or skips.

- **Several folders.** They act as one corpus, and each appears under its own name in Files (two folders with the same name become `notes` and `notes-2`). Search and Related each have a box per folder; untick one to leave it out of that panel (remembered in this browser). Search and Related then only compare the ticked folders by meaning and only scan them for phrases and regexes, so they cost about as much as those folders. A phrase or regex's rarity, and how often two of the task's topics meet (for Connections between topics), are estimated from the ticked folders; a meaning score is still measured against the whole corpus, so its thresholds mean the same. Related still looks for collected excerpts in every folder. Each keeps its own index in its `.isis/`, so it can later be served alone, or with other folders, without embedding it again. Task documents go in `_tasks` of the first folder, unless you pass `--tasks`.
- **Folders too large to embed.** Put an empty file named `.no-embed` in a folder (`touch big/.no-embed`): it and its subfolders are then searched by keywords only. Adding the file drops embeddings already made for that folder; removing it embeds the folder. Meaning search can't find those passages. In a hybrid search their keyword rank counts in place of the meaning rank they lack, so strong keyword matches aren't buried; **Adjustments** › *Passages with no meaning data* sets how much it counts, in both Search and Related. In Search, with it and *How important keywords are* both at 0, those passages are left out. Related shows them in Core only, and they get in on a top keyword rank alone, with no check on meaning.

- **First run.** It downloads the models (about 90 MB and 67 MB) and embeds every passage with each of them in the background, the default model first. On one slow CPU core that runs at about 11 passages/s with mdbr-leaf-ir and 15 with bge-micro-v2, and much faster on a laptop. Keyword search works while it runs.
- **Optional Rust extension.** If Rust is installed (with `rustup`), `isis serve` builds the extension `isis_native` from `native/` in the background when it is missing or older than its source there, as after a `git pull` that changed it. The server starts at once and uses the Python code until the build ends (a few minutes). A missing extension is used as soon as it is built, and a rebuilt one from the next start. Without Rust, or with `--no-build`, nothing is built, and `.venv/bin/pip install ./native` builds it by hand. With it, a search builds its results about twice as fast, and the first search for a `"phrase"` scans the passages about ten times faster (on 20,000 passages a search takes about 27 ms where it took 35, and a new phrase about 75 ms where it took 300). Results are the same with or without it. A server that runs as a user who can't install into its Python environment, or without Rust, only says in the terminal that the extension is missing or old.
- **Later runs.** The index is cached in `<corpus>/.isis/`, and only changed files are re-read. The server watches the folder, so edits show up in search within seconds. A folder it can't read (changed permissions, an unmounted drive, a link whose target is away) keeps its files and embeddings in the index until it can be read again, and they just don't show up in results meanwhile. If its link is removed, they go.
- **Options:**
  - `--tasks DIR`: where task documents live (default `_tasks` in the first folder). Task folders are never indexed.
  - `--no-model`: keyword search only.
  - `--model REPO`: offer only this embedding model (repeatable; the first one named is the default). Without it, isis offers every model in `MODELS` in `isis/engine/semantic.py`: `MongoDB/mdbr-leaf-ir` (the default, 23M parameters) and `TaylorAI/bge-micro-v2` (17M). Any Hugging Face model with ONNX weights and a `tokenizer.json` can join that list. Text a model wants in front of queries or passages comes from the `prompts` of its `config_sentence_transformers.json`, so mdbr-leaf-ir gets its query prefix without any setting. Task paragraphs in Related are embedded like passages. Each model's vectors are kept in `<corpus>/.isis/models/`, matched to passages by their text, so a model left out of a run keeps its vectors and only embeds what changed meanwhile when it comes back. The Related thresholds were calibrated with bge-micro-v2, so another model may need them redone (see CALIBRATE.md).
  - `--port`, `--host`.
  - `--allow-host NAME`: also answer requests addressed to this name (repeatable; see below).
  - `--public`: serve the public behind a proxy (see Public mode below).
  - `--reload`: restart when the Python code changes, handy while you tune the algorithm.
  - `--no-build`: don't build the Rust extension, even if it is missing or old.
- `isis index DIR...` builds the index without starting the server.
- `isis search` and `isis related` ask a running server from the terminal (see Command line below).

**Nothing leaves the machine.** Once the model is cached (after the first run), isis makes no network requests at all: model files are read from the local Hugging Face cache without asking the network, Hugging Face telemetry is off, the page loads nothing from outside (no fonts, CDNs or analytics; FastAPI's `/docs` pages, which load from a CDN, are turned off), and the search fields have spellcheck off so a browser's cloud spellcheck never sees them. The only downloads are `pip install`, the model on the first run, `scripts/fetch_sample_corpus.py`, and building the Rust extension (maturin and the Rust libraries it uses), which `isis serve` does only when the extension is missing or old. To be certain, set `HF_HUB_OFFLINE=1`, which also stops that build, or run it without network access.

The server has no authentication. Keep the default `--host 127.0.0.1`, and use an SSH tunnel (`ssh -L 8000:127.0.0.1:8000 yourserver`) to reach it on a remote machine. It answers only requests addressed to this machine or to the `--host` address, so a web page can't reach it by pointing its own domain at 127.0.0.1 (DNS rebinding); `--allow-host` adds names. With `--host 0.0.0.0` and no `--allow-host`, any name is answered.

### Public mode

`isis serve CORPUS --public --tasks FOLDER --allow-host NAME` lets anyone use the site without logging in, behind a proxy that sets `CF-Connecting-IP`, such as a Cloudflare Tunnel. Install the extra library first: `pip install -e ".[public]"`. It refuses to start on a non-loopback `--host`, or with a tasks folder inside the corpus. Without `--public`, none of this applies. The code is in `isis/server/public.py`, and the numbers are in `Limits` there. To serve the site under a path on a domain, add `--base-path /isis`: the proxy forwards the path unchanged, and the cookie stays within it.

- **Private tasks.** An anonymous cookie picks each visitor's folder, named by a hash of the cookie. The folder appears with the first task. Clearing cookies loses access to those tasks. Deleted tasks are removed, not moved to `.trash`.
- **Read-only corpus.** `/api/file` serves only indexed files, so `_tasks` and hidden files in the corpus stay private. Everything indexed is public.
- **Space.** 20 tasks, 512 KB per task and 2 MB per visitor, 1 GB and 5,000 workspaces in all. Requests that store anything other than a task's text are capped at 16 KB, and a task can have 200 liked passages. A workspace unused for 30 days is deleted.
- **Rates per address.** Separate limits for search, Related, saves and everything else, and 5 new workspaces per hour. A long Related request counts as several. IPv6 addresses count per /64.
- **Work per address.** Searches and Related refreshes are charged afterwards by the time they took: 30 s at first, then 0.25 s per second. An address in debt gets a 429 until it is paid off, so a request can't cost more than it is charged for, whatever made it slow.
- **Busy limit.** Two searches or Related refreshes run at once. Others wait up to 3 s, then get a 503.
- **Browser headers.** The cookie is always `Secure`, so the site works only over HTTPS (browsers also accept it on localhost). Every answer has a content security policy, and API answers have `Cache-Control: no-store`.
- **Regexes.** They run through the `regex` library, with 3 s for each request. Syntax is still checked by `re`, so what is valid stays the same. Patterns over 300 characters or with counted repeats multiplying past 2,000 are refused, because `regex` expands repeats when it compiles and `(x{65535}){65535}` would use all the memory. Fuzzy syntax such as `{e<=1}` is refused too, as `re` reads it as plain text.

## Command line

`isis search` and `isis related` ask a running `isis serve` from the terminal, like ripgrep. Each call is on its own: it sends its query or task document, its word rules and its settings, and the server stores nothing, so the page and its settings stay as they are. They ask exactly what the page asks, so the same query or task text with the same settings gives the same results, in the same order. Only a task's stored preferences (ignored keywords, hidden or liked passages, hidden documents) are the page's alone. The server already holds the index and the models, so a call only waits for the answer (Related for a short task takes about 0.15 s on the sample corpus). The client uses only the standard library and starts in 0.1 s.

```sh
isis search 'whale /harpoon\w*/ -ship'          # the query in quotes, written as in the page
isis search '"natural selection"' -n 5 -o hits.md
isis related notes.md                            # passages related to a task document
cat notes.md | isis related - '+squid -ship' -o found.json
isis search 'whale' -p '{"k1": 1.5, "rrf_k": 30}'  # settings for this call only
```

Each result is one line: `[path:line]` and then the whole passage, its line breaks turned into spaces, with the matches highlighted in a terminal (`NO_COLOR` turns colors off). The line is the passage's first, so `path:line` opens it in an editor. Related puts the name of each group before its results.

| Option | Does |
|---|---|
| `'QUERY'` (search) | One argument in quotes, written as in the page: words, `"phrases"`, `/regex/`, `+x` (required), `-x` (excluded) and `site:host`. It is sent as written, so a last word that no passage contains matches the words it starts, as while typing in the page. A `-word` or a second word outside the quotes is refused, so nothing is read two ways (a query that starts with a dash and has no space goes after `--`). |
| `FILE ['RULES']` (related) | The task document, or `-` for stdin, then the word rules as one argument in quotes, as on the page's word rules line. |
| `-n N` | Show only the first N results. Search asks for 50, as the page does, and Related shows all its results. Neither changes what is found: for more or fewer Related results, set `total_slots` with `-p`, as the page's *Number of results* does. |
| `-p JSON` | Settings for this call, inline or as `@file.json`. The names are those of `/api/params`, which the page's Adjustments show. An unknown name or a value out of range is refused, with the list of names. |
| `-o FILE` | Also save the results. `.json` saves the server's whole answer, `.md` Markdown with the matches in bold, anything else the plain lines. |
| `-q` | Print nothing (with `-o`). |
| `-l` | Print only the documents, once each, best first. |
| `--json` | Print the server's answer as JSON. |
| `--folder NAME` | Only this corpus folder (repeatable). |
| `--keywords` (search) | Keywords only, as the page's Meaning button when it is off (`meaning_weight` 0). |
| `--as-typed` (search) | No spelling correction. A corrected search says what it searched for, on stderr. |
| `--model ID` | The embedding model (the server's default). |
| `--server URL` | Where `isis serve` runs: `http://127.0.0.1:8000`, or `ISIS_SERVER`. A server under a path works, as in `https://example.org/isis`. |

The exit code is 0 with results, 1 without and 2 on an error, as in ripgrep. Related for any text is `POST /api/related` (`text`, `steer`, `params`, `skip_folders`, `model`), which public mode limits like Related for a task.

## Using it

| | |
|---|---|
| **Layout** | Drag the border right of the task list, the side panel or the reader to resize it (or focus it and use the arrow keys); double-click a border to reset it. Widths are kept in this browser. |
| **Tasks** (left column) | **+** creates a task, which goes at the end of the list. A rename keeps a task's place, and the order is saved in `_tasks/.order.json`. Click a tab to switch. Double-click it or press F2 to rename it in place (Enter or clicking away renames, Escape doesn't), and press Delete (or Backspace) to delete it after a confirmation (deleted tasks go to `_tasks/.trash/`). Drag a tab (a see-through copy follows the mouse and a line shows where it will land; Escape or letting go outside the list calls it off), or press Alt+↑/↓ on it, to move it; ↑/↓ move the focus between tabs. `POST /api/tasks/{name}/move` with `{"before": name or null}` saves the new place. Each task keeps its own cursor and undo history while the page is open. Each task also has its own reader and its own search: the open document with its highlighted passage and scroll position, and the query, the opened result, the hidden documents and the Search panel's scroll position. This browser remembers them, so a new task starts with an empty reader and an empty search. Search's sliders, folders and presets are shared, and so are the Files panel and the choice of side panel. |
| **Refresh** | The Refresh button in Related gets new results and scrolls the panel so that "Best results" is at the top. |
| **Hiding documents** | ⊠ on a Search result hides that whole document from that task's Search, up to 50 documents per task. They are listed under **Hidden documents** to bring back, and this browser remembers them. Related keeps its own list for each task, on the server. |
| **Presets** | Above Adjustments in Search and in Related, **Save** stores the panel's settings under a name, choosing a name restores them, and **Delete** removes it. Search presets hold the sliders, the Meaning button, the folders, the hidden documents and the search box, with any regex. Related presets hold the sliders, the folders, the word rules, the ignored keywords and the hidden documents. Likes and hidden passages stay with their task. Presets are kept in this browser. |
| **Find** | Ctrl/⌘+F in the reader or the task opens a find box for that pane only. Enter and Shift+Enter step through the matches and scroll to each one, and Esc closes the box. The search ignores case, and a space also matches a line break. With the cursor elsewhere, it is the browser's own find. |
| **Results** | A result in Search or Related is headed by its document's title: `title:` in the frontmatter, or else a top-level heading near the start, or else the file name. The dimmed text around a passage is at most about 90 characters on each side, however long its lines are. |
| **Files** | Folders start collapsed; click one to open it. A document opened from here starts where you last left it in any task, remembered in this browser, or at the top the first time; a search or Related result still opens at its match. Filtering shows every match with its folders open. Opening a file from anywhere reveals it here, opening its folders, and a result also clears a filter that leaves its file out. |
| **Search** (Ctrl/⌘+K) | A query is a list of words, `"phrases"` and `/regexes/` (a regex is case-insensitive unless it has a capital letter). Plain ones are preferred: words rank by BM25, and a phrase or regex is scored like one more word, so `whale /white/` ranks passages mentioning both first. A phrase or regex on its own finds only its matches. `+x` requires x and `-x` excludes it. Words are not stemmed: `conditional` finds "conditional" and not "condition", and `whale` doesn't find "whales" (case never counts). A half-typed last word matches the terms it starts. The **Meaning** button (on by default) blends in embedding similarity, and each hit then shows its rank by keywords and by meaning; off is the same as *How important meaning is* at 0, and the two move together. The `+` and `-` clauses filter the corpus before anything is ranked, so meaning never brings back what they rule out. **Adjustments** opens sliders for BM25 and the fusion, each with a *default* button once changed, and *Set all to default* for all of them. Click a hit to open it with the midpoint of the passage in the middle of the reader, as far as the document can scroll; **+** adds the passage to the task. A plain word that no passage contains is corrected to the nearest one (1 typo from 4 letters, 2 from 8), and the status line shows the correction with a link to search as typed. If nothing is near, the app still searches by meaning. `+` and `-` clauses, phrases and regexes are never corrected. Words to prefer also match corpus words within the same number of typos, so a document that misspells `information` as `informaton` is found. They count *Words with typos* (0.5 by default, an Adjustments slider) times as much per typo, and are weighed as if they were as common as your word, so the right spelling still ranks first. A near word must start with the same letter (as in Lucene's spelling suggester), and counts only if one of the two is in at least ten times as many passages as the other, since a typo is far rarer than the word it misspells, while `form` and `farm` are two real words. A word typed with a typo therefore also finds the right spelling. Typos are counted using rapidfuzz on a list of every distinct word in the corpus (saved in the index cache). `+` and `-` words always match exactly. `site:github.com` shows only documents whose frontmatter `url:` is on that host or one of its subdomains (`final_url:` and `site:` in the frontmatter are not used). It works with other clauses or on its own, several `site:` clauses mean any of them, and `-site:github.com` leaves them out. Word rules in Related accept it too. |
| **Reader** | Select text and press **Ctrl/⌘+Enter** (or *+ Add to task*). It goes in at the task's cursor, as its own paragraph. Hard-wrapped text flows to the width of the pane and is justified, here, in the task and in Search and Related results (the file is unchanged), except for code, headings, table rows and YAML cards; a line break you type with Enter stays a real line until you leave the task. A block enveloped by `---` lines (front matter, or further down a `---` after a blank line, lines with no blank between them, and a closing `---`) shows as a card: compact with one line per key if it reads as YAML, otherwise as a normal paragraph. Standard markdown would make such text a heading. A standalone `---` shows as a thin line. Lists show real bullets that change with the level, each level clearly indented, with wrapped lines aligned under the text. On the line you are editing in a task, its real marker (`-`) takes the bullet's place, without moving anything. Enter starts the next item straight below. |
| **Status bar** | The menu next to the passage count picks the meaning model for Search and Related. This browser remembers the choice, and the first model the server offers is the default. When the page's files on the server have changed, the bar turns red and offers a reload. The page carries the version of the files it came with, and browsers are told not to keep them (`no-store`), so a reopened tab or a return through history gets the current page. The button saves every task first, and stays on the page if a save fails. Any reload of a tab keeps the open document with its highlighted passage, the search, open folders, the scroll positions, and each task's cursor and scroll position. A new tab starts with the last task's document and search, but with fresh folders and task cursors. |
| **Related** | Suggestions for the whole task document. They stay as they are while you read them: an edit, or anything you do here, only lights up **Refresh**, and they update when you press it (or when you switch tasks). The chips are the keywords taken from the task: click one to boost it (always searched for, at full weight), × to ignore it (no longer searched for, though passages containing it can still be suggested), then ⊘ to exclude it. Below them are the **word rules** (the code calls this the steer line), with the same syntax as Search: plain words, phrases and regexes are preferred, `+x` is required, `-x` is never shown. Boosting and excluding a chip just write `word` or `-word` there. On a suggestion, **+** adds it, **↑** asks for more like it (it pulls the suggestions toward it, in meaning and keywords, without joining the document; Adjustments › *How strong ↑ is* sets how strongly), × dismisses it, and ⊘ avoids it: it is hidden along with every passage close to it in meaning (Adjustments › *How near for ⊘*). Both teach the panel which lanes you find useful. ⊠ excludes the suggestion's whole document: nothing from it is suggested for this task any more, though Search still finds it. Hover a suggestion for its scores. **Adjustments** opens sliders for every threshold of the pipeline below, each with a *default* button once changed, and *Set all to default* for all of them. **Your changes** lists everything the panel remembers about the task, each item with its undo, and *Remove all changes*. |

Task documents are ordinary markdown files, so you can edit them elsewhere too. If a file changes on disk while it is open, the app asks before overwriting it. Per-task preferences (the steer line, ignored keywords, dismissed and avoided passages, excluded documents, feedback) live in `_tasks/.meta/<task>.json`. They belong to the task, not to its text, so they still apply after you rewrite the document. *Remove all changes* moves that file to `_tasks/.trash/`.

## How "Related" works

The whole pipeline is in [`isis/engine/related.py`](isis/engine/related.py), and every threshold is a field of `RelatedConfig`:

1. **Units.** The task is split into paragraphs. Paragraphs copied from the corpus are recognised as collected excerpts; a whole collected passage reuses its embedding from the index, so a long task is quick to refresh. Clusters of similar paragraphs become themes; headings play no part in them.
2. **Signals.** The whole document is the query. Every corpus passage is scored two ways:
   - its *robust z-score* of embedding similarity to the document vector (the length-weighted mean of the paragraph embeddings), meaning how far it stands out from the typical passage (above 50,000 passages, "typical" is estimated from a fixed sample of 20,000);
   - BM25 for the document's ~30 most distinctive terms.

   With `RelatedConfig(query="paragraphs")`, each paragraph is a query of its own instead. A passage then scores by its best match. Use this when a document covers unrelated topics and you want strong matches to any one of them, not to their overlap.
3. **Gate.** Only clearly relevant passages survive: a strong z-score, a top keyword hit with some semantic support, or a clear relation to two themes at once. Already-collected and dismissed passages, and every passage of an excluded document, are dropped.
4. **Lanes.**
   - *Best results* (`core`): reciprocal-rank fusion of both signals.
   - *Same meaning, different words* (`hidden`): high meaning, low keyword overlap.
   - *Connections between topics* (`bridges`): related to two of the task's themes (clusters of its paragraphs) that rarely meet in the corpus.
   - *Small topics* (`gaps`): related to themes with few collected excerpts.
5. **Selection.**
   - Each lane is filled by MMR, so results differ from each other and from what you have collected, with at most two passages per file.
   - Slots per lane are drawn by Thompson sampling from your add/dismiss counts. The draw is seeded the same way for every task and for `isis related`, so the same text gives the same results anywhere until feedback moves it.

The defaults were calibrated on the sample corpus with bge-micro-v2, against blind relevance judgments: 1,026 search results for 38 queries, and about 970 Related suggestions for 19 task documents (5 of them long, the size of a document after a session of collecting). That lowered BM25's length normalisation (b 0.35), raised the meaning thresholds of the Related filter (3.25 alone, 1.75 with a top keyword match), weighted your own notes above pasted excerpts, and turned the dismissal push off (it pushed relevant passages down as often as junk). The similarity scores of this model are compressed (robust z rarely passes 4); a different model or corpus may want different thresholds, so measure before changing them. With mdbr-leaf-ir, now the default, search scores 0.742 on the same judgments against 0.708, and the Related thresholds have not been recalibrated for it (see CALIBRATE.md).

## Playing with the algorithm

**In the page:** open **Adjustments** in the Search or Related panel. Every slider re-runs the panel as you move it; hover one for what it does and its default. Your settings stay in this browser only and are sent with each request, so the server's defaults never change.

**Against judgments:** [`CALIBRATE.md`](CALIBRATE.md) explains how the defaults were measured against about 1,800 graded results, and how to redo it with `calibration/` (for example after changing the embedding model).

**In a notebook:** [`notebooks/explore.ipynb`](notebooks/explore.ipynb) walks through it: typed search, `related()` with different configs and preferences, the z-score distributions, and how to try another model.

```python
from isis.engine.index import Engine
from isis.engine.semantic import Embedder
from isis.engine.related import RelatedConfig
from isis.engine.report import show_related

engine = Engine("sample_corpus", embedder=Embedder(), exclude=("sample_corpus/_tasks",))
engine.refresh()
show_related(engine.related(open("sample_corpus/_tasks/My task.md").read(), config=RelatedConfig(bridge_z=1.25)))
```

## Layout

```
isis/engine/   chunker.py   markdown -> passages with line ranges
               text.py      tokenising: lowercase words, stopwords (no stemming)
               lexical.py   BM25 as a sparse matrix (weighted queries = one mat-vec)
               semantic.py  ONNX embedders (MODELS: mdbr-leaf-ir, bge-micro-v2) + a hash stand-in for tests
               index.py     Engine: incremental index, on-disk cache, snapshots for readers
               search.py    typed search, phrase filter, snippets with highlights
               related.py   the task-as-query pipeline
               report.py    plain-text printing for notebooks
               native.py    the optional Rust extension, with the Python code as fallback
isis/server/   app.py       FastAPI app and file watcher; tasks.py: task files and preferences; public.py: public mode
isis/cli.py    `isis index` / `isis serve`; client.py: `isis search` / `isis related`, which ask a running server
native/        isis_native, the optional Rust extension (pyo3, built by maturin): highlights and phrase counts
web/           index.html, app.js, style.css, insert.js, reflow.js, filetree.js; vendor/codemirror.js (bundled, see build-codemirror.sh)
scripts/       fetch_sample_corpus.py, browser_check.py
calibration/   judged queries and tasks, and the scripts that score settings against them (see CALIBRATE.md)
```

## Development

```sh
.venv/bin/pytest                         # engine, related pipeline, server API
node --test web/*.test.mjs              # insertion and reflow logic of the page
python scripts/browser_check.py http://127.0.0.1:8765 sample_corpus   # end to end, needs playwright
sh web/vendor/build-codemirror.sh        # rebuild the CodeMirror bundle (needs Node)
.venv/bin/pip install ./native           # rebuild the Rust extension after changing native/src/lib.rs
```

Limitations:
- Tokenising drops English stopwords and never stems. An index cache from when words were stemmed is split into words again from its cached passages on the first start (no files read, no embeddings redone).
- Outside public mode, the app assumes one user at a time.
- The Rust extension knows a newer Unicode than Python 3.10 does (13.0), so a letter added since then can be highlighted, or matched in a phrase whatever its case, only with the extension. Everything else is held to the same answers by `tests/test_native.py`.
- Passages are whole lines, so a very long single-line paragraph becomes one passage. Its embedding averages several windows.
- The index cache (`<corpus>/.isis/index.json`, `index.npz` and `models/*.npz`) is plain data, so a corpus that comes with someone else's cache can't run code.
