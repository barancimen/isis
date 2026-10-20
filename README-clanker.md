# isis

A vibe-coded local web app for working through a corpus of markdown files:

- **Read** any file, and **collect** selections into *task documents*: one markdown file per task, switched with vertical tabs.
- **Insert at the cursor.** Each selection goes in at the task document's cursor, as its own paragraph, even while you are editing that document. The task saves itself to disk.
- **Search** the whole corpus with BM25, optionally fused with embeddings. Each hit shows its context and location, and clicking it opens the file at the match.
- **Related** treats the task document itself as the query. It suggests passages in four lanes: core matches, hidden connections (same idea, other words), bridges between your themes, and gaps in your coverage. Embeddings come from a small CPU model, [bge-micro-v2](https://huggingface.co/TaylorAI/bge-micro-v2).

The search engine is plain Python (`isis/engine`), so you can take it apart in a notebook. The server is a thin FastAPI layer, and the page is one HTML file with CodeMirror.

## Quick start

```sh
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools   # editable installs need setuptools >= 64
.venv/bin/pip install -e ".[dev]"            # add ",notebook" for Jupyter
.venv/bin/python scripts/fetch_sample_corpus.py   # optional: three public-domain books as a corpus
.venv/bin/isis serve sample_corpus           # then open http://127.0.0.1:8000
```

Point `isis serve` at any folder of `.md` files.

- **First run.** It downloads the model (67 MB) and embeds every passage in the background. That runs at about 15 passages/s on one slow CPU core, and much faster on a laptop. Keyword search works while it runs.
- **Later runs.** The index is cached in `<corpus>/.isis/`, and only changed files are re-read. The server watches the folder, so edits show up in search within seconds.
- **Options:**
  - `--tasks DIR`: where task documents live (default `<corpus>/_tasks`, which is never indexed).
  - `--no-model`: keyword search only.
  - `--model REPO`: another Hugging Face model with ONNX weights, such as `BAAI/bge-small-en-v1.5`.
  - `--port`, `--host`.
  - `--reload`: restart when the Python code changes, handy while you tune the algorithm.
- `isis index CORPUS` builds the index without starting the server.

The server has no authentication. Keep the default `--host 127.0.0.1`, and use an SSH tunnel (`ssh -L 8000:127.0.0.1:8000 yourserver`) to reach it on a remote machine.

## Using it

| | |
|---|---|
| **Layout** | Drag the border right of the task list, the side panel or the reader to resize it (or focus it and use the arrow keys); double-click a border to reset it. Widths are kept in this browser. |
| **Tasks** (left column) | **+** creates a task. Click a tab to switch; hover for rename ✎ and delete × (deleted tasks go to `_tasks/.trash/`). Each task keeps its own cursor and undo history while the page is open. |
| **Files** | Filter and open any file in the reader. |
| **Search** (Ctrl/⌘+K) | A query is a list of words, `"phrases"` and `/regexes/` (a regex is case-insensitive unless it has a capital letter). Plain ones are preferred: words rank by BM25, and a phrase or regex is scored like one more word, so `whale /white/` ranks passages mentioning both first. A phrase or regex on its own finds only its matches. `+x` requires x and `-x` excludes it; `-whale` also drops "whales" and "whaling". A half-typed last word matches the terms it starts. Tick *meaning* to blend in embedding similarity; each hit then shows its rank by keywords and by meaning. The `+` and `-` clauses filter the corpus before anything is ranked, so meaning never brings back what they rule out. **Parameters** opens sliders for BM25 and the fusion. Click a hit to open it at the match; **+** adds the passage to the task. |
| **Reader** | Select text and press **Ctrl/⌘+Enter** (or *+ Add to task*). It goes in at the task's cursor, as its own paragraph. |
| **Related** | Suggestions for the whole task document, refreshed when you pause typing (moving the cursor or selecting text changes nothing). The chips are the keywords taken from the task: click one to boost it (always searched for, at full weight), × to ignore it (no longer searched for, though passages containing it can still be suggested), then ⊘ to exclude it. Below them is the **steer line**, with the same syntax as Search: plain words, phrases and regexes are preferred, `+x` is required, `-x` is never shown. Boosting and excluding a chip just write `word` or `-word` there. On a suggestion, **+** adds it, × dismisses it, and ⊘ avoids it: it is hidden along with every passage close to it in meaning (Parameters › *Avoid: similarity*). Both teach the panel which lanes you find useful. Hover a suggestion for its scores. **Parameters** opens sliders for every threshold of the pipeline below. **Feedback** lists everything the panel remembers about the task, each item with its undo, and *Forget all*. |

Task documents are ordinary markdown files, so you can edit them elsewhere too. If a file changes on disk while it is open, the app asks before overwriting it. Per-task preferences (the steer line, ignored keywords, dismissed and avoided passages, feedback) live in `_tasks/.meta/<task>.json`. They belong to the task, not to its text, so they still apply after you rewrite the document. *Forget all* moves that file to `_tasks/.trash/`.

## How "Related" works

The whole pipeline is in [`isis/engine/related.py`](isis/engine/related.py), and every threshold is a field of `RelatedConfig`:

1. **Units.** The task is split into paragraphs. Paragraphs copied from the corpus are recognised as collected excerpts, and headings (or clusters of similar paragraphs) become themes.
2. **Signals.** The whole document is the query. Every corpus passage is scored two ways:
   - its *robust z-score* of embedding similarity to the document vector (the length-weighted mean of the paragraph embeddings), meaning how far it stands out from the typical passage;
   - BM25 for the document's ~30 most distinctive terms.

   With `RelatedConfig(query="paragraphs")`, each paragraph is a query of its own instead. A passage then scores by its best match, and `cursor_boost` can weight the paragraph at the cursor. Use this when a document covers unrelated topics and you want strong matches to any one of them, not to their overlap.
3. **Gate.** Only clearly relevant passages survive: a strong z-score, a top keyword hit with some semantic support, or a clear relation to two themes at once. Already-collected and dismissed passages are dropped.
4. **Lanes.**
   - *Core:* reciprocal-rank fusion of both signals.
   - *Hidden connections:* high meaning, low keyword overlap.
   - *Bridges:* related to two of the task's themes (its headings, or clusters of its paragraphs) that rarely meet in the corpus.
   - *Gaps:* related to themes with few collected excerpts.
5. **Selection.**
   - Each lane is filled by MMR, so results differ from each other and from what you have collected, with at most two passages per file.
   - Slots per lane are drawn by Thompson sampling from your add/dismiss counts.

The defaults were calibrated on the sample corpus with bge-micro-v2, against blind relevance judgments: 1,026 search results for 38 queries, and about 740 Related suggestions for 14 task documents. That lowered BM25's length normalisation (b 0.35), raised the meaning thresholds of the Related filter (3.25 alone, 1.75 with a top keyword match), weighted your own notes above pasted excerpts, and turned the dismissal push off (it pushed relevant passages down as often as junk). The similarity scores of this model are compressed (robust z rarely passes 4); a different model or corpus may want different thresholds, so measure before changing them.

## Playing with the algorithm

**In the page:** open **Parameters** in the Search or Related panel. Every slider re-runs the panel as you move it; hover one for what it does and its default. Your settings stay in this browser only and are sent with each request, so the server's defaults never change. The drawer also shows the same settings as Python (`SearchConfig(...)` or `RelatedConfig(...)`) to paste into the notebook.

**Against judgments:** [`CALIBRATE.md`](CALIBRATE.md) explains how the defaults were measured against about 1,800 graded results, and how to redo it with `calibration/` (for example after changing the embedding model).

**In a notebook:** [`notebooks/explore.ipynb`](notebooks/explore.ipynb) walks through it: typed search, `related()` with different configs and preferences, the z-score distributions, and a **hold-out evaluation**. That evaluation (`isis.engine.evaluate`) hides some of a task's excerpts and measures how many come back, and through which lane. `compare()` runs it over several tasks and configs.

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
               text.py      tokenising: lowercase, stopwords, Snowball stems
               lexical.py   BM25 as a sparse matrix (weighted queries = one mat-vec)
               semantic.py  ONNX embedder (bge-micro-v2 by default) + a hash stand-in for tests
               index.py     Engine: incremental index, on-disk cache, snapshots for readers
               search.py    typed search, phrase filter, snippets with highlights
               related.py   the task-as-query pipeline
               evaluate.py  hold-out recall, config comparison, synthetic tasks
               report.py    plain-text printing for notebooks
isis/server/   app.py       FastAPI app and file watcher; tasks.py: task files and preferences
isis/cli.py    `isis index` / `isis serve`
web/           index.html, app.js, style.css, insert.js; vendor/codemirror.js (bundled, see build-codemirror.sh)
scripts/       fetch_sample_corpus.py, browser_check.py
calibration/   judged queries and tasks, and the scripts that score settings against them (see CALIBRATE.md)
```

## Development

```sh
.venv/bin/pytest                         # engine, related pipeline, server API
node --test web/insert.test.mjs          # insertion logic of the page
python scripts/browser_check.py http://127.0.0.1:8765 sample_corpus   # end to end, needs playwright
sh web/vendor/build-codemirror.sh        # rebuild the CodeMirror bundle (needs Node)
```

Limitations:
- Tokenising is English-only (Snowball stemmer and stopwords).
- The app assumes one user at a time.
- Passages are whole lines, so a very long single-line paragraph becomes one passage. Its embedding averages several windows.
- The index cache is a pickle. It is safe because the app only reads the cache it wrote itself.
