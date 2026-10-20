# Calibrating isis

How the default parameters were chosen, and how to choose them again: after switching the
embedding model, changing the chunking, pointing isis at a very different corpus, or changing the
algorithm.

The short version: **judge, don't eyeball.** Results that look fine one query at a time can be
37% junk, and a change that fixes the query you are looking at often breaks five others. Build a
small set of judged queries and tasks once, then every change is a number you can compare.

## What is in `calibration/`

| | |
|---|---|
| `lib.py` | Loads the engine, names passages (`path:start_line`), parses `--set` / `--grid` overrides. |
| `judge.py` | Records your grades for a printed pool, by position. |
| `search.py` | Search: `pool` results for judging, `eval` settings by nDCG@10. |
| `related.py` | Related: `build` task documents, `pool` suggestions, `eval` settings, `signals` tables, `feedback` (dismiss/avoid simulation). |
| `data/queries.json` | 38 search queries in three kinds: `keyword`, `question`, `paraphrase`. |
| `data/augment.json` | Helper queries per search query, to find relevant passages that no setting ranks well. |
| `data/qrels_search.json` | 1,026 graded search results. |
| `data/tasks.json` | 19 task documents: notes, headings and excerpts (by passage), each with its intent. 14 are short (a first note and a few excerpts) and 5 are `"kind": "long"`: a session's worth of collecting, 28–42 paragraphs and 2,900–5,100 words, built from the good hits of 4–5 related search queries. |
| `data/task_docs.json` | Those tasks as text (`related.py build` regenerates it). |
| `data/qrels_related.json` | About 970 graded Related suggestions. |

All of it was made on `sample_corpus/` with bge-micro-v2 (2026-10-20). The judgments are about
passages, not about the model, so **they stay valid when you change the model**, as long as the
corpus and its chunking stay the same.

Every command takes overrides named as in the Parameters drawers and in `SearchConfig(...)` /
`RelatedConfig(...)`: `--set z_strong=3` fixes a value, `--grid z_strong=2.5,3,3.5` tries each
value, and several grids multiply. Dict parameters are set per key: `--set mmr_lambda.core=0.8`.

## Grading

Grade each passage against the query or the task's written intent, not against what the system
seems to be aiming at, and without looking at which setting produced it (the pool scripts don't
say).

- **2**: what you were looking for; you would collect it.
- **1**: partly relevant: mentions the topic meaningfully, useful context, or a weaker example.
- **0**: not relevant: an incidental word match, or off topic.

Pitfalls that came up:

- Index entries, tables of contents and footnotes match many words but are rarely useful: 0, or 1
  for a chapter outline that names the topic.
- One-line fragments ("Britain; and this is the result:—") are 0 even inside the right chapter.
- Be consistent across settings rather than generous: the numbers only compare settings.
- Write the intent of each task before judging, and judge against it.

## Recalibrating after a model change

```sh
export ISIS_MODEL=BAAI/bge-small-en-v1.5   # the new model; its index goes to calibration/.cache/
```

The first run embeds the whole corpus (minutes to an hour on a slow CPU). Then:

1. **Fill the pools.** A new model surfaces passages nobody has judged, and unjudged results count
   as irrelevant, which makes the new model look worse than it is.

   ```sh
   .venv/bin/python calibration/search.py pool --augment --grid rrf_k=10,30,60 --grid keyword_weight=0.5,1
   .venv/bin/python calibration/related.py pool --grid z_strong=2,2.5,3,3.5 --grid z_weak=0.5,1.5,2.5
   ```

   Read the printout and grade each block:

   ```sh
   .venv/bin/python calibration/judge.py qrels_search.json pool_search.json "q05: 2 0 1 ..." "q09: ..."
   .venv/bin/python calibration/judge.py qrels_related.json pool_related.json "t01: 2 1 0 ..." ...
   ```

   Pool the settings you intend to compare, and pool again around any setting that wins: the
   `judged=` column of `eval` should stay near 1.00 for every setting you take seriously.

2. **Look at the signals before sweeping.** The z-score thresholds depend on the model: each model
   compresses similarity differently (bge-micro rarely gives a robust z above 4).

   ```sh
   .venv/bin/python calibration/related.py signals
   ```

   It shows the share of relevant passages per band of document z-score, with and without a
   top-30 keyword rank. With bge-micro, passages without keyword support needed z ≥ 3 to be
   worth showing (73% relevant), and top keyword matches needed z ≥ 2 (61%). Put `z_strong`
   and `z_weak` near where your model crosses roughly 60–70% relevant. `hidden_z`, `gap_z` and
   `bridge_z` are on the same scale.

3. **Sweep one parameter at a time from the defaults**, to see which ones matter:

   ```sh
   .venv/bin/python calibration/related.py eval --grid z_strong=2.5,3,3.5,4 --top 10
   .venv/bin/python calibration/related.py eval --grid theme_distance=0.1,0.2,0.3,0.5
   .venv/bin/python calibration/search.py eval --mode hybrid --grid rrf_k=5,10,20,40,60
   ```

4. **Then grid the few that matter together**, judge what the best settings surface (step 1), and
   re-run. Prefer the middle of a plateau over a single best point, and check the wins/losses
   column: a setting that wins on average but loses on many tasks is fragile.

5. **Simulate feedback** if the semantic thresholds moved:

   ```sh
   .venv/bin/python calibration/related.py feedback --pool   # print and save what needs judging
   .venv/bin/python calibration/related.py feedback
   ```

6. **Change the defaults** in `isis/engine/search.py` (`SearchConfig`) and
   `isis/engine/related.py` (`RelatedConfig`), update `tests/test_params.py` if it names one, and
   run the tests and `scripts/browser_check.py`.

Keyword parameters (BM25 `k1`, `b`, `n_query_terms`) don't depend on the model; they only need
redoing for a different corpus or chunking.

## Recalibrating for another corpus

The judgments name passages of `sample_corpus/`, so a new corpus needs its own:

1. Write 30–40 queries in `data/queries.json` (copy the old file away first), a third of each
   kind: keyword queries, natural questions, and paraphrases that avoid the text's own words.
   Paraphrases are where meaning search earns its keep, and where pools miss the most, so add
   helper queries for them in `data/augment.json`.
2. Write 10–15 tasks in `data/tasks.json`, the way you really use the app: your own notes plus a
   few excerpts (passage keys from a pool printout), some with two `#` headings (to exercise
   bridges and gaps), some notes-only, and one very short. Write each task's intent. Add a few
   long ones (`"kind": "long"`, 30+ paragraphs): a task document after an hour of collecting
   behaves differently from a first note, and the first calibration missed that.
3. `related.py build`, then pool and judge as above. Expect about 1,000 judgments for search and
   600–800 for Related; it takes a few hours and is the whole value of the exercise.

Changing `target_words` / `max_words` (the chunking) moves passage boundaries, so it invalidates
every judgment key. Decide the chunking first.

## How to read the numbers

- **Search: nDCG@10.** Rewards relevant passages near the top, with grade 2 worth three times
  grade 1. Look at it per query kind: lexical settings mostly move the keyword queries, fusion
  settings mostly the paraphrases.
- **Related: utility per task**, overall and for short and long tasks separately. Each suggestion
  shown scores +1 (grade 2), +0.5 (grade 1) or −1 (grade 0). A suggestion is worth showing when it is relevant more than about 55% of the
  time. Utility therefore favours showing fewer, better suggestions over filling every slot.
  Also watch:
  - `prec`: the share of relevant suggestions;
  - `nonobv`: relevant finds with no top-30 keyword rank, the serendipity the hidden, bridges and
    gaps lanes exist for;
  - the precision per lane.

## What the 2026-10-20 calibration found

On `sample_corpus/` with bge-micro-v2. A starting point for intuition, not a law.

- **BM25 `b` 0.75 → 0.35** in both panels (search nDCG 0.62 → 0.66). Passages range from one
  line to 180 words; full length normalisation favoured short fragments.
- **Search fusion `rrf_k` 10 → 20.** 20–40 scored the same (0.71). 20 keeps a passage that
  ranks first on meaning alone visible. Keyword and meaning weights 1:1 and the candidate depth
  hardly mattered.
- **Related filter.**
  - `z_strong` 2.5 → 3.25 and `z_weak` 1.0 → 1.75 were the biggest gains. The old values let in
    bands that were 50–70% junk.
  - `lexical_top` 30 → 50.
- **`excerpt_weight` 1.0 → 0.5.** Your notes say what you want; pasted excerpts are full of
  incidental detail (a whiteness excerpt mentions polar bears, which pulled in Darwin on
  animal colours).
- **`theme_distance` 0.5 → 0.2.** Headingless tasks now split into themes, which fills the
  bridges and gaps lanes. Those lanes were the most precise ones (75–93%) but nearly empty.
- **`n_query_terms` 20 → 30, core `rrf_k` 60 → 30:** small gains.
- **`negative_weight` 0.5 → 0.** Dismissed suggestions are usually still on topic, so pushing
  away from them displaced relevant suggestions without reducing junk.
- **`avoid_z` 3.0 → 3.5.** At 3.0, avoiding one passage also hid relevant ones.
- **Didn't matter or got worse:** `hidden_max_obviousness`, `duplicate_cos`, `candidates`,
  `gap_z`, `mmr_lambda`, and `query=paragraphs` (slightly better here, but the whole-document
  query is the intended behaviour).
- **Overall.** Related precision went from 63% to 75%, utility per task from 2.1 to 6.9, better
  on 13 of 14 tasks. The one loss, a task mixing kelp forests and the glowing sea, is the kind
  the whole-document query handles worst.

### Second round: long task documents (also 2026-10-20)

The first round only had tasks of 1–4 paragraphs, but the app is meant to be used on a document
that keeps growing. Five long tasks were added and about 230 more suggestions judged.

- **The old defaults held up better than expected on long tasks** (utility 10.0 per task), but
  without headings a 44-paragraph task split into 36 themes with labels like "ye · three ·
  starbuck", and the first refresh took 10 s.
- **Collected passages reuse their corpus vectors** instead of being embedded again. The cold
  refresh went from 10.5 s to 0.4 s, and long-task utility went up (10.0 → 11.5): the corpus
  vector of a collected passage turned out to be a better query than its re-embedding under the
  task's heading. Short tasks lost a little (6.86 → 6.64).
- **Fewer, bigger themes cost relevance.** Capping themes at about √paragraphs and merging
  single-paragraph themes gave readable themes but dropped long-task utility to 8.4: small
  themes make the bridges and gaps lanes work like per-paragraph matching, which suits a long
  task. A cap of 2√paragraphs without merging cost nothing, so that is what `assign_themes` does.
- **Theme labels** rank words by frequency in the theme times rarity in the corpus, not by
  frequency alone.
- **`theme_distance` 0.2 → 0.15**: better on 3 tasks, worse on none.
- **`n_query_terms` 50 looked best (8.42) but was a spike** (40: 8.13, 55: 7.89), so it stays 30.
- **Overall**, on the same 968 judgments: utility 7.68 → 8.21 (short 6.86 → 7.18, long 10.0 →
  11.1), precision 76% → 77%, gaps lane 89% → 93% precise. Search is unchanged.

Known limits of this calibration:

- One person judged everything, on one small corpus.
- The tasks were written by the same person who judged them. The long tasks were built from
  passages already judged relevant to search queries, so they are tidier than a real session.
- bge-micro is weak: even at z 3–3.5 only 71% of suggestions were relevant.

A better model should let the thresholds discriminate more; measure rather than assume.
