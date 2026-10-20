"""Plain-text views of search and related results, for notebooks and the terminal."""

from __future__ import annotations

import textwrap


def _excerpt(text: str, width: int) -> str:
    return textwrap.shorten(" ".join(text.split()), width, placeholder=" …")


def show_hits(hits, width: int = 110) -> None:
    for h in hits:
        print(f"{h.path}:{h.focus_line + 1}  {h.heading}  score={h.score:.3f}")
        print("    " + _excerpt(h.text, width))


def show_related(result, per_lane: int = 5, width: int = 110) -> None:
    print("terms: " + ", ".join(f"{c.label}{'*' if c.boosted else ''}" for c in result.terms if not c.ignored))
    ignored = [c.label for c in result.terms if c.ignored]
    if ignored:
        print("ignored: " + ", ".join(ignored))
    print("themes: " + " | ".join(f"{t.label} ({t.units} paragraphs, {t.excerpts} collected)" for t in result.themes))
    for lane in result.lanes:
        print(f"\n{lane.title} ({len(lane.items)})" + (f": {lane.note}" if lane.note else ""))
        for h in lane.items[:per_lane]:
            why = {k: v for k, v in h.why.items() if v is not None and k not in ("lane_score", "mmr")}
            print(f"  {h.path}:{h.start_line + 1}  {why}")
            print("      " + _excerpt(h.text, width))
    print(f"\n{result.stats}")


def show_table(rows: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    widths = [max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in cols]
    print("  ".join(str(c).ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))
