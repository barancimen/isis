"""`isis search` and `isis related`: the command line client of a running `isis serve`.

Stateless like ripgrep: every call sends its query, word rules and settings, and the server
stores nothing. The server already holds the index and the models, so a call only waits for the
answer. This module uses the standard library alone, so it starts in a fraction of a second.

Each result is one line: [path:line] and the whole passage, its line breaks turned into spaces,
with the matches highlighted when printing to a terminal."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_SERVER = "http://127.0.0.1:8000"
DEFAULT_LIMIT = 10
RELATED_SLOTS = (4, 80)  # the range of the Related setting total_slots
TIMEOUT = 300  # seconds; the first Related call of a long task embeds all of it
VALUE_OPTIONS = {"-n", "--limit", "-p", "--params", "-o", "--output", "--folder", "--model", "--server"}
FLAG_OPTIONS = {"-q", "--quiet", "-l", "--files", "--json", "--keywords", "--as-typed", "-h", "--help"}
LOCATION, LINE, MATCH, HEADING, RESET = "\x1b[35m", "\x1b[32m", "\x1b[1;31m", "\x1b[1m", "\x1b[0m"


class Failure(Exception):
    """Something to say in one line before leaving with exit code 2."""


# ---- talking to the server ----


def fetch(server: str, method: str, path: str, query: dict | None = None, body: dict | None = None) -> dict:
    """One request to the server's JSON API. Tests replace this function."""
    url = server.rstrip("/") + path + ("?" + urllib.parse.urlencode(query) if query else "")
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8")).get("detail")
        except (ValueError, AttributeError):
            detail = None
        if path == "/api/related" and e.code in (404, 405):  # a server from before this command: the page's files answer
            raise Failure(f"the server at {server} is older than this command. Restart it with the current isis.") from None
        raise Failure(str(detail) if detail else f"the server at {server} answered {e.code} {e.reason}") from None
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        reason = getattr(e, "reason", e)
        reason = getattr(reason, "strerror", None) or reason
        raise Failure(f"isis found no server at {server} ({reason}). Start one with isis serve FOLDER or pass --server.") from None


# ---- reading the command line ----


def _parser(command: str) -> argparse.ArgumentParser:
    if command == "search":
        p = argparse.ArgumentParser(
            prog="isis search", description="Search the corpus of a running isis server.",
            epilog='Write the query as in the page, with words, "phrases", /regex/, +required, -excluded and site:host. '
            'Settings are the names of the sliders in Adjustments.')
        p.add_argument("query", nargs="+", help="what to look for (several words need no quotes)")
        p.add_argument("--keywords", action="store_true", help="keywords only, no meaning")
        p.add_argument("--as-typed", action="store_true", help="don't correct misspelled words")
    else:
        p = argparse.ArgumentParser(
            prog="isis related", description="Find passages related to a task document, from a running isis server.",
            epilog='Write word rules as in the page, with words, "phrases", /regex/, +required and -excluded. '
            'Settings are the names of the sliders in Adjustments.')
        p.add_argument("file", help="the task document, or - to read it from stdin")
        p.add_argument("rules", nargs="*", help="word rules, such as +squid -ship /harpoon\\w*/")
    p.add_argument("-n", "--limit", type=int, default=DEFAULT_LIMIT, metavar="N", help=f"how many results (default {DEFAULT_LIMIT})")
    p.add_argument("-p", "--params", metavar="JSON", help="settings for this call, as JSON or @file.json")
    p.add_argument("-o", "--output", metavar="FILE", help="also save the results (.json, .md or plain text)")
    p.add_argument("-q", "--quiet", action="store_true", help="print nothing (with -o)")
    p.add_argument("-l", "--files", action="store_true", help="print only the documents")
    p.add_argument("--json", action="store_true", help="print the server's answer as JSON")
    p.add_argument("--folder", action="append", default=[], metavar="NAME", help="only this corpus folder (repeatable)")
    p.add_argument("--model", default="", metavar="ID", help="the embedding model (default the server's)")
    p.add_argument("--server", default=os.environ.get("ISIS_SERVER") or DEFAULT_SERVER, metavar="URL",
                   help=f"where isis serve runs (default {DEFAULT_SERVER}, or ISIS_SERVER)")
    return p


def clauses_last(args: list[str]) -> list[str]:
    """The arguments with every positional one after "--". A query clause such as -ship starts
    with a dash, and argparse would take it for an option (-open even for -o pen). Only the
    options defined here are options. Any other argument with one leading dash is a clause."""
    options, positional, i = [], [], 0
    while i < len(args):
        a = args[i]
        if a == "--":
            positional += args[i + 1:]
            break
        if a in VALUE_OPTIONS:
            options += args[i:i + 2]
            i += 2
            continue
        if a in FLAG_OPTIONS or (a.startswith("--") and len(a) > 2):  # an unknown --option is still an error
            options.append(a)
        else:
            positional.append(a)
        i += 1
    return options + ["--"] + positional


def _read_json(text: str) -> dict:
    if text.startswith("@"):
        try:
            text = Path(text[1:]).read_text(encoding="utf-8")
        except OSError as e:
            raise Failure(f"could not read {text[1:]} ({e.strerror or e})") from None
    try:
        value = json.loads(text)
    except ValueError as e:
        raise Failure(f"the settings aren't valid JSON ({e})") from None
    if not isinstance(value, dict):
        raise Failure('the settings must be a JSON object, such as {"k1": 1.5}')
    return value


def check_params(params: dict, specs: list[dict]) -> None:
    """Refuse a name the server doesn't know or a value outside its range, which the server would
    quietly ignore or clamp."""
    known = {s["name"]: s for s in specs}
    for name, value in params.items():
        spec = known.get(name)
        if spec is None:
            raise Failure(f"there is no setting named {name}. These are the settings\n" + describe(specs))
        if spec["type"] == "choice":
            if value not in spec["choices"]:
                raise Failure(f"{name} can be {' or '.join(map(json.dumps, spec['choices']))}")
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Failure(f"{name} needs a number")
        elif spec["type"] == "int" and value != int(value):
            raise Failure(f"{name} needs a whole number")
        elif (spec["min"] is not None and value < spec["min"]) or (spec["max"] is not None and value > spec["max"]):
            raise Failure(f"{name} goes from {spec['min']} to {spec['max']}")


def describe(specs: list[dict]) -> str:
    width = max((len(s["name"]) for s in specs), default=0)
    rows = []
    for s in specs:
        span = " or ".join(map(json.dumps, s["choices"])) if s["type"] == "choice" else f"{s['min']} to {s['max']}"
        rows.append(f"  {s['name']:<{width}}  default {json.dumps(s['default'])}, {span}")
    return "\n".join(rows)


def skipped_folders(server: str, wanted: list[str]) -> list[str]:
    """The corpus folders to leave out, when only `wanted` are asked for."""
    if not wanted:
        return []
    folders = fetch(server, "GET", "/api/status").get("folders") or []
    if not folders:
        raise Failure("this server has one corpus folder, so there is no folder to choose")
    unknown = [f for f in wanted if f not in folders]
    if unknown:
        raise Failure(f"there is no folder named {unknown[0]}. The folders are {', '.join(folders)}")
    return [f for f in folders if f not in wanted]


# ---- what is shown ----


def passage(hit: dict, highlight: tuple[str, str] | None) -> str:
    """The whole passage on one line, with its matches between `highlight`'s two strings."""
    text = hit["text"]
    lines = text.split("\n")
    starts, at = [], 0
    for line in lines:
        starts.append(at)
        at += len(line) + 1
    spans = []
    for line, start, end in hit.get("marks") or []:
        i = line - hit["start_line"]
        if 0 <= i < len(lines) and 0 <= start < end <= len(lines[i]):  # not a heading above the passage
            spans.append((starts[i] + start, starts[i] + end))
    if highlight:
        for start, end in sorted(spans, reverse=True):
            text = text[:start] + highlight[0] + text[start:end] + highlight[1] + text[end:]
    return text.replace("\n", " ")


def location(hit: dict, color: bool) -> str:
    line = hit["start_line"] + 1
    if color:
        return f"[{LOCATION}{hit['path']}{RESET}:{LINE}{line}{RESET}]"
    return f"[{hit['path']}:{line}]"


def render(groups: list[tuple[str, list[dict]]], style: str, files: bool) -> str:
    """The results as text. `style` is "color" (a terminal), "plain" or "markdown"."""
    color = style == "color"
    highlight = {"color": (MATCH, RESET), "markdown": ("**", "**")}.get(style)
    out = []
    if files:
        seen = dict.fromkeys(hit["path"] for _, hits in groups for hit in hits)
        return "".join(f"{path}\n" for path in seen)
    for title, hits in groups:
        if title and hits:
            out.append(f"{HEADING}{title}{RESET}" if color else f"## {title}" if style == "markdown" else title)
        for hit in hits:
            out.append(f"{location(hit, color)} {passage(hit, highlight)}")
    sep = "\n\n" if style == "markdown" else "\n"
    return sep.join(out) + "\n" if out else ""


def save(path: str, answer: dict, groups: list[tuple[str, list[dict]]], files: bool) -> None:
    if path.lower().endswith(".json"):
        text = json.dumps(answer, ensure_ascii=False, indent=2) + "\n"
    else:
        text = render(groups, "markdown" if path.lower().endswith((".md", ".markdown")) else "plain", files)
    try:
        Path(path).write_text(text, encoding="utf-8")
    except OSError as e:
        raise Failure(f"could not save {path} ({e.strerror or e})") from None


# ---- the commands ----


def run_search(a, params: dict) -> tuple[dict, list[tuple[str, list[dict]]]]:
    query = {
        "q": " ".join(a.query) + " ",  # a final space, or the last word would count as half typed
        "mode": "lexical" if a.keywords else "hybrid",
        "limit": a.limit,
        "params": json.dumps(params),
        "skip": json.dumps(skipped_folders(a.server, a.folder)),
        "as_typed": "true" if a.as_typed else "false",
    }
    if a.model:
        query["model"] = a.model
    answer = fetch(a.server, "GET", "/api/search", query=query)
    if answer.get("corrected"):
        print(f'searched for "{answer["corrected"].strip()}" instead. Pass --as-typed to search as typed.', file=sys.stderr)
    return answer, [("", answer.get("hits") or [])]


def run_related(a, params: dict) -> tuple[dict, list[tuple[str, list[dict]]]]:
    try:
        text = sys.stdin.read() if a.file == "-" else Path(a.file).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise Failure(f"could not read {a.file} ({e.strerror or e})") from None
    params = {"total_slots": min(max(a.limit, RELATED_SLOTS[0]), RELATED_SLOTS[1]), **params}
    body = {"text": text, "steer": " ".join(a.rules), "params": params,
            "skip_folders": skipped_folders(a.server, a.folder), "model": a.model}
    answer = fetch(a.server, "POST", "/api/related", body=body)
    groups, left = [], a.limit
    for lane in answer.get("lanes") or []:
        items = lane.get("items") or []
        groups.append((lane.get("title", ""), items[:max(left, 0)]))
        left -= len(items)
    if not any(hits for _, hits in groups):
        note = next((lane.get("note") for lane in answer.get("lanes") or [] if lane.get("note")), "")
        if note:
            print(note, file=sys.stderr)
    return answer, groups


def main(argv: list[str]) -> int:
    command, rest = argv[0], argv[1:]
    a = _parser(command).parse_args(clauses_last(rest))
    try:
        if a.limit < 1:
            raise Failure("-n needs a number of 1 or more")
        params = _read_json(a.params) if a.params else {}
        if params:
            check_params(params, fetch(a.server, "GET", "/api/params")[command])
        answer, groups = (run_search if command == "search" else run_related)(a, params)
        if a.output:
            save(a.output, answer, groups, a.files)
        if a.json and not a.quiet:
            print(json.dumps(answer, ensure_ascii=False, indent=2))
        elif not a.quiet:
            color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
            sys.stdout.write(render(groups, "color" if color else "plain", a.files))
    except Failure as e:
        print(e, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:  # piped into head: say nothing more
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    return 0 if any(hits for _, hits in groups) else 1
