"""Tunable parameters: which config fields the web page may change, with ranges and help text.

A tunable field of a config dataclass carries metadata (label, group, range, help). `specs()`
describes those fields for the page, and `configure()` builds a config from the page's
overrides: unknown names are ignored and values are clamped to their ranges, so a stale or
hand-edited override can never break a query.
"""

from __future__ import annotations

from dataclasses import MISSING, field, fields


def tunable(default=MISSING, *, label: str, help: str, group: str, min=None, max=None, step=None,
            choices=None, default_factory=MISSING, keys=None):
    """A dataclass field the page may change. `keys` makes a dict field tunable key by key."""
    meta = {"label": label, "help": help, "group": group, "min": min, "max": max, "step": step,
            "choices": choices, "keys": keys}
    if default_factory is not MISSING:
        return field(default_factory=default_factory, metadata={"tunable": meta})
    return field(default=default, metadata={"tunable": meta})


def _tunables(cls):
    return [(f, f.metadata["tunable"]) for f in fields(cls) if "tunable" in f.metadata]


def _kind(value, meta) -> str:
    if meta["choices"]:
        return "choice"
    return "int" if isinstance(value, int) and not isinstance(value, bool) else "float"


def specs(cls) -> list[dict]:
    """One entry per tunable value; dict fields contribute one entry per key ("field.key")."""
    defaults = cls()
    out = []
    for f, meta in _tunables(cls):
        value = getattr(defaults, f.name)
        entries = [(f"{f.name}.{k}", f"{meta['label']}: {k}", value[k]) for k in meta["keys"]] if meta["keys"] else [
            (f.name, meta["label"], value)
        ]
        for name, label, default in entries:
            out.append({
                "name": name,
                "label": label,
                "group": meta["group"],
                "help": meta["help"],
                "type": _kind(default, meta),
                "min": meta["min"],
                "max": meta["max"],
                "step": meta["step"],
                "choices": meta["choices"],
                "default": default,
            })
    return out


def _coerce(value, default, meta):
    kind = _kind(default, meta)
    if kind == "choice":
        return value if value in meta["choices"] else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    if meta["min"] is not None:
        number = max(meta["min"], number)
    if meta["max"] is not None:
        number = min(meta["max"], number)
    return int(round(number)) if kind == "int" else number


def configure(cls, overrides: dict | None):
    """A `cls` instance with the given overrides applied, as far as they are valid."""
    config = cls()
    tunables = {f.name: meta for f, meta in _tunables(cls)}
    for name, value in (overrides or {}).items():
        field_name, _, key = str(name).partition(".")
        meta = tunables.get(field_name)
        if meta is None or bool(key) != bool(meta["keys"]) or (key and key not in meta["keys"]):
            continue
        current = getattr(config, field_name)
        coerced = _coerce(value, current[key] if key else current, meta)
        if coerced is None:
            continue
        if key:
            setattr(config, field_name, {**current, key: coerced})
        else:
            setattr(config, field_name, coerced)
    return config
