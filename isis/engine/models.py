"""The embedding models of a corpus: the ones it has, and the active one, which is the one
`isis serve` and `isis index` load unless they are told to load others (see cli.models). `isis model`
changes them while isis is stopped.

They are kept in .isis/models.json of the first corpus folder, as {"models": [...], "active": ...}.
A corpus without that file has the models in MODELS, the first of them active."""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from .semantic import MODELS

FILE = "models.json"
HUB = re.compile(r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/", re.IGNORECASE)
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?")
PAGES = ("tree", "blob", "resolve", "discussions", "commits", "commit")  # after a repository's name, in a page's address


class ModelError(Exception):
    """Something `isis model` can't do, or a list of models that can't be read, in words for the
    person at the terminal."""


def repo_id(text: str) -> str:
    """The Hugging Face repository that `text` names, as its id, such as MongoDB/mdbr-leaf-ir.
    The address of one of its pages names it too."""
    name, address = HUB.subn("", text.strip())
    if address:
        parts = re.split(r"[?#]", name)[0].strip("/").split("/")
        cut = next((i for i, part in enumerate(parts) if part in PAGES), len(parts))
        name = "/".join(parts[:min(cut, 2)])
    if not REPO.fullmatch(name):
        raise ModelError(f"{text.strip() or 'an empty name'} is not the name of a Hugging Face repository, "
                         "which looks like MongoDB/mdbr-leaf-ir")
    return name


def listing(names: list[str], last: str = "and") -> str:
    """'A, B and C'."""
    return f" {last} ".join(filter(None, [", ".join(names[:-1]), *names[-1:]]))


def made_by(name: str, model: str) -> bool:
    """Whether the embedder by this name is this model's. An embedder's name starts with its
    model's, and goes on with what else changes its vectors (see Embedder.name)."""
    name, model = name.lower(), model.lower()
    return name == model or name.startswith(f"{model}:")


@dataclass
class ModelList:
    path: Path  # the file it is kept in
    models: list[str]
    active: str | None  # one of the models, or None when there are none

    @classmethod
    def of(cls, folder: str | Path) -> ModelList:
        """The models of the corpus whose first folder this is. Reading them writes nothing."""
        path = Path(folder) / ".isis" / FILE
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls(path, list(MODELS), MODELS[0])
        except OSError as e:
            raise ModelError(f"cannot read the list of models in {path}: {e.strerror or e}") from None
        try:
            data = json.loads(text)
            if not isinstance(data["models"], list):
                raise TypeError
            models = list(dict.fromkeys(repo_id(m) for m in data["models"]))
            listed = cls(path, models, None)
            if models:
                active = data.get("active") or models[0]
                listed.active = listed.find(repo_id(active))
                if listed.active is None:
                    raise ModelError(f"its active model, {active}, is not among its models")
        except (ValueError, LookupError, TypeError, AttributeError, ModelError) as e:
            why = str(e) if isinstance(e, ModelError) else "it should hold a list named models and the active one"
            raise ModelError(f"cannot use the list of models in {path}: {why}. Mend the file, or delete it "
                             f"to start again from {' and '.join(MODELS)}") from None
        return listed

    def find(self, repo: str) -> str | None:
        """The model of the list that `repo` names. Hugging Face reads a name the same in any
        case, so this does too."""
        return next((m for m in self.models if m.lower() == repo.lower()), None)

    def has(self) -> str:
        """'It has A, B and C.', for a message about a model that isn't one of them."""
        return f"It has {listing(self.models) or 'none'}."

    def add(self, repo: str, active: bool = True) -> str:
        """Puts a model on the list, and makes it the active one unless told not to. The only
        model is always active."""
        name = self.find(repo) or repo
        if name not in self.models:
            self.models.append(name)
        if active or self.active is None:
            self.active = name
        return name

    def use(self, repo: str) -> str:
        name = self.find(repo)
        if name is None:
            raise ModelError(f"{repo} is not one of this corpus's models. {self.has()} "
                             "Add it first with isis model add.")
        self.active = name
        return name

    def remove(self, repo: str, then: str | None = None) -> None:
        """Takes a model off the list. `then` names the model to be active afterwards, which has
        to be said when the active one goes and others are left."""
        name = self.find(repo)
        left = [m for m in self.models if m != name]
        if then is not None:
            chosen = next((m for m in left if m.lower() == then.lower()), None)
            if chosen is None:
                raise ModelError(f"{then} can't be the next active model. Without {name or repo}, this corpus has "
                                 f"{listing(left) or 'no model'}.")
        elif name == self.active and left:
            raise ModelError(f"{name} is the active model. Name the one to use next with --next, "
                             f"which can be {listing(left, 'or')}.")
        else:
            chosen = self.active if self.active != name else None
        self.models, self.active = left, chosen

    def save(self) -> None:
        """Writes the list, readable by a server that runs as another user."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Named as the index cache's own temporary files, which are cleared if a save is cut short.
        tmp = self.path.with_name(f"models.{os.getpid()}-{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps({"models": self.models, "active": self.active}, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)
