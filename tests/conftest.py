from pathlib import Path

import pytest

from isis.engine.index import Engine
from isis.engine.semantic import HashEmbedder

FILES = {
    "whales.md": "# Whales\n\nThe sperm whale dives deep for squid.\n\nAmbergris is found in whales.\n",
    "ships/pequod.md": "# The Pequod\n\nThe Pequod was a whaling ship.\n\nAhab was her captain, and he hunted the white whale.\n",
    "finches.md": "# Finches\n\nThe finches of the Galapagos differ in the shape of their beaks.\n",
    ".hidden/secret.md": "# Hidden\n\nAmbergris here should never be indexed.\n",
    "notes.txt": "Ambergris in a text file is not indexed either.\n",
}


def write_corpus(root: Path, files: dict[str, str] = FILES) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def corpus(tmp_path) -> Path:
    return write_corpus(tmp_path / "corpus")


@pytest.fixture
def engine(corpus, tmp_path) -> Engine:
    e = Engine(corpus, embedder=HashEmbedder(), index_dir=tmp_path / "index", target_words=0)
    e.refresh()
    return e
