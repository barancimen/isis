import pytest

from isis.engine.chunker import chunk_markdown, document_site, document_title, heading_texts, split_lines


def spans(passages):
    return [(p.heading, p.start_line, p.end_line, p.text) for p in passages]


def test_headings_frontmatter_and_line_numbers():
    md = "---\ntags: x\n---\nintro line\n\n# Title\npara one\nstill one\n\n## Sub\npara two"
    assert spans(chunk_markdown("a.md", md, target_words=0)) == [
        ("", 3, 3, "intro line"),
        ("Title", 6, 7, "para one\nstill one"),
        ("Sub", 10, 10, "para two"),
    ]


def test_short_paragraphs_merge_within_a_section_only():
    md = "# A\none two\n\nthree four\n\n# B\nfive"
    assert spans(chunk_markdown("a.md", md, target_words=10)) == [
        ("A", 1, 3, "one two\n\nthree four"),
        ("B", 6, 6, "five"),
    ]


def test_merging_stops_at_target_size():
    md = "\n\n".join(["word " * 8] * 3)
    assert [(p.start_line, p.end_line) for p in chunk_markdown("a.md", md, target_words=16)] == [(0, 2), (4, 4)]


def test_sections_with_the_same_title_stay_apart():
    md = "# Notes\na\n\n# Notes\nb"
    assert [p.text for p in chunk_markdown("a.md", md)] == ["a", "b"]


def test_long_runs_are_cut_at_line_boundaries():
    md = "\n".join(f"w{i} " * 10 for i in range(7))  # 7 lines of 10 words, one paragraph
    passages = chunk_markdown("a.md", md, max_words=30)
    assert [(p.start_line, p.end_line) for p in passages] == [(0, 2), (3, 5), (6, 6)]


def test_code_fences_keep_blank_lines_and_hashes():
    md = "```\n# not a heading\n\ncode\n```\nafter"
    passages = chunk_markdown("a.md", md, target_words=0)
    assert len(passages) == 1 and passages[0].end_line == 5


def test_blocks_without_words_are_dropped():
    assert chunk_markdown("a.md", "text\n\n---\n\n***") == chunk_markdown("a.md", "text")


def test_line_splitting_matches_codemirror():
    assert split_lines("a\r\nb\rc\nd") == ["a", "b", "c", "d"]


def test_embed_text_includes_heading():
    (p,) = chunk_markdown("a.md", "# Whales\nThey dive.")
    assert p.embed_text == "Whales\nThey dive."


def test_heading_texts_lists_every_heading_line():
    text = "---\ntitle: no\n---\n# One\n\ntext\n\n## Two ##\n\n# One\n```\n# not a heading\n```\n"
    assert heading_texts(text) == ["One", "Two", "One"]


def test_a_rule_at_the_top_is_not_frontmatter():
    md = "---\nfirst line of prose\n\nNote: more prose\n---\nafter rule\n"
    assert spans(chunk_markdown("a.md", md, target_words=0)) == [
        ("", 1, 1, "first line of prose"),
        ("", 3, 5, "Note: more prose\n---\nafter rule"),
    ]


def test_frontmatter_of_every_usual_shape_is_still_skipped():
    for head in ("tags: x", "title: A\n\nDate Created: 2020\ntags:\n  - one\n- two\n# a comment", "description: >\n  folded text", ""):
        passages = chunk_markdown("a.md", f"---\n{head}\n---\nbody text\n", target_words=0)
        assert [p.text for p in passages] == ["body text"], head


@pytest.mark.parametrize("text,title", [
    ('---\ntitle: "A \\"Quoted\\" Title"\nauthor: X\n---\n\n# Heading\n', 'A "Quoted" Title'),
    ("---\ntitle: Plain: with colon\n---\n\nText\n", "Plain: with colon"),
    ("---\ntitle: 'It''s'\n---\n", "It's"),
    ("---\ntags: [a]\n---\n\n# From the heading\n\nText\n", "From the heading"),
    ("# First\n\n# Second\n", "First"),
    ("## Not top level\n\nText\n", ""),
    ("```\n# in code\n```\n", ""),
    ("Just text\n", ""),
    ("", ""),
])
def test_document_title(text, title):
    assert document_title(split_lines(text)) == title


@pytest.mark.parametrize("text, site", [
    ('---\nurl: "https://GitHub.com/a/b?x=1"\nfinal_url: "https://web.archive.org/x"\n---\n', "github.com"),
    ("---\nfinal_url: https://web.archive.org/x\nurl: https://www.example.org:8080/\n---\n", "example.org"),
    ("---\nurl: gist.github.com/a\n---\n", "gist.github.com"),
    ("---\ntitle: No address\nsite: github.com\n---\n", ""),
    ("url: https://github.com\n", ""),
    ("", ""),
])
def test_document_site(text, site):
    assert document_site(split_lines(text)) == site
