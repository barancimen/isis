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
    assert [(p.start_line, p.end_line) for p in chunk_markdown("a.md", md, target_words=16, min_words=0)] == [(0, 2), (4, 4)]


def test_sections_with_the_same_title_stay_apart():
    md = "# Notes\na\n\n# Notes\nb"
    assert [p.text for p in chunk_markdown("a.md", md)] == ["a", "b"]


def test_long_runs_are_cut_into_nearly_equal_pieces():
    md = "\n".join(f"w{i} " * 10 for i in range(7))  # 7 lines of 10 words, one paragraph
    passages = chunk_markdown("a.md", md, max_words=30, overlap_words=0)
    assert [(p.start_line, p.end_line) for p in passages] == [(0, 1), (2, 4), (5, 6)]  # 20, 30, 20: no short tail


def test_pieces_of_a_cut_run_overlap():
    md = "\n".join(f"w{i} " * 10 for i in range(20))  # 200 words in lines of 10
    assert [(p.start_line, p.end_line) for p in chunk_markdown("a.md", md)] == [(0, 9), (8, 19)]


def test_a_piece_too_short_to_stand_alone_joins_its_neighbour():
    md = "long " * 170 + "\n" + "tail " * 15  # one very long line, then a short one
    assert [(p.start_line, p.end_line) for p in chunk_markdown("a.md", md)] == [(0, 1)]


def test_short_paragraphs_join_their_neighbours():
    body = "\n".join(["word " * 12] * 13)  # 156 words: too many to merge with up to the target
    md = f"# Essay\nBy Someone\n\n{body}\n\nParis, 2019\n\n# Notes\nA short note."
    assert spans(chunk_markdown("a.md", md))[0][1:3] == (1, 17)  # the byline, the body and the signature
    assert spans(chunk_markdown("a.md", md))[1] == ("Notes", 20, 20, "A short note.")  # nothing else under its heading
    assert len(chunk_markdown("a.md", md, target_words=0)) == 4  # every paragraph on its own


def test_code_fences_keep_blank_lines_and_hashes():
    md = "```\n# not a heading\n\ncode\n```\nafter"
    passages = chunk_markdown("a.md", md, target_words=0)
    assert len(passages) == 1 and passages[0].end_line == 5


def test_blocks_without_words_are_dropped():
    assert chunk_markdown("a.md", "text\n\n---\n\n***") == chunk_markdown("a.md", "text")


def test_line_splitting_matches_codemirror():
    assert split_lines("a\r\nb\rc\nd") == ["a", "b", "c", "d"]


def test_embed_text_is_the_passage_alone():
    (p,) = chunk_markdown("a.md", "# Whales\nThey dive.")
    assert p.heading == "Whales" and p.embed_text == "They dive."


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


def test_a_heading_line_is_read_in_one_pass():
    """Anyone may send a task to Related, and its lines go through these patterns: a line made to
    be slow (spaces before more text) once took over a minute, with everything else waiting."""
    import time

    from isis.engine.chunker import heading_text

    started = time.perf_counter()
    assert heading_text("# a" + " " * 200_000 + "b").startswith("a ")
    assert document_title(split_lines("---\ntitle:" + " " * 200_000 + "x" + " " * 200_000 + "y\n---\ntext")).startswith("x ")
    assert time.perf_counter() - started < 1
    assert [heading_text(line) for line in ("# Title", "## Title ##", "# C#", "#  ##", "#hashtag", "####### seven")] == \
        ["Title", "Title", "C#", "", None, None]
    assert heading_text("# Results\x00 2019") == "Results 2019"  # a NUL would split the list of headings in the index cache


def test_code_fences_open_and_close_as_markdown_has_them():
    inline = "# Notes\n\n```pip install isis``` installs it.\n\n## Usage\n\nRun the server first.\n"
    assert [(p.heading, p.text) for p in chunk_markdown("a.md", inline, 0)] == \
        [("Notes", "```pip install isis``` installs it."), ("Usage", "Run the server first.")]
    assert document_title(split_lines("````\n```\n# inside\n```\n````\n# After\ntext\n")) == "After"  # a longer fence holds a shorter one
    assert document_title(split_lines("~~~\ncode\n```\n# inside\n~~~\n# After\ntext\n")) == "After"
    assert heading_texts("```\n# not one\n```\n# One\n") == ["One"]


def test_every_heading_of_a_document_is_with_one_of_its_passages():
    md = "# Moby Dick\n\n## Chapter 1\n\nCall me Ishmael.\n\nSome years ago.\n\n## Chapter 2\n\nMore text.\n\n## Notes\n\n### Sources\n"
    passages = chunk_markdown("a.md", md, 0)
    assert [(p.heading, p.section, p.text) for p in passages] == [
        ("Moby Dick\nChapter 1", "Chapter 1", "Call me Ishmael."),
        ("Moby Dick\nChapter 1", "Chapter 1", "Some years ago."),
        ("Notes\nSources\nChapter 2", "Chapter 2", "More text."),  # those after the last text go with it
    ]
    assert chunk_markdown("a.md", "# Only a heading\n") == []


def test_a_folded_title_and_a_byte_order_mark():
    assert document_title(split_lines("---\ntitle: >\n  A long title\n  that continues\nurl: https://example.org/x\n---\nText\n")) == "A long title that continues"
    assert document_title(split_lines("---\ntitle: |-\n  One line\n---\nText\n")) == "One line"
    assert document_site(split_lines("---\ntitle: >\n  A long title\nurl: https://example.org/x\n---\nText\n")) == "example.org"
