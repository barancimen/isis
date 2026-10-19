from isis.engine.chunker import chunk_markdown, split_lines


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
