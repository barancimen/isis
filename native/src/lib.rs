//! Optional speedups for isis. Two jobs of a search are a loop over text in Python's regex
//! engine, and many times faster here: finding the words to highlight in the lines of a result,
//! and counting a phrase in every passage. isis works the same without this module
//! (isis/engine/native.py falls back to the Python code), and tests/test_native.py holds the
//! two to the same answers.

use std::collections::HashSet;
use std::sync::OnceLock;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBytes, PyString};
use regex::bytes::RegexBuilder;
use regex::Regex;

/// A word as isis's tokeniser sees it (`[^\W_]+` in text.py): a run of letters and digits.
fn word() -> &'static Regex {
    static WORD: OnceLock<Regex> = OnceLock::new();
    WORD.get_or_init(|| Regex::new(r"[\p{L}\p{N}]+").unwrap())
}

/// For each line, the (start, end) of every word whose lowercase form is in `terms` and is longer
/// than one character. Positions count characters, as Python's do, not bytes.
#[pyfunction]
fn term_spans(lines: Vec<PyBackedStr>, terms: HashSet<String>) -> Vec<Vec<(usize, usize)>> {
    let mut lower = String::new();
    lines.iter().map(|line| spans(line, &terms, &mut lower)).collect()
}

fn spans(line: &str, terms: &HashSet<String>, lower: &mut String) -> Vec<(usize, usize)> {
    let ascii = line.is_ascii();
    let (mut at_byte, mut at_char) = (0, 0);
    let mut out = Vec::new();
    for m in word().find_iter(line) {
        let word = m.as_str();
        lower.clear();
        if word.is_ascii() {
            lower.push_str(word);
            lower.make_ascii_lowercase();
        } else {
            lower.push_str(&word.to_lowercase());
        }
        if lower.chars().nth(1).is_none() || !terms.contains(lower.as_str()) {
            continue;
        }
        if ascii {
            out.push((m.start(), m.end()));
        } else {
            at_char += line[at_byte..m.start()].chars().count();
            at_byte = m.start();
            out.push((at_char, at_char + word.chars().count()));
        }
    }
    out
}

/// One word of a phrase as a pattern. Python's `re` takes i, I, İ and ı for one letter when it
/// ignores case, which Unicode's simple case folding (what the regex crate uses) does not.
fn literal(word: &str) -> String {
    let mut out = String::new();
    for c in word.chars() {
        if matches!(c, 'i' | 'I' | '\u{130}' | '\u{131}') {
            out.push_str(r"[iI\x{130}\x{131}]");
        } else {
            out.push_str(&regex::escape(c.encode_utf8(&mut [0; 4])));
        }
    }
    out
}

/// How often the phrase `words` is in each text: its words in order with any whitespace between
/// them, in any case. The same matches as `phrase_pattern` in search.py finds with Python's `re`,
/// whose `\s` also takes the four separators \x1c to \x1f.
///
/// Each text is encoded for the scan and the copy dropped afterwards. Borrowing Python's own
/// UTF-8 form would leave a second copy of every passage with a non-ASCII character in memory
/// for as long as the index lives. Other requests run during the scan.
#[pyfunction]
fn count_phrase<'py>(py: Python<'py>, texts: Vec<Bound<'py, PyString>>, words: Vec<String>) -> PyResult<Vec<u32>> {
    if words.iter().any(|w| w.is_empty()) || words.is_empty() {
        return Err(PyValueError::new_err("a phrase needs words"));
    }
    let pattern = words.iter().map(|w| literal(w)).collect::<Vec<_>>().join(r"[\s\x1c-\x1f]+");
    let phrase = RegexBuilder::new(&pattern)
        .case_insensitive(true)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let encoded: Vec<Bound<'py, PyBytes>> = texts.iter().map(|t| t.encode_utf8()).collect::<PyResult<_>>()?;
    let haystacks: Vec<&[u8]> = encoded.iter().map(|b| b.as_bytes()).collect();
    Ok(py.detach(|| haystacks.iter().map(|text| phrase.find_iter(text).count() as u32).collect()))
}

#[pymodule]
fn isis_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(term_spans, m)?)?;
    m.add_function(wrap_pyfunction!(count_phrase, m)?)?;
    Ok(())
}
