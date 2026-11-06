"""Text perturbations for the shortcut stress test.

The audit's biggest red flag was a 0.99 -> 0.57 drop from in-distribution to an unseen language, which
smells of formatting artifacts rather than authorship signal. The test: strip the things a detector
should NOT need (comments, indentation style, blank lines) and re-score the SAME OOD rows. A model that
collapses was reading surface form; a model that holds up was reading something deeper.

Deliberately syntax-light (regex, no parser): the data mixes Python / C++ / Java and a parser-based
cleaner would silently no-op on two of them - exactly the bug the old radon features had.
"""
from __future__ import annotations

import re

_PY_LIKE = ("python", "py", "ruby", "shell", "bash", "r")

_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_SLASH = re.compile(r"(^|\s)//[^\n]*")
_LINE_HASH = re.compile(r"(^|\s)#[^\n]*")
_TRIPLE = re.compile(r"(\"\"\".*?\"\"\"|'''.*?''')", re.DOTALL)


def strip_comments(code: str, language: str | None = None) -> str:
    """Remove comments and docstrings. Language only picks which comment syntaxes to try."""
    s = str(code)
    lang = (language or "").strip().lower()
    s = _BLOCK.sub(" ", s)
    s = _TRIPLE.sub(" ", s)
    if lang.startswith(_PY_LIKE) or not lang:
        s = _LINE_HASH.sub(r"\1", s)
    if not lang.startswith(_PY_LIKE) or not lang:
        s = _LINE_SLASH.sub(r"\1", s)
    lines = [l.rstrip() for l in s.split("\n")]
    return "\n".join(l for l in lines if l.strip())


def normalize_whitespace(code: str, language: str | None = None) -> str:
    """Collapse indentation width, tabs, trailing spaces and blank lines.

    Python keeps one space per indent level so the code stays syntactically plausible; for brace
    languages indentation is dropped entirely.
    """
    s = str(code).replace("\t", "    ").replace("\r\n", "\n")
    lang = (language or "").strip().lower()
    out = []
    for line in s.split("\n"):
        body = line.strip()
        if not body:
            continue
        if lang.startswith(_PY_LIKE) or not lang:
            indent = (len(line) - len(line.lstrip(" "))) // 4
            out.append(" " * indent + re.sub(r"[ ]{2,}", " ", body))
        else:
            out.append(re.sub(r"[ ]{2,}", " ", body))
    return "\n".join(out)


def strip_and_normalize(code: str, language: str | None = None) -> str:
    return normalize_whitespace(strip_comments(code, language), language)


PERTURBATIONS = {
    "nocomments": strip_comments,
    "normws": normalize_whitespace,
    "stripped": strip_and_normalize,
}


def apply_perturbation(name: str, codes, languages=None) -> list[str]:
    fn = PERTURBATIONS[name]
    languages = list(languages) if languages is not None else [None] * len(codes)
    return [fn(c, l) for c, l in zip(codes, languages)]
