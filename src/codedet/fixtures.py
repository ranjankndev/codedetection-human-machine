"""Synthetic parquet-like data so the whole pipeline can be tested on a laptop CPU with no downloads.

Machine-like snippets: docstrings, comments, descriptive names, 4-space indent.
Human-like: terse, no comments, cramped. (Separable on purpose - the dry run checks plumbing, not science.)

Every snippet is randomised (identifiers, operation, helper tail) so the dedup / near-duplicate stage has
something realistic to do: a fixture made of identical templates would be deleted by the MinHash check.
Two generators per family give task B a genuine held-out-generator split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MODELS = [
    "Qwen/Qwen2.5-Coder-7B", "Qwen/Qwen2.5-Coder-1.5B",
    "deepseek-ai/deepseek-coder-6.7b", "deepseek-ai/deepseek-coder-1.3b",
    "meta-llama/CodeLlama-7b", "meta-llama/CodeLlama-13b",
    "bigcode/starcoder2-7b", "bigcode/starcoder2-3b",
    "mistralai/Mistral-7B", "mistralai/Codestral-22B",
    "gpt-4o-mini", "gpt-4.1-mini",
]

_OPS = [("sum", "+=", "total"), ("product", "*=", "product"), ("maximum", "max", "maximum")]
_NOUNS = ["numbers", "values", "items", "input_list", "elements", "entries"]
_SHORT = ["a", "v", "xs", "arr", "nums", "q"]


def _tail_comment(r, lang: str) -> str:
    """A unique trailing token per row: keeps rows from being exact/near duplicates of each other."""
    tag = f"{r.integers(0, 10 ** 9)}-{r.integers(0, 10 ** 9)}"
    return f"# case {tag}\n" if lang == "Python" else f"// case {tag}\n"


def _py_machine(r):
    noun = _NOUNS[int(r.integers(len(_NOUNS)))]
    name, op, acc = _OPS[int(r.integers(len(_OPS)))]
    body = (f"    {acc} = {noun}[0]\n    for item in {noun}[1:]:\n        {acc} = max({acc}, item)\n"
            if op == "max" else
            f"    {acc} = {0 if op == '+=' else 1}\n    for item in {noun}:\n        {acc} {op} item\n")
    extra = "\n    if not numbers:\n        raise ValueError('empty input')\n" if r.random() < 0.4 else ""
    return (f'def compute_{name}({noun}):\n    """Return the {name} of every item in {noun}."""\n'
            f"    # accumulate the result in a single pass\n{body}{extra}    return {acc}\n")


def _py_human(r):
    v = _SHORT[int(r.integers(len(_SHORT)))]
    name, op, _ = _OPS[int(r.integers(len(_OPS)))]
    if op == "max":
        return f"def f({v}):\n  m={v}[0]\n  for x in {v}[1:]:\n   if x>m: m=x\n  return m\n"
    return f"def f({v}):\n  s={0 if op == '+=' else 1}\n  for x in {v}: s{op}x\n  return s\n"


def _java_machine(r):
    name = ["findMaximum", "computeSum", "countMatches"][int(r.integers(3))]
    return ("public class Solution {\n    /**\n     * " + f"{name} over the given array.\n     */\n"
            f"    public int {name}(int[] numbers) {{\n        int result = numbers[0];\n"
            "        for (int number : numbers) {\n            if (number > result) {\n"
            "                result = number;\n            }\n        }\n        return result;\n    }\n}\n")


def _java_human(r):
    v = _SHORT[int(r.integers(len(_SHORT)))]
    return f"class S{{int m(int[] {v}){{int b={v}[0];for(int x:{v})if(x>b)b=x;return b;}}}}\n"


def _cpp_machine(r):
    name = ["countEvenNumbers", "accumulateValues", "findLargest"][int(r.integers(3))]
    return ("#include <vector>\n\n// " + f"{name}: iterate once over the input vector.\n"
            f"int {name}(const std::vector<int>& values) {{\n    int result = 0;\n"
            "    for (int value : values) {\n        if (value % 2 == 0) {\n            result++;\n"
            "        }\n    }\n    return result;\n}\n")


def _cpp_human(r):
    v = _SHORT[int(r.integers(len(_SHORT)))]
    return f"#include<bits/stdc++.h>\nint c(std::vector<int>&{v}){{int k=0;for(int x:{v})k+=!(x&1);return k;}}\n"


_GEN = {"Python": (_py_human, _py_machine), "Java": (_java_human, _java_machine), "C++": (_cpp_human, _cpp_machine)}


def make_fixture(n: int = 240, seed: int = 0, human_share: float = 0.5,
                 languages=("Python", "Java", "C++")) -> pd.DataFrame:
    """Columns mimic the official parquet: code, label (0 human / 1 machine), language, generator, domain."""
    r = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        lang = languages[i % len(languages)]
        human = r.random() < human_share
        h, m = _GEN[lang]
        code = (h if human else m)(r) + _tail_comment(r, lang)
        rows.append({"code": code, "label": 0 if human else 1, "language": lang,
                     "generator": "Human" if human else _MODELS[int(r.integers(len(_MODELS)))],
                     "domain": "algorithmic"})
    return pd.DataFrame(rows)
