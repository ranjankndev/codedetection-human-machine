import numpy as np
from codedet.labels import normalize_family, mark_seen, binary_labels
from codedet.data import (stratified_subsample, cap_human_share, dedup_exact, holdout_language_split,
                          holdout_generator_split, head_tail_ids)
from codedet.fixtures import make_fixture


def test_normalize_family():
    cases = {"Human": "human", "meta-llama/CodeLlama-7b": "meta-llama", "gpt-4o-mini": "openai",
             "microsoft/phi-2": "phi", "google/gemma-7b": "gemma", "Qwen/Qwen2.5-Coder": "qwen",
             "01-ai/Yi-Coder": "01-ai", "mistralai/Mistral-7B": "mistral", "deepseek-ai/x": "deepseek"}
    for k, v in cases.items():
        assert normalize_family(k) == v, k


def test_binary_and_seen():
    df = make_fixture(60)
    assert set(binary_labels(df)) == {0, 1}
    d2 = mark_seen(df, ["Qwen/Qwen2.5-Coder-7B"])
    assert (d2.loc[d2.generator == "Human", "seen"] == "seen").all()
    assert (d2.loc[d2.generator == "gpt-4o-mini", "seen"] == "unseen").all()


def test_subsample_cap_dedup_split():
    df = make_fixture(300, seed=1)
    sub = stratified_subsample(df, 90, ["label", "language"])
    assert 80 <= len(sub) <= 100 and sub.groupby(["label", "language"]).ngroups == df.groupby(["label", "language"]).ngroups
    capped = cap_human_share(df, 0.2)
    assert (capped.generator == "Human").mean() <= 0.21
    train, ood = holdout_language_split(df, "java")
    assert (ood.language == "Java").all() and "Java" not in set(train.language)
    tr2, unseen = holdout_generator_split(df, ["gpt-4o-mini"], human_frac=0.1)
    assert "gpt-4o-mini" not in set(tr2.generator) and "gpt-4o-mini" in set(unseen.generator)
    clean, removed = dedup_exact(df.iloc[:100], [df.iloc[50:150]])
    assert removed >= 50 and not set(clean.code) & set(df.iloc[50:150].code)


def test_head_tail(tok):
    ids = head_tail_ids(tok, "x" * 1000, max_len=64)
    assert len(ids) == 64 and ids[0] == 0 and ids[-1] == 2
    assert len(head_tail_ids(tok, "abc", max_len=64)) == 5
