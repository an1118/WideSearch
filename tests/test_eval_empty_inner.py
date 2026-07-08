"""Regression test: evaluate_single_query must not crash when the response
table shares NO unique-key values with the answer (empty inner-join).

Bug: with an empty df_inner, ``df_inner.apply(func, axis=1)`` returns the empty
DataFrame (not a Series), so ``df_inner_score[col] = ...`` raised
``ValueError: Cannot set a DataFrame with multiple columns to the single
column``. Such a response should simply score 0 (no matched rows), not crash.
"""
import json

from src.evaluation.data_loader import WideSearchDataLoaderHF, WideSearchResponse
from src.evaluation.evaluation import evaluate_single_query


def _find_multi_key_instance():
    """Pick a real instance whose answer has many rows (so a 1-row wrong
    response won't overlap on the unique key)."""
    dl = WideSearchDataLoaderHF()
    for iid in dl.get_instance_id_list():
        q = dl.load_query_by_instance_id(iid)
        if q.answer is not None and len(q.answer) >= 5:
            return q
    return None


def test_empty_inner_join_scores_zero_not_crash():
    q = _find_multi_key_instance()
    assert q is not None, "need a multi-row-answer instance"
    required = q.evaluation["required"]
    # Build a response table with the right columns but a bogus key value that
    # cannot match any answer row -> empty inner join.
    header = "| " + " | ".join(required) + " |"
    sep = "| " + " | ".join(["---"] * len(required)) + " |"
    row = "| " + " | ".join(["___no_such_value___"] * len(required)) + " |"
    resp_text = "```markdown\n" + "\n".join([header, sep, row]) + "\n```"
    resp = WideSearchResponse(instance_id=q.instance_id, response=resp_text)

    # Must not raise; should score 0 across the board (no matched rows).
    result = evaluate_single_query(q, resp, None, "default_eval_config")
    assert result is not None
    assert result.f1_by_item == 0.0, f"expected 0 f1, got {result.f1_by_item}"
    assert result.precision_by_item == 0.0
    assert result.recall_by_item == 0.0
    assert "error" not in (result.msg or "").lower(), f"unexpected error: {result.msg}"
