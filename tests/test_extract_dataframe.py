"""Tests for WideSearchResponse.extract_dataframe block selection.

Regression: the model sometimes echoes the task's markdown format template
(e.g. ```markdown{数据内容}```) as the FIRST ```markdown block, then emits the
real answer table in a LATER block. extract_dataframe used to always take
block[0], so it parsed the empty placeholder -> crash/None -> f1=0, even though
a real table was present. It must instead pick the first NON-EMPTY table block,
and must not crash on an unparseable block.
"""
import pandas as pd

from src.evaluation.data_loader import WideSearchResponse


def _resp(text):
    return WideSearchResponse(instance_id="t", response=text)


def test_placeholder_first_then_real_table_picks_real():
    # block[0] is the echoed format placeholder; block[1] is the real table.
    text = (
        "输出如下：\n"
        "```markdown{数据内容}```\n"
        "```markdown\n"
        "| 年份 | 姓名 |\n"
        "| :--- | :--- |\n"
        "| 2020 | 拜登 |\n"
        "| 2024 | 特朗普 |\n"
        "```\n"
    )
    df = _resp(text).extract_dataframe()
    assert df is not None, "should recover the real table from block[1]"
    assert len(df) == 2, f"expected 2 data rows, got {len(df) if df is not None else None}"
    assert list(df.columns) == ["年份", "姓名"]


def test_single_real_table_unchanged():
    text = (
        "```markdown\n"
        "| a | b |\n"
        "| :--- | :--- |\n"
        "| 1 | 2 |\n"
        "```\n"
    )
    df = _resp(text).extract_dataframe()
    assert df is not None
    assert len(df) == 1
    assert list(df.columns) == ["a", "b"]
    assert str(df.iloc[0]["a"]) == "1"


def test_all_placeholder_blocks_returns_none_no_crash():
    text = "```markdown{数据内容}```\n```markdown{data}```"
    # must not raise; no real table anywhere -> None
    df = _resp(text).extract_dataframe()
    assert df is None


def test_raw_pipe_table_without_fence_still_works():
    # no ```markdown fence at all -> pipe-scan fallback path (unchanged)
    text = "Here it is:\n| x | y |\n| :--- | :--- |\n| 7 | 8 |\n"
    df = _resp(text).extract_dataframe()
    assert df is not None
    assert len(df) == 1
    assert list(df.columns) == ["x", "y"]


def test_unparseable_first_block_skipped():
    # A block that raises inside pd.read_csv (ragged) must be skipped, then the
    # good block used — not crash the whole eval.
    text = (
        "```markdown\n{只是一个占位符，没有表格}\n```\n"
        "```markdown\n| c | d |\n| :--- | :--- |\n| 9 | 10 |\n```\n"
    )
    df = _resp(text).extract_dataframe()
    assert df is not None
    assert list(df.columns) == ["c", "d"]
    assert len(df) == 1
