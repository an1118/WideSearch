# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""Tests for src/utils/compact_client.py — VM-side HTTP client for kv_compact."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.utils.compact_client import (
    complete,
    delete_session,
    get_session_id,
    session_context,
)


# -------- session_context / get_session_id --------------------------------- #


def test_session_context_sets_and_clears():
    assert get_session_id() is None
    with session_context("instance1", trial_idx=0) as sid:
        assert get_session_id() == sid
        assert sid is not None
    assert get_session_id() is None


def test_session_id_format():
    """ID should contain the instance_id and trial idx so pod logs are readable."""
    with session_context("instance_xyz", trial_idx=7) as sid:
        assert sid.startswith("instance_xyz_trial7_")
        # 8-char uuid suffix
        suffix = sid.rsplit("_", 1)[-1]
        assert len(suffix) == 8


def test_session_context_nesting_isolated():
    """Nested context managers should restore the outer id on exit."""
    with session_context("outer", trial_idx=0) as outer_sid:
        with session_context("inner", trial_idx=1) as inner_sid:
            assert inner_sid != outer_sid
            assert get_session_id() == inner_sid
        assert get_session_id() == outer_sid
    assert get_session_id() is None


# -------- complete() ------------------------------------------------------- #


def _fake_ok_response(tool_calls=None, content="ok"):
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "session_id": "anything",
        "message": {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        },
        "finish_reason": "tool_calls" if tool_calls else "stop",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "compaction_info": None,
        "error": None,
    }
    fake.raise_for_status = MagicMock()
    return fake


def test_complete_strips_tool_call_id():
    """Pod expects BCP-shape tool messages WITHOUT tool_call_id."""
    fake = _fake_ok_response(content="reply")
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_a", "type": "function",
                 "function": {"name": "search", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_a",
         "content": "results: ..."},
    ]
    with patch("src.utils.compact_client.requests.post",
               return_value=fake) as post:
        with session_context("instX", trial_idx=0):
            complete(
                messages=messages,
                tools=[],
                base_url="http://localhost:8080",
                compact_config={
                    "enabled": True,
                    "trigger": {"type": "every_turn"},
                    "use_uncompacted_latest": False,
                    "thinking_ratio": 0.3,
                    "tool_response_ratio": 0.5,
                    "proxy_future_turns": 2,
                    "dual_mode": True,
                    "algorithm_kwargs": {},
                },
                sampling={
                    "temperature": 0.7, "top_p": 0.95, "top_k": None,
                    "max_new_tokens": 64, "presence_penalty": None,
                },
            )
    # Inspect the body that was sent.
    call = post.call_args
    assert call.args[0].endswith("/llm_completion")
    body = call.kwargs["json"]
    for msg in body["messages"]:
        assert "tool_call_id" not in msg, (
            f"tool_call_id was NOT stripped from {msg}"
        )
    # And nothing else changed
    assert body["messages"][2]["role"] == "tool"
    assert body["messages"][2]["content"] == "results: ..."
    # session_id was inserted from the contextvar
    assert body["session_id"].startswith("instX_trial0_")


def test_complete_returns_openai_shape():
    """Tool-calls in the JSON response surface as ChatCompletionMessage tool_calls."""
    fake = _fake_ok_response(
        tool_calls=[
            {"type": "function",
             "function": {"name": "search", "arguments": '{"q":"x"}'}},
            {"type": "function",
             "function": {"name": "fetch", "arguments": '{}'}}
        ],
        content=None,
    )
    with patch("src.utils.compact_client.requests.post", return_value=fake):
        with session_context("inst1", trial_idx=0):
            msg = complete(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                base_url="http://pod:8080",
                compact_config={
                    "enabled": True,
                    "trigger": {"type": "every_turn"},
                    "use_uncompacted_latest": False,
                    "thinking_ratio": 0.3,
                    "tool_response_ratio": 0.5,
                    "proxy_future_turns": 2,
                    "dual_mode": True,
                    "algorithm_kwargs": {},
                },
                sampling={
                    "temperature": 0.7, "top_p": 0.95, "top_k": None,
                    "max_new_tokens": 64, "presence_penalty": None,
                },
            )
    assert msg.role == "assistant"
    # content was None in the response → client coerces to empty string
    assert msg.content == ""
    assert msg.tool_calls is not None and len(msg.tool_calls) == 2
    # Pod doesn't supply id; client fabricates one.
    assert msg.tool_calls[0].id == "call_0"
    assert msg.tool_calls[1].id == "call_1"
    assert msg.tool_calls[0].function.name == "search"
    assert msg.tool_calls[0].function.arguments == '{"q":"x"}'
    assert msg.tool_calls[1].function.name == "fetch"


def test_complete_returns_openai_shape_no_tool_calls():
    """Final-answer responses (no tool_calls) → tool_calls=None on the message."""
    fake = _fake_ok_response(tool_calls=None, content="final answer")
    with patch("src.utils.compact_client.requests.post", return_value=fake):
        with session_context("inst1", trial_idx=0):
            msg = complete(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                base_url="http://pod:8080",
                compact_config={
                    "enabled": True,
                    "trigger": {"type": "every_turn"},
                    "use_uncompacted_latest": False,
                    "thinking_ratio": 0.3,
                    "tool_response_ratio": 0.5,
                    "proxy_future_turns": 2,
                    "dual_mode": True,
                    "algorithm_kwargs": {},
                },
                sampling={
                    "temperature": 0.7, "top_p": 0.95, "top_k": None,
                    "max_new_tokens": 64, "presence_penalty": None,
                },
            )
    assert msg.content == "final answer"
    assert msg.tool_calls is None


def test_complete_raises_if_no_session_context():
    """Calling complete() outside session_context should raise a clear error."""
    assert get_session_id() is None
    with pytest.raises(RuntimeError, match="session_context"):
        complete(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            base_url="http://pod:8080",
            compact_config={
                "enabled": True,
                "trigger": {"type": "every_turn"},
                "use_uncompacted_latest": False,
                "thinking_ratio": 0.3,
                "tool_response_ratio": 0.5,
                "proxy_future_turns": 2,
                "dual_mode": True,
                "algorithm_kwargs": {},
            },
            sampling={
                "temperature": 0.7, "top_p": 0.95, "top_k": None,
                "max_new_tokens": 64, "presence_penalty": None,
            },
        )


# -------- delete_session() ------------------------------------------------ #


def test_delete_session_ignores_404():
    """404 from the pod (session already deleted) should NOT raise."""
    fake = MagicMock(status_code=404, text="not found")
    with patch("src.utils.compact_client.requests.delete",
               return_value=fake) as deleter:
        delete_session("sess_x", base_url="http://pod:8080")
    call = deleter.call_args
    assert call.args[0].endswith("/session/sess_x")


def test_delete_session_swallows_network_error():
    """Network blip during cleanup must not bubble up."""
    with patch("src.utils.compact_client.requests.delete",
               side_effect=requests.ConnectionError("boom")):
        # Should not raise.
        delete_session("sess_y", base_url="http://pod:8080")


def test_delete_session_calls_endpoint_on_success():
    fake = MagicMock(status_code=200)
    with patch("src.utils.compact_client.requests.delete",
               return_value=fake) as deleter:
        delete_session("sess_z", base_url="http://pod:8080/")
    call = deleter.call_args
    assert call.args[0].endswith("/session/sess_z")
    # rstrip strips the trailing slash from base_url
    assert "//session" not in call.args[0]


# -------- llm.py dispatch ------------------------------------------------- #


def test_llm_completion_dispatches_to_compact_backend(monkeypatch):
    """llm_completion routes 'compact-*' model_name to compact_client.complete."""
    from src.utils.llm import llm_completion

    fake_response = MagicMock(status_code=200)
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "session_id": "x_trial0_abc",
        "message": {"role": "assistant", "content": "result", "tool_calls": None},
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    with patch("src.utils.compact_client.requests.post", return_value=fake_response):
        with session_context("test_inst", 0):
            result = llm_completion(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                model_config_name="compact-qwen3_5-4b-no-compaction",
            )
    # The exact return type depends on llm_completion's contract; just check
    # that we got a non-None response and that compact_client was called.
    assert result is not None
    assert result.role == "assistant"
    assert result.content == "result"
