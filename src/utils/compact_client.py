# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""VM-side HTTP client for the kv_compact compact backend on the GPU pod.

The pod runs an HTTP server (see ``kv_compact/server/compact_server.py``). This
module provides:

  * ``session_context(instance_id, trial_idx)`` — a context manager that
    generates a deterministic-but-unique ``session_id`` and stashes it in a
    ``contextvars.ContextVar`` for the lifetime of one WideSearch query
    trajectory. ``llm_completion`` (added in Task 11) picks the id up from the
    contextvar so WideSearch's ``Runner`` plumbing never has to thread it
    explicitly.
  * ``complete(...)`` — wraps ``POST /llm_completion`` and returns an
    ``openai.types.chat.ChatCompletionMessage`` so callers expecting OpenAI's
    return type don't break.
  * ``delete_session(...)`` — best-effort cleanup of pod-side session state.

Targets kv_compact v1.5 schema (``future_proxy + dual_mode``); see the
``CompactConfig`` shape in ``kv_compact/server/protocol.py``.

Operational notes (from pod-side Task 9 smoke test):

  1. **Tool messages MUST NOT include ``tool_call_id``.** The pod's Qwen3.5
     chat-template handling expects BCP-shape tool messages
     (``{"role": "tool", "content": "..."}``) without ``tool_call_id``. WideSearch's
     existing rollout produces OpenAI-style tool messages WITH ``tool_call_id``;
     ``complete()`` strips that key before sending.

  2. **Message list is append-only after the first call.** The pod's
     ``_compute_incremental_text`` diffs the new messages against the
     conversation snapshot from the previous call and requires the new list to
     start with the old list. WideSearch sends one initial user query and then
     only appends ``assistant`` + ``tool`` turns, so this is fine in
     production. If a test harness ever splices in a new user message
     mid-rollout, the server will raise a "Conversation diverged" RuntimeError.

  3. **Don't call ``complete()`` again after a final-answer response.** The
     server has a defensive guard (commit ``15d63d2``, 2026-05-18) for the
     empty-incremental-text case, but the round-trip is wasted work.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import requests
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

logger = logging.getLogger(__name__)

# Module-level ContextVar carries the active pod session id for the duration of
# one WideSearch query. Reads default to None when no session is active.
_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "compact_session_id", default=None,
)


def get_session_id() -> Optional[str]:
    """Return the active session id, or None if no `session_context` is open."""
    return _session_id.get()


@contextmanager
def session_context(instance_id: str, trial_idx: int) -> Iterator[str]:
    """Bind a freshly generated session_id to the current context.

    The id has the form ``f"{instance_id}_trial{trial_idx}_{uuid.uuid4().hex[:8]}"``,
    which is both human-readable in pod logs and unique per call.

    The context manager only manages the contextvar lifetime — it does NOT
    automatically call the pod's ``DELETE /session/{id}`` endpoint, because
    cleanup needs the ``base_url`` and we'd rather keep the manager's signature
    free of backend-specific args. Callers should invoke ``delete_session(id,
    base_url)`` from their own ``finally:`` block (or rely on the pod's LRU
    eviction in cases where best-effort cleanup is acceptable).

    Usage::

        with session_context(instance_id, trial_idx) as sid:
            try:
                msg = complete(messages=..., tools=..., base_url=URL, ...)
                ...
            finally:
                delete_session(sid, URL)
    """
    sid = f"{instance_id}_trial{trial_idx}_{uuid.uuid4().hex[:8]}"
    token = _session_id.set(sid)
    try:
        yield sid
    finally:
        _session_id.reset(token)


def _strip_tool_call_id(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop ``tool_call_id`` from every message.

    See module docstring note 1 — Qwen3.5's parser produces ``tool_calls``
    without an ``id`` and the chat template doesn't handle the OpenAI
    round-trip shape, so we strip the field for ALL roles (assistant tool-call
    messages don't carry ``tool_call_id`` themselves, but tool-response
    messages do).
    """
    return [
        {k: v for k, v in msg.items() if k != "tool_call_id"}
        for msg in messages
    ]


def complete(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    base_url: str,
    compact_config: Dict[str, Any],
    sampling: Dict[str, Any],
    timeout: float = 300.0,
) -> ChatCompletionMessage:
    """POST to ``/llm_completion`` and return an OpenAI-shape message.

    Args:
        messages: OpenAI-shape chat messages. Tool responses may carry
            ``tool_call_id``; the client strips it before sending.
        tools: OpenAI tool definitions (passed through unchanged).
        base_url: Pod base URL, e.g. ``"http://localhost:18080"``.
        compact_config: Dict matching ``server.protocol.CompactConfig`` (the v1.5
            schema — ``enabled``, ``trigger``, ``use_uncompacted_latest``,
            ``thinking_ratio``, ``tool_response_ratio``, ``proxy_future_turns``,
            ``dual_mode``, ``algorithm_kwargs``).
        sampling: Dict matching ``server.protocol.Sampling`` (``temperature``,
            ``top_p``, ``top_k``, ``max_new_tokens``, optional
            ``presence_penalty``).
        timeout: Per-request HTTP timeout (seconds). The pod's decode pass can
            take a while for long outputs, hence the generous default.

    Returns:
        ``ChatCompletionMessage`` with ``role="assistant"``, populated
        ``content`` (may be empty string when the model only emits tool calls),
        and ``tool_calls`` either ``None`` (for final answers) or a list of
        ``ChatCompletionMessageToolCall``. Because the pod doesn't supply a
        per-call ``id`` for tool calls (Qwen3.5's BCP parser shape), the client
        fabricates ``id=f"call_{i}"`` so downstream OpenAI-style consumers
        always have one.

    Raises:
        RuntimeError: when called outside a ``session_context``.
        requests.HTTPError: on non-2xx responses (raised by ``raise_for_status``).
    """
    sid = _session_id.get()
    if sid is None:
        raise RuntimeError(
            "compact_client.complete() called outside session_context. "
            "Wrap your WideSearch rollout in `with session_context(instance_id, "
            "trial_idx):` so the client knows which pod session to target."
        )

    sanitized_messages = _strip_tool_call_id(messages)
    payload = {
        "session_id": sid,
        "messages": sanitized_messages,
        "tools": tools,
        "compact_config": compact_config,
        "sampling": sampling,
    }
    url = f"{base_url.rstrip('/')}/llm_completion"
    logger.debug(
        "POST %s session=%s msgs=%d tools=%d",
        url, sid, len(sanitized_messages), len(tools),
    )
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()

    msg = body["message"]
    raw_tool_calls = msg.get("tool_calls") or []
    tool_calls_typed: List[ChatCompletionMessageToolCall] = []
    for i, tc in enumerate(raw_tool_calls):
        # Pod may or may not supply an id; fabricate one for OpenAI consumers.
        fn = tc.get("function") or {}
        tool_calls_typed.append(
            ChatCompletionMessageToolCall(
                id=tc.get("id") or f"call_{i}",
                type=tc.get("type", "function"),
                function=Function(
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", "") or "",
                ),
            )
        )

    return ChatCompletionMessage(
        role="assistant",
        content=msg.get("content") or "",
        tool_calls=tool_calls_typed or None,
    )


def delete_session(
    session_id: str,
    base_url: str,
    timeout: float = 10.0,
) -> None:
    """Best-effort ``DELETE /session/{session_id}``.

    Logs and swallows all errors — a 404 means the session is already gone,
    and a network blip during cleanup shouldn't tear down WideSearch's outer
    loop. (Stale sessions on the pod are bounded by its LRU eviction.)
    """
    url = f"{base_url.rstrip('/')}/session/{session_id}"
    try:
        response = requests.delete(url, timeout=timeout)
    except requests.RequestException as exc:
        logger.warning("DELETE %s failed: %s", url, exc)
        return
    if response.status_code == 404:
        logger.debug("DELETE %s: session already gone (404)", url)
        return
    if response.status_code >= 400:
        logger.warning(
            "DELETE %s returned %d: %s",
            url, response.status_code, response.text[:200],
        )
