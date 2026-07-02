# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os

# Compact backend base URL. Default points at the single-GPU port-forward
# (localhost:18080). For multi-GPU parallel runs each WideSearch worker sets
# COMPACT_BASE_URL to its own lane (e.g. http://localhost:18081 for GPU 1).
_COMPACT_BASE_URL = os.environ.get("COMPACT_BASE_URL", "http://localhost:18080")

# Qwen3.5-397B judge endpoint used by eval (default_eval_config). Set
# EVAL_BASE_URL to whichever local port-forward fronts the judge pod for
# kv_compact eval runs (e.g. http://localhost:39000/v1) — keeps us off the
# 38000 LB which other projects on this VM also depend on.
_EVAL_BASE_URL = os.environ.get("EVAL_BASE_URL", "http://localhost:38000/v1")

model_config = {
    "model_config_name": {
        "model_name": "MODEL_NAME",
        "base_url": "YOUR_BASE_URL",
        "api_key": "YOUR_API_KEY",
    },
    "k2": {
        "model_name": "kimi-k2-250711",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "max_tokens": 32768,
        },
    },
    "doubao-1.6": {
        "model_name": "doubao-seed-1-6-250615",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "thinking": {"type": "enabled"},
            "max_tokens": 65535,
        },
    },
    "deepseek-r1": {
        "model_name": "deepseek-r1",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "max_tokens": 65535,
        },
    },
    "doubao-1.6-non-thinking": {  # for eval
        "model_name": "doubao-seed-1-6-250615",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "thinking": {"type": "disabled"},
            "max_tokens": 65535,
        },
    },
    "claude37-sonnet-thinking": {
        "model_name": "gcp-claude37-sonnet",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "temperature": 1,
            "extra_body": {"thinking": {"type": "enabled", "budget_tokens": 4096}},
            "max_tokens": 10240,
        },
        "is_claude_thinking": True,
    },
    "claude4-sonnet-thinking": {
        "model_name": "claude4-sonnet",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "temperature": 1,
            "extra_body": {"thinking": {"type": "enabled", "budget_tokens": 32768}},
            "max_tokens": 64000,
        },
        "is_claude_thinking": True,
    },
    "o3-medium": {
        "model_name": "o3-2025-04-16",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "max_tokens": 65535,
            "reasoning_effort": "medium",
        },
        "default_system_prompt": "Formatting re-enabled",
    },
    "gemini-2.5-pro": {
        "model_name": "gemini-2.5-pro-preview-06-05",
        "base_url": "",
        "api_key": "",
        "generate_kwargs": {
            "max_tokens": 65535,
        },
    },
    "compact-qwen3_5-4b-future-proxy-dual": {
        "model_name": "compact-qwen3.5-4b",
        "base_url": _COMPACT_BASE_URL,
        "api_key": "unused",
        "is_claude_thinking": False,
        "default_system_prompt": "",
        "generate_kwargs": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 4096,
            "presence_penalty": 1.5,
        },
        "compact_config": {
            "enabled": True,
            "trigger": {"type": "every_turn"},
            "use_uncompacted_latest": False,
            "thinking_ratio": 0.05,                    # match BCP
            "tool_response_ratio": 0.05,               # match BCP
            "proxy_future_turns": 4,                   # match BCP
            "dual_mode": True,
            "num_thinking_query_tokens": 15000,        # match BCP
            "num_tool_response_query_tokens": 15000,   # match BCP
            "algorithm_kwargs": {},
        },
    },
    "compact-qwen3_5-4b-no-compaction": {
        "model_name": "compact-qwen3.5-4b",
        "base_url": _COMPACT_BASE_URL,
        "api_key": "unused",
        "is_claude_thinking": False,
        "default_system_prompt": "",
        "generate_kwargs": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 4096,
            "presence_penalty": 1.5,
        },
        "compact_config": {
            "enabled": False,
            "trigger": {"type": "no_compaction"},
            "use_uncompacted_latest": False,
            "thinking_ratio": 0.3,
            "tool_response_ratio": 0.5,
            "proxy_future_turns": 2,
            "dual_mode": False,
            "algorithm_kwargs": {},
        },
    },
    # BCP-parity Qwen3.5 compact: Yujian's confirmed config locked 2026-05-25.
    # Flags it mirrors:
    #   --query-method self_study --proxy-future-turns 2
    #   --include-boundary-proxy --thinking-ratio 0.2
    #   --tool-response-ratio 0.2 --preserve-boundaries --preserve-specials
    # (--dual-mode-beta-c2 NOT set → dual_mode=False;
    #  --optimize-beta / --optimize-values NOT set → TE flavor)
    "compact-qwen3_5-4b-bcp-parity": {
        "model_name": "compact-qwen3.5-4b",
        "base_url": _COMPACT_BASE_URL,
        "api_key": "unused",
        "is_claude_thinking": False,
        "default_system_prompt": "",
        "generate_kwargs": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 4096,
            "presence_penalty": 1.5,
        },
        "compact_config": {
            "enabled": True,
            "trigger": {"type": "every_turn"},
            "use_uncompacted_latest": False,
            "thinking_ratio": 0.2,
            "tool_response_ratio": 0.2,
            "proxy_future_turns": 2,
            "dual_mode": False,
            "query_method": "self_study",
            "preserve_boundaries": True,
            "preserve_specials": True,
            "include_boundary_proxy": True,
            "optimize_beta": False,
            "optimize_values": False,
            "num_thinking_query_tokens": None,
            "num_tool_response_query_tokens": None,
            "algorithm_kwargs": {},
        },
    },
    # EMNLP-paper "TE Boundary delay=0" config: per Table 1, this scored
    # 45.25% on Qwen3.5-4B BCP (-0.75pp vs no-compact 46.00) — the
    # strongest immediate-compaction config in the paper. Boundary Q's
    # (mr_suffix / tr_suffix) are used as the PRIMARY trajectory_proxies
    # rather than future asst_q (which doesn't exist at pft=0).
    #
    # Critical config differences from compact-qwen3_5-4b-bcp-parity:
    #   proxy_future_turns: 2 → 0        (immediate compaction)
    #   include_boundary_proxy: True → False  (boundary IS the primary,
    #                                          not an auxiliary entry)
    # Other compaction flags match Yujian's BCP-parity config.
    #
    # Requires server-side branch feat/qwen-compact-bcp-parity HEAD with
    # the pft=0 + self_study code path (mirrors BCP
    # run_browsecomp_evaluation.py:2076-2101).
    "compact-qwen3_5-4b-boundary-te-pft0": {
        "model_name": "compact-qwen3.5-4b",
        "base_url": _COMPACT_BASE_URL,
        "api_key": "unused",
        "is_claude_thinking": False,
        "default_system_prompt": "",
        "generate_kwargs": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 4096,
            "presence_penalty": 1.5,
        },
        "compact_config": {
            "enabled": True,
            "trigger": {"type": "every_turn"},
            "use_uncompacted_latest": False,
            "thinking_ratio": 0.2,
            "tool_response_ratio": 0.2,
            "proxy_future_turns": 0,
            "dual_mode": False,
            "query_method": "self_study",
            "preserve_boundaries": True,
            "preserve_specials": True,
            "include_boundary_proxy": False,
            "optimize_beta": False,
            "optimize_values": False,
            "num_thinking_query_tokens": None,
            "num_tool_response_query_tokens": None,
            "algorithm_kwargs": {},
        },
    },
    # Non-dynamic pft=5 one-shot BASELINE for the dynamic-compaction comparison.
    # Identical SS + TE setting (self_study, pft=5, ratio 0.2, preserve
    # boundaries+specials, same sampling) but dynamic_compaction=False, so each
    # turn is held full in the delayed buffer until a single one-shot compaction
    # on eviction. This is the apples-to-apples reference for
    # compact-qwen3_5-4b-te-ss-dynamic-pft5-r0.2: dynamic's frozen (steady-state)
    # cache is designed to match this one-shot result, so SCORES should track
    # closely while dynamic decodes faster (smaller in-flight window).
    #
    # BCP flags mirrored (no --dynamic-compaction):
    #   --query-method self_study --proxy-future-turns 5
    #   --thinking-ratio 0.2 --tool-response-ratio 0.2
    #   --preserve-boundaries --preserve-specials
    #   (TE flavor: neither --optimize-beta nor --optimize-values)
    "compact-qwen3_5-4b-te-ss-pft5-r0.2": {
        "model_name": "compact-qwen3.5-4b",
        "base_url": _COMPACT_BASE_URL,
        "api_key": "unused",
        "is_claude_thinking": False,
        "default_system_prompt": "",
        "generate_kwargs": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 4096,
            "presence_penalty": 1.5,
        },
        "compact_config": {
            "enabled": True,
            "trigger": {"type": "every_turn"},
            "use_uncompacted_latest": False,
            "thinking_ratio": 0.2,
            "tool_response_ratio": 0.2,
            "proxy_future_turns": 5,
            "dual_mode": False,
            "query_method": "self_study",
            "preserve_boundaries": True,
            "preserve_specials": True,
            "include_boundary_proxy": False,
            "dynamic_compaction": False,
            "optimize_beta": False,
            "optimize_values": False,
            "num_thinking_query_tokens": None,
            "num_tool_response_query_tokens": None,
            "algorithm_kwargs": {},
        },
    },
    # Dynamic (progressive) token-eviction: mirrors kv_compact's
    # scripts/run_te_dynamic.sh (qwen35-4b-te-ss-dynamic-pft5-r0.2). Same
    # SS + TE setting as the pft=5 delayed-buffer config but adds
    # dynamic_compaction: each turn is first-cut at its boundary then shrunk
    # one geometric step per future turn (cached-order proxy, force skeleton
    # kept every step, frozen body == ratio). In-flight window turns stay small
    # so decode is faster, while the steady-state (frozen) cache matches the
    # one-shot pft=5 result — so scores should track the non-dynamic pft=5 run.
    #
    # BCP flags mirrored:
    #   --query-method self_study --proxy-future-turns 5 --dynamic-compaction
    #   --thinking-ratio 0.2 --tool-response-ratio 0.2
    #   --preserve-boundaries --preserve-specials
    #   (TE flavor: neither --optimize-beta nor --optimize-values)
    #
    # Requires the server-side compact server (server/session.py) with the
    # dynamic-compaction port (CompactConfig.dynamic_compaction +
    # CompactSession._dynamic_step). dynamic_steps is derived server-side as
    # proxy_future_turns + 1 = 6.
    "compact-qwen3_5-4b-te-ss-dynamic-pft5-r0.2": {
        "model_name": "compact-qwen3.5-4b",
        "base_url": _COMPACT_BASE_URL,
        "api_key": "unused",
        "is_claude_thinking": False,
        "default_system_prompt": "",
        "generate_kwargs": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 4096,
            "presence_penalty": 1.5,
        },
        "compact_config": {
            "enabled": True,
            "trigger": {"type": "every_turn"},
            "use_uncompacted_latest": False,
            "thinking_ratio": 0.2,
            "tool_response_ratio": 0.2,
            "proxy_future_turns": 5,
            "dual_mode": False,
            "query_method": "self_study",
            "preserve_boundaries": True,
            "preserve_specials": True,
            "include_boundary_proxy": False,
            "dynamic_compaction": True,
            "optimize_beta": False,
            "optimize_values": False,
            "num_thinking_query_tokens": None,
            "num_tool_response_query_tokens": None,
            "algorithm_kwargs": {},
        },
    },

    # the concatenated proxy queries. RP+SS feeds the LSQ system with
    # both a teacher-forced replay forward (RP, run at push time per
    # BCP :1854-1875 to avoid pop-time cache pollution) and natural
    # future-turn asst_q (SS). Requires server-side push-time RP
    # precompute landed in session.py.
    "compact-qwen3_5-4b-am-rpss-pft1": {
        "model_name": "compact-qwen3.5-4b",
        "base_url": _COMPACT_BASE_URL,
        "api_key": "unused",
        "is_claude_thinking": False,
        "default_system_prompt": "",
        "generate_kwargs": {
            "temperature": 0.7, "top_p": 0.95,
            "max_new_tokens": 4096, "presence_penalty": 1.5,
        },
        "compact_config": {
            "enabled": True,
            "trigger": {"type": "every_turn"},
            "use_uncompacted_latest": False,
            "thinking_ratio": 0.2,
            "tool_response_ratio": 0.2,
            "proxy_future_turns": 1,
            "dual_mode": False,
            "query_method": "repeat_prefill+self_study",
            "preserve_boundaries": True,
            "preserve_specials": True,
            "include_boundary_proxy": False,
            "optimize_beta": True,
            "optimize_values": True,
            "num_thinking_query_tokens": None,
            "num_tool_response_query_tokens": None,
            "algorithm_kwargs": {},
        },
    },
    "compact-qwen3_5-4b-am-rpss-pft2": {
        "model_name": "compact-qwen3.5-4b",
        "base_url": _COMPACT_BASE_URL,
        "api_key": "unused",
        "is_claude_thinking": False,
        "default_system_prompt": "",
        "generate_kwargs": {
            "temperature": 0.7, "top_p": 0.95,
            "max_new_tokens": 4096, "presence_penalty": 1.5,
        },
        "compact_config": {
            "enabled": True,
            "trigger": {"type": "every_turn"},
            "use_uncompacted_latest": False,
            "thinking_ratio": 0.2,
            "tool_response_ratio": 0.2,
            "proxy_future_turns": 2,
            "dual_mode": False,
            "query_method": "repeat_prefill+self_study",
            "preserve_boundaries": True,
            "preserve_specials": True,
            "include_boundary_proxy": False,
            "optimize_beta": True,
            "optimize_values": True,
            "num_thinking_query_tokens": None,
            "num_tool_response_query_tokens": None,
            "algorithm_kwargs": {},
        },
    },
    # Phase 1 model-comparison entry: same pipeline as the Qwen3.5-4B
    # baseline, just pointing at the gemma-4-E4B-it snapshot. defaults
    # below come from the model's generation_config.json
    # (temperature=1.0, top_p=0.95, top_k=64). presence_penalty is
    # omitted — Gemma 4's generation_config doesn't set one, and BCP
    # eval doesn't pass it for Gemma 4 either. default_system_prompt
    # left empty per evaluation/utils.py:167-170 — Gemma family uses no
    # system prompt prefix.
    "gemma4-e4b-no-compaction": {
        "model_name": "compact-gemma4-e4b",
        "base_url": _COMPACT_BASE_URL,
        "api_key": "unused",
        "is_claude_thinking": False,
        "default_system_prompt": "",
        "generate_kwargs": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_new_tokens": 4096,
        },
        "compact_config": {
            "enabled": False,
            "trigger": {"type": "no_compaction"},
            "use_uncompacted_latest": False,
            "thinking_ratio": 0.3,
            "tool_response_ratio": 0.5,
            "proxy_future_turns": 2,
            "dual_mode": False,
            "algorithm_kwargs": {},
        },
    },
    "default_eval_config": {
        "model_name": "qwen-judge",
        "base_url": _EVAL_BASE_URL,
        "api_key": "dummy",
        "served_model_name": "qwen",
        "generate_kwargs": {
            "max_tokens": 4096,
            "temperature": 0,
        },
        "temperature": 0,
    },
}
