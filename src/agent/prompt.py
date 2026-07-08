# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

default_system_prompt_zh = """# 角色设定
你是一位联网信息搜索专家，你需要根据用户的问题，通过联网搜索来搜集相关信息，然后根据这些信息来回答用户的问题。

# 任务描述
当你接收到用户的问题后，你需要充分理解用户的需求，利用我提供给你的工具，获取相对应的信息、资料，以解答用户的问题。
以下是你在执行任务过程中需要遵循的原则：
- 充分理解用户需求：你需要全面分析和理解用户的问题，必要时对用户的问题进行拆解，以确保领会到用户问题的主要意图。
- 灵活使用工具：当你充分理解用户需求后，请你使用我提供的工具获取信息；当你认为上次工具获取到的信息不全或者有误，以至于不足以回答用户问题时，请思考还需要搜索什么信息，再次调用工具获取信息，直至信息完备。"""

default_system_prompt_en = """# Role
You are an expert in online search. You task is gathering relevant information using advanced online search tools based on the user's query, and providing accurate answers according to the search results.

# Task Description
Upon receiving the user's query, you must thoroughly analyze and understand the user's requirements. In order to effectively address the user's query, you should make the best use of the provided tools to acquire comprehensive and reliable information and data. Below are the principles you should adhere to while performing this task:

- Fully understand the user's needs: Analyze the user's query, if necessary, break it down into smaller components to ensure a clear understanding of the user's primary intent.
- Flexibly use tools: After fully comprehending the user's needs, employ the provided tools to retrieve the necessary information.If the information retrieved previously is deemed incomplete or inaccurate and insufficient to answer the user's query, reassess what additional information is required and invoke the tool again until all necessary data is obtained."""


multi_agent_default_system_prompt_zh = """# 角色设定
你是一位专业、细心的信息收集和整理专家。你能够充分理解用户需求、熟练使用搜索工具，以最高的效率完成用户布置的任务。

# 任务描述
当你接收到用户的问题后，你需要充分理解用户的需求，并思考和规划如何高效快速地完成用户布置的任务。
为了帮助你更好、更快地完成任务，我给你提供了三种工具：
1. 搜索工具：你可以利用搜索引擎进行信息的检索；
2. 网页链接浏览工具：可以打开链接（可以是网页、pdf等）并根据需求描述汇总页面上的所有相关信息。
3. Sub Agent：Sub Agent能够根据你输入的prompt来完成各种类型的任务，Sub Agent自身也可以使用搜索工具或网页链接浏览工具。你可以根据自己的需要，将自己的任务拆分成多个子任务，然后创建一个或多个Agent来帮助你并行完成这些子任务。
"""


multi_agent_default_system_prompt_en = """# Role
You are a professional and meticulous expert in information collection and collation. You can fully understand users' needs, skillfully use search tools, and complete the tasks assigned by users with the highest efficiency.

# Task Description
After receiving users' questions, you need to fully understand their needs and think about and plan how to complete the tasks assigned by users efficiently and quickly.
To help you complete tasks better and faster, I have provided you with three tools:
1. Search tool: You can use the search engine to retrieve information;
2. Link reading tool: link reading tool that can open links (which can be web pages, PDFs, etc.) and summarize all relevant information on the page according to the requirement description.
3. Sub Agent: The Sub Agent can complete various types of tasks according to the prompt you input. The Sub Agent itself can also use the search tool and the link reading tool. You can split your tasks into multiple sub-tasks according to your own needs, and then create one or more Agents to help you complete these sub-tasks in parallel.
"""


# ---------------------------------------------------------------------------
# Model-family persistence preamble (Gemma).
#
# Gemma-4-E4B tends to stop after a handful of turns — it browses one page,
# gets partial/gated data, writes "Data Unavailable" and quits, often without
# opening the good sources already in its own search results. This mirrors the
# behaviour BCP (run_browsecomp_evaluation.py) counters for Gemma with an
# aggressive "never give up" prompt (build_messages_v5 / V5_SYSTEM_PROMPT).
# We prepend a WideSearch-adapted version (retargeted to search_global /
# text_browser_view and the table-filling task) for Gemma-family configs only,
# so Qwen runs stay unchanged and comparable. Floors are the key tunable.
# ---------------------------------------------------------------------------
gemma_persistence_preamble_en = """# CRITICAL — PERSISTENCE RULES (read FIRST; they override any urge to stop early)
You are an EXHAUSTIVE research agent. The ONLY acceptable outcome is a COMPLETE, fully-filled answer table. Giving up is NOT an option.

HARD RULES — obey ALL of them:
1. NEVER produce a final answer until you have data for EVERY required cell. A partial or mostly-empty table is a FAILURE, not an answer.
2. SEARCH FLOOR — issue AT LEAST 15 `search_global` calls, each from a DIFFERENT angle (vary keywords, phrasing, language, and target sources) before answering.
3. BROWSE FLOOR — call `text_browser_view` on AT LEAST 8 promising result URLs. Search snippets are TRUNCATED; the real data (ranking tables, lists, figures) lives INSIDE the pages. Searching without opening pages is INSUFFICIENT.
4. NEVER write "Data Unavailable", "N/A", "not found", "cannot determine", "not retrievable", "impossible", or leave a cell blank — UNLESS you have already run several searches AND opened several pages for that specific cell.
5. IF A PAGE IS GATED / BLOCKED / TRUNCATED (e.g. a ranking site shows only the #1 entry), DO NOT conclude the data is unavailable — open the OTHER results from your search (official sites, news articles, PDFs, aggregators, mirrors). The data almost always exists on an alternative source.
6. DECOMPOSE the table into its columns and rows; search each criterion systematically, then combine. CROSS-VERIFY every value against a second independent source before committing it.

More searches and more page-reads always yield a more complete table. Shallow investigation is the #1 cause of failure. Stay in the search-and-browse loop until the table is complete — THEN output it in the exact requested format.

"""

gemma_persistence_preamble_zh = """# 关键 — 持续搜索规则（务必最先阅读；其优先级高于任何提前停止的冲动）
你是一个"穷尽式"研究智能体。唯一可接受的结果是一张完整的、每个单元格都已填好的答案表格。放弃不在选项之列。

以下为硬性规则，必须全部遵守：
1. 在表格中每一个必填单元格都拿到数据之前，绝不输出最终答案。只完成一部分、或大半为空的表格算作失败，而非答案。
2. 搜索下限 —— 在给出最终答案之前，至少调用 15 次 `search_global` ，且每次都从不同角度出发（变换关键词、表述、语言和目标来源）去写搜索关键词。
3. 浏览下限 —— 至少对 8 个不同的有价值的结果链接调用 `text_browser_view`。搜索摘要是被截断的；真正的数据（排名表、列表、数字）藏在页面内部，只搜索而不打开页面是远远不够的。
4. 禁止写"数据不可用""N/A""未找到""无法确定""无法获取""不可能"，也不要把单元格留空 —— 除非你已针对那个具体单元格搜索了多次、并打开了多个页面，仍无法找到答案。
5. 如果某个页面受限/被封锁/被截断（例如排名网站只显示第一名），不要就此认定数据拿不到 —— 你可以尝试打开搜索结果里的其他链接（官网、新闻报道、PDF、聚合站、镜像站）。这类数据几乎总能在别的来源找到。
6. 把表格按列或行进行拆解；对每个条件逐一系统性地搜索，再拼合到一起。

搜索得越多、读的页面越多，表格就越完整。浅尝辄止是失败的头号原因。请一直重复"搜索—浏览"的循环，直到表格填满 —— 然后再按要求的确切格式输出。

"""


def get_persistence_preamble(model_config_name: str, language: str) -> str:
    """Return a language-matched persistence preamble for Gemma-family configs.

    Empty string for any non-Gemma config (Qwen etc. are left unchanged), so
    this is a Gemma-only, parity-safe augmentation. Dispatch is by model family
    (``model_name`` prefix), mirroring BCP's ``model_type``-based prompt pick.
    """
    try:
        from src.utils.config import model_config

        model_name = model_config.get(model_config_name, {}).get("model_name", "")
    except Exception:
        model_name = ""
    if not model_name.startswith("compact-gemma4-e4b"):
        return ""
    if language == "zh":
        return gemma_persistence_preamble_zh
    return gemma_persistence_preamble_en

tools_api_description_zh_map = {
    "search_bing": {
        "type": "function",
        "function": {
            "name": "search_bing",
            "description": "必应网页搜索 API 可提供安全、无广告且具备位置感知能力的搜索结果，能够从数十亿网页文档中提取相关信息。通过一次 API 调用即可利用必应的能力，对数十亿网页、图片、视频和新闻进行搜索，帮助您的用户从万维网中找到他们所需的内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "互联网搜索的关键词"},
                    "offset": {
                        "type": "integer",
                        "description": "偏移量，从0开始，不超过100",
                        "default": 0,
                    },
                    "count": {
                        "type": "integer",
                        "description": "每页返回条数，最多20条，默认10条",
                        "default": 10,
                    },
                    "mkt": {
                        "type": "string",
                        "description": "Market codes（市场代码），搜索中文内容时使用 zh-CN，搜索英文内容时使用 en-US，默认值 zh-CN",
                        "default": "zh-CN",
                    },
                },
            },
            "required": ["query"],
        },
    },
    "create_sub_agents": {
        "type": "function",
        "function": {
            "name": "create_sub_agents",
            "description": "创建agent函数，可以创建一个或多个Agent，每个agent可以根据输入的prompt完成特定的任务。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_agents": {
                        "type": "array",
                        "description": "创建的agent列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "此agent的prompt",
                                },
                                "index": {
                                    "type": "integer",
                                    "description": "此Agent的编号，int类型，每一个agent的编号需要不同。",
                                },
                            },
                        },
                        "required": ["prompt", "index"],
                    },
                    # "required": ["sub_agents"],
                },
            },
        },
    },
    "text_browser_view": {
        "type": "function",
        "function": {
            "name": "text_browser_view",
            "description": "这是一个链接浏览工具，可以打开链接（可以是网页、pdf等）并根据需求描述汇总页面上的所有相关信息。对所有有价值的链接都可以调用该工具来获取信息，有价值的链接包括但不限于以下几种：1.任务中明确提供的网址，2.搜索结果提供的带有相关摘要的网址，3. 之前调用TextBrowserView返回的内容中包含的且判断可能含有有用信息的网址。请尽量避免自己凭空构造链接。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "目标链接，应该是一个完整的url（以 http 开头）",
                    },
                    "description": {
                        "type": "string",
                        "description": "需求描述文本，详细描述在当前url内想要获取的内容",
                    },
                },
            },
        },
    },
    "search_global": {
        "type": "function",
        "function": {
            "name": "search_global",
            "description": "这是一个联网搜索工具，输入搜索问题，返回网页列表与对应的摘要信息。搜索问题应该简洁清晰，复杂问题应该拆解成多步并一步一步搜索。如果没有搜索到有用的页面，可以调整问题描述（如减少限定词、更换搜索思路）后再次搜索。搜索结果质量和语种有关，对于中文资源可以尝试输入中文问题，非中文的资源可以尝试使用英文或对应语种。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索问题",
                    },
                    "count": {
                        "type": "integer",
                        "description": "每页返回条数，最多50条，默认10条",
                        "default": 10,
                    },
                    "summary_type": {
                        "type": "string",
                        "description": "总结类型，可选值为：short, long。默认为short",
                        "default": "short",
                    },
                    "use_english": {
                        "type": "boolean",
                        "description": "是否使用英文进行搜索，默认为false",
                        "default": False,
                    },
                },
            },
        },
    },
}


tools_api_description_en_map = {
    "search_bing": {
        "type": "function",
        "function": {
            "name": "search_bing",
            "description": "The Bing Web Search API can provide search results that are secure, ad-free, and location-aware, and it can extract relevant information from billions of web documents. With just one API call, you can leverage the power of Bing to search billions of web pages, images, videos, and news, helping your users find what they need on the World Wide Web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "The offset to start the search from. The offset must be a number between 0 and 100.",
                        "default": 0,
                    },
                    "count": {
                        "type": "integer",
                        "description": "The number of results to return. The number must be a number between 1 and 20.",
                        "default": 10,
                    },
                    "mkt": {
                        "type": "string",
                        "description": "The market to search in. The market must be a two-letter country code. Use en-US for English searches and zh-CN for searches in China.",
                        "default": "zh-CN",
                    },
                },
            },
            "required": ["query"],
        },
    },
    "create_sub_agents": {
        "type": "function",
        "function": {
            "name": "create_sub_agents",
            "description": "Creates sub-agents that can perform specific tasks based on the input prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_agents": {
                        "type": "array",
                        "description": "The sub-agents to create. Each sub-agent must have a prompt and an index.",
                        "items": {
                            "type": "object",
                            # "required": ["prompt", "index"],
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The prompt for the sub-agent.",
                                },
                                "index": {
                                    "type": "integer",
                                    "description": "The index of the sub-agent. The index must be an integer and unique for each sub-agent.",
                                },
                            },
                        },
                        "required": ["prompt", "index"],
                    },
                    # "required": ["sub_agents"],
                },
            },
        },
    },
    "text_browser_view": {
        "type": "function",
        "function": {
            "name": "text_browser_view",
            "description": "This is a link reading tool that can open links (which can be web pages, PDFs, etc.) and summarize all relevant information on the page according to the requirement description. This tool can be called to obtain information for all valuable links. Valuable links include but are not limited to the following types: 1. URLs explicitly provided in the task; 2. URLs with relevant summaries provided in search results; 3. URLs contained in the content returned by previous calls to TextBrowserView that are judged to potentially contain useful information. Please try to avoid constructing links out of thin air by yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Target link: should be a complete URL (starting with http)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Requirement description text: a detailed description of the content to be obtained within the current URL",
                    },
                },
            },
        },
    },
    "search_global": {
        "type": "function",
        "function": {
            "name": "search_global",
            "description": "This is a search tool. Enter search queries, and it will return a list of web pages along with their corresponding summary information. Search queries should be concise and clear; complex questions should be broken down into multiple steps and searched step by step. If no useful pages are found, you can adjust the question description (such as reducing qualifiers or changing the search approach) and search again. The quality of search results is related to the language: for Chinese resources, you can try entering Chinese queries; for non-Chinese resources, you can try using English or the corresponding language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "question to be searched.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "The number of results to return. Must be less than or equal to 50, and default is 10",
                        "default": 10,
                    },
                    "summary_type": {
                        "type": "string",
                        "description": "Summary type, optional values are: short, long. Default is short",
                        "default": "short",
                    },
                    "use_english": {
                        "type": "boolean",
                        "description": "Whether to use English for search, default is false",
                        "default": False,
                    },
                },
            },
        },
    },
}


def get_system_prompt(language: str) -> str:
    if language == "zh":
        return default_system_prompt_zh
    elif language == "en":
        return default_system_prompt_en
    else:
        raise ValueError(f"Unknown language {language}")


def get_multi_agent_system_prompt(language: str) -> str:
    if language == "zh":
        return multi_agent_default_system_prompt_zh
    elif language == "en":
        return multi_agent_default_system_prompt_en
    else:
        raise ValueError(f"Unknown language {language}")


def _apply_url_map_mode(spec: dict, lang: str, mode: str) -> dict:
    """Mutate a tool spec's description fields based on URL_MAP_MODE.

    Static file text is the URL-only original WideSearch wording (mode "url",
    the default). This function only mutates when mode != "url".

    Modes:
      "url"    → no change (static text already URL-only)
      "doc_id" → search_global mentions [doc_id]; text_browser_view url param
                 is described as a doc_id
      "both"   → search_global mentions both [Url] and [doc_id];
                 text_browser_view url param accepts either form

    Only affects search_global (mentions the result fields) and
    text_browser_view (its url parameter wording).
    """
    if mode == "url":
        return spec

    name = spec["function"]["name"]
    fn = spec["function"]
    params = fn.get("parameters", {}).get("properties", {})

    if name == "search_global":
        if mode == "doc_id":
            if lang == "zh":
                fn["description"] = "这是一个联网搜索工具，输入搜索问题，返回网页列表与对应的摘要信息。每个结果带一个 doc_id（如 'doc_3'），调用 text_browser_view 时传该 doc_id 即可。搜索问题应该简洁清晰，复杂问题应该拆解成多步并一步一步搜索。如果没有搜索到有用的页面，可以调整问题描述（如减少限定词、更换搜索思路）后再次搜索。搜索结果质量和语种有关。"
            else:
                fn["description"] = "This is a search tool. Enter search queries, and it will return a list of web pages along with their corresponding summary information. Each result carries a doc_id (e.g. 'doc_3'); pass that doc_id to text_browser_view to open it. Search queries should be concise and clear; complex questions should be broken down into multiple steps. If no useful pages are found, adjust the query and search again."
        elif mode == "both":
            if lang == "zh":
                fn["description"] = "这是一个联网搜索工具，输入搜索问题，返回网页列表与对应的摘要信息。每个结果同时包含 [Url] 和 [doc_id] 两个字段，调用 text_browser_view 时任选其一传入即可。搜索问题应该简洁清晰，复杂问题应该拆解成多步并一步一步搜索。如果没有搜索到有用的页面，可以调整问题描述（如减少限定词、更换搜索思路）后再次搜索。搜索结果质量和语种有关，对于中文资源可以尝试输入中文问题，非中文的资源可以尝试使用英文或对应语种。"
            else:
                fn["description"] = "This is a search tool. Enter search queries, and it will return a list of web pages along with their corresponding summary information. Each result includes both a [Url] field and a [doc_id] field; either form can be passed to text_browser_view. Search queries should be concise and clear; complex questions should be broken down into multiple steps and searched step by step. If no useful pages are found, you can adjust the question description (such as reducing qualifiers or changing the search approach) and search again. The quality of search results is related to the language: for Chinese resources, you can try entering Chinese queries; for non-Chinese resources, you can try using English or the corresponding language."

    elif name == "text_browser_view":
        url_param = params.get("url", {})
        if mode == "doc_id":
            if lang == "zh":
                url_param["description"] = "search_global 返回的 doc_id（如 'doc_3'）"
            else:
                url_param["description"] = "A doc_id from search_global results (e.g. 'doc_3')"
        elif mode == "both":
            if lang == "zh":
                url_param["description"] = "目标链接。可以是完整的 http(s):// URL，也可以是 search_global 返回结果中的 doc_id（如 'doc_3'）；两者都被接受，系统会自动解析。"
            else:
                url_param["description"] = "Target link. May be a complete http(s):// URL, or a doc_id from a search_global result (e.g. 'doc_3'); both are accepted and the system will resolve either form."

    return spec


def get_tools_api_description(language: str, func_list: list[str]) -> list[dict]:
    import copy
    import os as _os
    mode = _os.environ.get("URL_MAP_MODE", "url")
    if language == "zh":
        base = [tools_api_description_zh_map[k] for k in func_list]
    elif language == "en":
        base = [tools_api_description_en_map[k] for k in func_list]
    else:
        raise ValueError(f"Unknown language {language}")
    # Deep-copy so per-call mode adjustments don't mutate the module-level dict.
    return [_apply_url_map_mode(copy.deepcopy(spec), language, mode) for spec in base]
