# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import functools
import json
import os
import traceback
from typing import Annotated, Any, Awaitable, Callable, Dict, List, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field


# Per-session URL ↔ doc_id mapping. Small models hallucinate URLs (rewriting
# letters, dropping path segments) when copying long strings from search
# results into text_browser_view calls. To eliminate this, search_global
# returns opaque short doc_ids (e.g. "doc_3") instead of full URLs, and
# text_browser_view resolves them back via this map. Raw URLs are still
# accepted by text_browser_view as a fallback (e.g. for hard-coded URLs in
# the user query or for URLs the model knows from training).
#
# Keyed by compact-backend session_id when running through compact_client,
# else "default" (fine for thread_num=1 batches; multi-thread non-compact
# would share namespace).
_doc_id_map: Dict[str, Dict[str, str]] = {}        # session_id -> {doc_id: url}
_doc_id_reverse: Dict[str, Dict[str, str]] = {}    # session_id -> {url: doc_id} (dedup)
_doc_id_counter: Dict[str, int] = {}               # session_id -> next int


def _get_doc_session_id() -> str:
    try:
        from src.utils.compact_client import get_session_id
        return get_session_id() or "default"
    except Exception:
        return "default"


def _register_url(url: str) -> str:
    """Register URL under the current session; return its doc_id. Idempotent
    per (session_id, url): same URL twice yields the same doc_id."""
    if not url:
        return ""
    sid = _get_doc_session_id()
    forward = _doc_id_map.setdefault(sid, {})
    reverse = _doc_id_reverse.setdefault(sid, {})
    if url in reverse:
        return reverse[url]
    n = _doc_id_counter.get(sid, 0) + 1
    _doc_id_counter[sid] = n
    doc_id = f"doc_{n}"
    forward[doc_id] = url
    reverse[url] = doc_id
    return doc_id


def _resolve_to_url(token: str) -> str:
    """If token is a known doc_id for the current session, return the mapped
    URL. If it already looks like a URL, return as-is. Otherwise (unknown
    doc_id or freeform text), return unchanged — downstream fetch will fail
    visibly and the model will see the error."""
    if not token:
        return token
    if token.startswith(("http://", "https://")):
        return token
    sid = _get_doc_session_id()
    forward = _doc_id_map.get(sid, {})
    if token in forward:
        return forward[token]
    return token


class InternalResponse(BaseModel):
    data: object | None = None
    """The data of the response."""

    error: str | None = None
    """The error message of the response."""

    system_error: str | None = None
    """The system error message of the response."""

    extra: dict | None = None


class BingSearchRequest(BaseModel):
    q: str = Field(description="query key")
    count: int = Field(default=10, ge=1, le=50)
    offset: int = Field(default=0, ge=0)
    mkt: str = Field(default="")
    safeSearch: str = Field(default="Moderate")
    responseFilter: List[str] = Field(default=[])
    freshness: str | None = Field(default=None)
    answerCount: Optional[Annotated[int, Field(ge=1)]] = Field(default=None)
    promote: List[str] = Field(default=[])
    textDecorations: bool = Field(default=False)
    textFormat: str = Field(default="Raw")


async def async_bing_search_basic(request_data: BingSearchRequest, api_key=""):
    # Two auth paths supported:
    #   1. BING_APPID env var → MS-internal endpoint bingapis.com, AppID
    #      passed via URL param `appid`, no header.
    #   2. Default (BING_APPID unset) → Azure public endpoint, key passed via
    #      `Ocp-Apim-Subscription-Key` header.
    appid = os.getenv("BING_APPID")
    params_from_req = request_data.model_dump(
        mode="json", exclude_none=True, exclude_unset=True
    )
    if params_from_req.get("mkt", ""):
        params_from_req["setLang"] = params_from_req["mkt"]

        if "en" in params_from_req["mkt"]:
            params_from_req["ensearch"] = 1

    if appid:
        url = os.getenv("BING_SEARCH_URL", "https://www.bingapis.com/api/v7/search")
        params_from_req["appid"] = appid
        params_from_req["traffictype"] = "Internal_monitor"
        headers = {}
        # bingapis.com presents a cert that aiohttp's default trust store
        # doesn't verify on every host (Python ssl module on some Linux
        # distros ignores system CA bundle). Pin to certifi's bundle.
        import ssl
        import certifi
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
    else:
        url = os.getenv("BING_SEARCH_URL", "https://api.bing.microsoft.com/v7.0/search")
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        connector = None

    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.get(
            url, headers=headers, params=params_from_req
        ) as response:
            response.raise_for_status()
            return await response.json()


def return_error(error_msg: str, verbose: bool, req: str, context: str):
    warning_msg = f"req={req}, context={context}"
    logger.warning(f"error_msg={error_msg}, {warning_msg}")
    if not verbose:
        return error_msg
    else:
        return error_msg + f"\n{warning_msg}"


# search tools


def timeout_handler(timeout: int = 120):
    def decorator(
        func: Callable[..., Awaitable[InternalResponse]],
    ) -> Callable[..., Awaitable[InternalResponse]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                resp = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                return resp
            except asyncio.TimeoutError:
                return InternalResponse(error="TimeoutError")
            except Exception:
                return InternalResponse(system_error=traceback.format_exc())

        return wrapper

    return decorator


async def search_bing(
    query: str,
    offset: int = 0,
    count: int = 10,
    mkt: str = "zh-CN",
    verbose: bool = True,
):
    api_key = str(os.getenv("BingSearch_APIKEY"))

    try:
        bing_search_request = BingSearchRequest(
            q=query,
            offset=offset,
            count=count,
            mkt=mkt,
        )
    except Exception:
        return InternalResponse(
            error=return_error(
                error_msg="ERROR: not valid argument search_bing",
                verbose=verbose,
                req=query,
                context=traceback.format_exc(),
            )
        )

    try:
        r = await async_bing_search_basic(bing_search_request, api_key)  # noqa: E501
        sections = []
        for num, web_page in enumerate(r.get("webPages", {}).get("value", []), start=1):
            lines = []
            lines.append(f"[index] {num}")
            lines.append(f"[title] {web_page.get('name', '')}")
            lines.append(f"[datePublished] {web_page.get('datePublished', '')}")
            lines.append(f"[siteName] {web_page.get('siteName', '')}")
            # URL/doc_id emission gated by URL_MAP_MODE env var:
            #   "url"    (default) — show only [Url]; matches original WideSearch
            #   "doc_id"           — show only [doc_id]
            #   "both"             — show both [Url] and [doc_id]
            # _register_url is always called so the back-end map is populated
            # regardless of what's shown; text_browser_view resolves either form.
            url = web_page.get('url', '')
            doc_id = _register_url(url)
            mode = os.environ.get("URL_MAP_MODE", "url")
            if mode in ("url", "both"):
                lines.append(f"[Url] {url}")
            if mode in ("doc_id", "both"):
                lines.append(f"[doc_id] {doc_id}")
            lines.append(f"[snippt] {web_page.get('snippet', '')}")
            sections.append("\n".join(lines))
        return InternalResponse(data="\n\n".join(sections))

    except Exception:
        return InternalResponse(
            error=return_error(
                "SYSTEM_ERROR",
                verbose=True,
                req=query,
                context=traceback.format_exc(),
            )
        )


@timeout_handler(timeout=120)
async def search_global(
    query: str,
    count: int = 10,
    summary_type: str = "short",
    use_english: bool = False,
):
    # Some models (e.g. Gemma-4) serialize numeric tool-call arguments as JSON
    # strings ("10"). The `int` type hint is not enforced at runtime, so coerce
    # defensively — otherwise the `count > 50` comparison and the downstream
    # Bing request raise TypeError ('>' not supported between str and int),
    # which surfaces to the model as a tool failure and makes it give up early.
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 10
    if not query:
        return InternalResponse(
            error=return_error(
                error_msg="error: query is empty",
                verbose=True,
                req=query,
                context="",
            )
        )
    if summary_type not in ["short", "long"]:
        return InternalResponse(
            error=return_error(
                error_msg=f'summary_type="{summary_type}" not in ["short", "long"]',
                verbose=True,
                req=query,
                context="",
            )
        )

    if count > 50:
        return InternalResponse(
            error=return_error(
                error_msg=f"count={count} exceeds the maximum of 50; please retry with count<=50",
                verbose=True,
                req=query,
                context="",
            )
        )

    # Per README §Configuration ("Implement custom search tools"), the upstream
    # `search_global` / `text_browser_view` are Bytedance-internal placeholders.
    # When SEARCH_TOOL_API_URL is unset, fall back to Bing (configured via
    # BING_APPID for the MS-internal endpoint or BingSearch_APIKEY for the
    # public Azure endpoint — see `async_bing_search_basic`).
    if not os.getenv("SEARCH_TOOL_API_URL"):
        mkt = "en-US" if use_english else "zh-CN"
        return await search_bing(query=query, count=min(count, 50), mkt=mkt, verbose=False)

    try:
        arguments = {
            "query": query,
            "count": count,
            "SummaryType": summary_type,
        }
        if use_english:
            arguments["Filter"] = {"Language": "global"}

        payload = {
            "name": "GlobalSearch",
            "arguments": json.dumps(arguments),
            "traffic_group": os.getenv("TEXTBROWSER_TRAFFIC_GROUP", ""),
            "traffic_id": os.getenv("TEXTBROWSER_TRAFFIC_ID", ""),
            "mcp_namespace": "search_tool_api",
        }

        async def _request_api():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120)
            ) as session:
                async with session.post(
                    os.getenv("SEARCH_TOOL_API_URL", ""),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result, response.headers

        result, headers = await _request_api()

        try:
            assert "result" in result
            result = json.loads(result["result"])
            assert "documents" in result, f"documents not in {result}"
            documents = result["documents"]
        except Exception:
            return InternalResponse(
                error=return_error(
                    error_msg="search_global invalid result",
                    verbose=True,
                    req=query,
                    context=(
                        traceback.format_exc()
                        + "\n"
                        + f"search_global invalid result={result}, headers={headers}, payload={payload}"
                    ),
                )
            )

        sections = []
        for num, document in enumerate(documents):
            if "render" not in document:
                continue
            link = document["render"]["link"]
            snippet = document["content"][0]["text"]

            lines = []
            lines.append(f"[index] {num}")
            lines.append(f"[siteName] {link.get('sitename', '')}")
            lines.append(f"[snippt] {snippet}")
            sections.append("\n".join(lines))

        # snippets = _parse_response(query, documents)

        return InternalResponse(data="\n\n".join(sections))

    except Exception:
        return InternalResponse(
            error=return_error(
                "SYSTEM_ERROR",
                verbose=True,
                req=query,
                context=traceback.format_exc(),
            )
        )


# reader tools


@timeout_handler(timeout=120)
async def text_browser_view(url: str, description: str = ""):
    # ``description`` is optional context only. Some models (e.g. Gemma-4) omit
    # it; keep a default so the call doesn't raise "missing required positional
    # argument: 'description'" (a TypeError the model sees as a tool failure).
    # If the model passed a doc_id from search_global (e.g. "doc_3"), resolve
    # it to the real URL via the per-session map. Raw URLs pass through
    # unchanged. Unknown doc_ids fall through too — downstream fetch will
    # surface a clear error.
    resolved = _resolve_to_url(url)
    if resolved != url:
        logger.info(f"text_browser_view: resolved {url} -> {resolved}")
        url = resolved

    # Fallback (when SEARCH_TOOL_API_URL is unset, per README §Configuration):
    # primary path uses Jina Reader (https://r.jina.ai/<url>) which renders
    # JS-heavy pages and bypasses most anti-bot blocks (sites like QS /
    # topuniversities.com return 403 to plain aiohttp).
    # Secondary path: naive aiohttp + BeautifulSoup. Truncated to ~16k chars.
    if not os.getenv("SEARCH_TOOL_API_URL"):
        import ssl
        import certifi
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        def _connector():
            return aiohttp.TCPConnector(ssl=ssl_ctx)

        # Primary: Jina Reader. Optional auth via JINA_API_KEYS (comma-separated;
        # any one is picked). Without a key, hits the free-tier rate limit.
        jina_keys = [
            k.strip()
            for k in os.getenv("JINA_API_KEYS", "").split(",")
            if k.strip()
        ]
        jina_headers = {"Accept": "text/markdown"}
        if jina_keys:
            import random
            jina_headers["Authorization"] = f"Bearer {random.choice(jina_keys)}"
            # With an auth'd key, opt into the browser engine — real headless
            # rendering that bypasses most anti-bot protections (QS,
            # topuniversities.com etc. return 403 to Jina's default engine).
            # Costs ~10x more tokens per call; only worth it when authenticated.
            jina_headers["X-Engine"] = "browser"
        jina_url = f"https://r.jina.ai/{url}"
        jina_traceback = None
        try:
            async with aiohttp.ClientSession(
                connector=_connector(),
                timeout=aiohttp.ClientTimeout(total=90),
                headers=jina_headers,
            ) as session:
                async with session.get(jina_url) as response:
                    response.raise_for_status()
                    text = await response.text()
            if len(text) > 16000:
                text = text[:16000] + "\n\n[...truncated by Jina Reader fallback...]"
            return InternalResponse(data=text)
        except Exception:
            jina_traceback = traceback.format_exc()

        # Secondary: direct aiohttp + bs4. Used if Jina fails (rate limit,
        # network error, target site blocks Jina too).
        try:
            from bs4 import BeautifulSoup
            async with aiohttp.ClientSession(
                connector=_connector(),
                timeout=aiohttp.ClientTimeout(total=60),
                headers={"User-Agent": "Mozilla/5.0 (compatible; WideSearch-Agent/1.0)"},
            ) as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    html = await response.text()
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            if len(text) > 16000:
                text = text[:16000] + "\n\n[...truncated by direct-fetch fallback...]"
            return InternalResponse(data=text)
        except Exception:
            return InternalResponse(
                error=return_error(
                    error_msg="text_browser_view fetch failed (both Jina + direct)",
                    verbose=True,
                    req=url,
                    context=(
                        f"Jina path traceback:\n{jina_traceback}\n\n"
                        f"Direct path traceback:\n{traceback.format_exc()}"
                    ),
                )
            )

    arguments = {
        "url": url,
        "description": description,
        "is_offline": True,
        "from_mcp_call": True,
    }

    payload = {
        "name": "TextBrowserView",
        "arguments": json.dumps(arguments),
        "traffic_group": os.getenv("TEXTBROWSER_TRAFFIC_GROUP", ""),
        "traffic_id": os.getenv("TEXTBROWSER_TRAFFIC_ID", ""),
    }

    try:

        async def _request_api():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120)
            ) as session:
                async with session.post(
                    os.getenv("SEARCH_TOOL_API_URL", ""),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result, response.headers

        result, headers = await _request_api()

        try:
            assert "result" in result
            result = json.loads(result["result"])
            assert "documents" in result
            documents = result["documents"]
        except Exception:
            return InternalResponse(
                error=return_error(
                    error_msg="text_browser_view invalid result",
                    verbose=True,
                    req=url,
                    context=(
                        traceback.format_exc()
                        + "\n"
                        + "text_browser_view invalid "
                        + f"result={result}, headers={headers}, payload={payload}"
                    ),
                )
            )

        chunks = []
        if documents is None:
            return InternalResponse(data="Read url failed. No documents found.")
        for doc in documents:
            for content in doc["content"]:
                if content["type"] == "text":
                    chunks.append(content["text"])
        content = "\n".join(chunks)

        return InternalResponse(data=content)
    except Exception:
        return InternalResponse(
            error=return_error(
                "SYSTEM_ERROR",
                verbose=True,
                req=url,
                context=traceback.format_exc(),
            )
        )


_default_tools = {
    "search_global": search_global,
    "text_browser_view": text_browser_view,
}
