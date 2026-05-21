# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import functools
import json
import os
import traceback
from typing import Annotated, Any, Awaitable, Callable, List, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field


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
            lines.append(f"[Url] {web_page.get('url', '')}")
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
async def text_browser_view(url: str, description: str):
    # Fallback (when SEARCH_TOOL_API_URL is unset, per README §Configuration):
    # primary path uses Jina Reader (https://r.jina.ai/<url>) which renders
    # JS-heavy pages and bypasses most anti-bot blocks (sites like QS /
    # topuniversities.com return 403 to plain aiohttp).
    # Secondary path: naive aiohttp + BeautifulSoup. Truncated to ~32k chars.
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
