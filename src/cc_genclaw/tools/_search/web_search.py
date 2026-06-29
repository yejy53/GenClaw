"""Web Search tool."""
from __future__ import annotations

import json
import logging
from typing import Any

from .cache import TTLCache, make_cache_key
from .tavily.client import TavilyClient

logger = logging.getLogger(__name__)


class WebSearchTool:
    name = "web_search"

    def __init__(
        self,
        tavily: TavilyClient,
        cache: TTLCache | None = None,
        search_depth: str = "advanced",
        max_result_chars: int = 100_000,
    ):
        self.tavily = tavily
        self.cache = cache
        self.search_depth = search_depth
        self.max_result_chars = max_result_chars

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        query = (args.get("query") or "").strip()
        if len(query) < 2:
            return {"error": "Error: Missing or too-short query (need >= 2 chars)", "errorCode": 1}

        allowed = args.get("allowed_domains")
        blocked = args.get("blocked_domains")
        if allowed and blocked:
            return {
                "error": "Cannot specify both allowed_domains and blocked_domains in the same request",
                "errorCode": 2,
            }

        max_results = int(args.get("max_results", 8))
        max_results = max(1, min(15, max_results))
        include_images = bool(args.get("include_images", True))

        cache_key = None
        if self.cache:
            cache_key = make_cache_key("web_search", query, max_results, allowed, blocked, include_images)
            cached = self.cache.get(cache_key)
            if cached:
                cached["_cache_hit"] = True
                return cached

        try:
            data = self.tavily.search(
                query=query,
                max_results=max_results,
                search_depth=self.search_depth,
                include_answer="advanced",
                include_images=include_images,
                include_image_descriptions=include_images,
                allowed_domains=list(allowed) if allowed else None,
                blocked_domains=list(blocked) if blocked else None,
            )
        except Exception as e:
            logger.exception("[web_search] Tavily call failed")
            return {"error": f"Tavily error: {type(e).__name__}: {e}", "errorCode": 3}

        out = self._format(query, data)
        if self.cache and cache_key:
            self.cache.set(cache_key, out)
        return out

    def _format(self, query: str, data: dict[str, Any]) -> dict[str, Any]:
        results = data.get("results") or []
        images = data.get("images") or []
        answer = data.get("answer") or ""

        lines = [f'Web search results for query: "{query}"']
        if answer:
            lines.append("\n[Synthesized Answer]")
            lines.append(answer)

        if not results:
            lines.append("\nNo organic results.")
        else:
            lines.append(f"\n[Organic Results: {len(results)}]")
            for i, r in enumerate(results, 1):
                title = r.get("title") or "(untitled)"
                url = r.get("url") or ""
                score = r.get("score") or 0.0
                snippet = (r.get("content") or "").strip().replace("\n", " ")
                if len(snippet) > 800:
                    snippet = snippet[:800] + "..."
                lines.append(f"\n{i}. [{title}]({url})  score={score:.2f}")
                if snippet:
                    lines.append(f"   {snippet}")

        if images:
            lines.append(f"\n[Images: {len(images)}]")
            for i, im in enumerate(images, 1):
                if isinstance(im, dict):
                    desc = (im.get("description") or "").strip()
                    url = im.get("url") or ""
                    lines.append(f"  {i}. {url}")
                    if desc:
                        lines.append(f"     desc: {desc[:200]}")
                else:
                    lines.append(f"  {i}. {im}")

        body = "\n".join(lines)
        if len(body) > self.max_result_chars:
            body = body[: self.max_result_chars] + f"\n\n[truncated, original length={len(body)}]"

        body += (
            "\n\nREMINDER: You MUST include the sources above in your response to the user "
            "using markdown hyperlinks under a 'Sources:' section."
        )

        compact_results = [
            {"title": r.get("title"), "url": r.get("url"), "score": r.get("score")}
            for r in results
        ]
        compact_images = [
            {"url": im.get("url"), "description": im.get("description")} if isinstance(im, dict) else {"url": im}
            for im in images
        ]

        return {
            "query": query,
            "answer": answer,
            "results": compact_results,
            "images": compact_images,
            "content": body,
        }

    @staticmethod
    def to_tool_result_block(tool_use_id: str, output: dict[str, Any]) -> dict[str, Any]:
        if "error" in output:
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "is_error": True,
                "content": output["error"],
            }
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": output.get("content") or json.dumps(output, ensure_ascii=False),
        }
