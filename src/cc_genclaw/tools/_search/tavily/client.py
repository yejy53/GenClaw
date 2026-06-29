"""Tavily HTTP client: /search + /extract."""
from __future__ import annotations

import logging
import random
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


class TavilyError(RuntimeError):
    pass


class TavilyClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.tavily.com",
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ):
        if not api_key:
            raise ValueError("TAVILY_API_KEY must not be empty")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._session = requests.Session()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = dict(payload)
        body.setdefault("api_key", self.api_key)

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(
                    url,
                    json=body,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"},
                )
            except requests.RequestException as e:
                last_exc = e
                if attempt < self.max_retries:
                    sleep = self.retry_base_delay * (2**attempt) + random.uniform(0, 0.5)
                    logger.warning("[tavily] %s request error %s, retry %d/%d in %.1fs",
                                   path, e, attempt + 1, self.max_retries, sleep)
                    time.sleep(sleep)
                    continue
                raise TavilyError(f"Tavily {path} request failed: {e}") from e

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as e:
                    raise TavilyError(f"Tavily {path} returned invalid JSON: {e}") from e

            if resp.status_code == 429:
                if attempt < self.max_retries:
                    sleep = min(20.0, self.retry_base_delay * (2**attempt)) + random.uniform(0, 5.0)
                    logger.warning("[tavily] 429 on %s, retry %d/%d in %.1fs",
                                   path, attempt + 1, self.max_retries, sleep)
                    time.sleep(sleep)
                    continue
                raise TavilyError(f"Tavily {path} rate limited (429): {resp.text[:300]}")

            if 500 <= resp.status_code < 600 and attempt < self.max_retries:
                sleep = self.retry_base_delay * (2**attempt) + random.uniform(0, 1.0)
                logger.warning("[tavily] %d on %s, retry %d/%d in %.1fs",
                               resp.status_code, path, attempt + 1, self.max_retries, sleep)
                time.sleep(sleep)
                continue

            raise TavilyError(f"Tavily {path} HTTP {resp.status_code}: {resp.text[:300]}")

        raise TavilyError(f"Tavily {path} failed after {self.max_retries + 1} attempts: {last_exc}")

    def search(
        self,
        query: str,
        *,
        max_results: int = 8,
        search_depth: str = "advanced",
        include_answer: str | bool = "advanced",
        include_images: bool = True,
        include_image_descriptions: bool = True,
        include_raw_content: bool = False,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        topic: str | None = None,
        days: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_images": include_images,
            "include_image_descriptions": include_image_descriptions,
            "include_raw_content": include_raw_content,
        }
        if allowed_domains:
            payload["include_domains"] = allowed_domains
        if blocked_domains:
            payload["exclude_domains"] = blocked_domains
        if topic:
            payload["topic"] = topic
        if days is not None:
            payload["days"] = days

        return self._post("/search", payload)

    def extract(
        self,
        urls: list[str],
        *,
        include_images: bool = False,
        extract_depth: str = "advanced",
    ) -> dict[str, Any]:
        if isinstance(urls, str):
            urls = [urls]
        payload = {
            "urls": urls,
            "include_images": include_images,
            "extract_depth": extract_depth,
        }
        return self._post("/extract", payload)
