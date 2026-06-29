from __future__ import annotations

import os
from pathlib import Path


class Config:
    def __init__(self):
        self.tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
        self.tavily_base_url = os.environ.get("TAVILY_BASE_URL", "https://api.tavily.com")
        self.cache_ttl_seconds = int(os.environ.get("SEARCH_KIT_CACHE_TTL_SECONDS", "900"))
        self.image_dir = Path(os.environ.get("SEARCH_KIT_IMAGE_DIR", "./outputs/images")).resolve()
        self.max_result_chars = int(os.environ.get("SEARCH_KIT_MAX_RESULT_CHARS", "100000"))
        self.search_depth = os.environ.get("SEARCH_KIT_SEARCH_DEPTH", "advanced")
        self.request_timeout = 60.0
        self.max_retries = 3
        self.retry_base_delay = 1.0

    def ensure_image_dir(self) -> Path:
        self.image_dir.mkdir(parents=True, exist_ok=True)
        return self.image_dir
