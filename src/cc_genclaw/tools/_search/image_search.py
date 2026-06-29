"""Image Search tool (enhanced version)."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .cache import TTLCache, make_cache_key
from .prompts import (
    get_description_rerank_prompt,
    get_vision_rerank_user_prompt,
    get_vision_rerank_batch_user_prompt,
    VISION_RERANK_SYSTEM,
    VISION_RERANK_BATCH_SYSTEM,
)
from .tavily.client import TavilyClient
from .tavily.image_downloader import (
    download_images_parallel,
    download_images_serial_until_n,
)

logger = logging.getLogger(__name__)


_QUERY_VARIANTS = [
    "{q} official",
    "{q} HD photo",
    "{q} high resolution",
]


def _parse_json_array_lenient(text: str) -> list[dict] | None:
    """Extract a JSON array from model output (tolerates ```json fences and leading text)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    arr_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if arr_match:
        text = arr_match.group(0)
    try:
        return json.loads(text)
    except Exception:
        return None


class ImageSearchTool:
    name = "image_search"

    def __init__(
        self,
        tavily: TavilyClient,
        anthropic=None,
        rerank_model: str | None = None,
        cache: TTLCache | None = None,
        search_depth: str = "advanced",
        image_dir: str | Path = "outputs/images",
        max_result_chars: int = 100_000,
        llm_provider: str = "anthropic",
        llm_client=None,
    ):
        self.tavily = tavily
        self.anthropic = anthropic
        self.rerank_model = rerank_model or "claude-opus-4-7"
        self.cache = cache
        self.search_depth = search_depth
        self.image_dir = Path(image_dir)
        self.max_result_chars = max_result_chars
        self.llm_provider = llm_provider
        self.llm_client = llm_client if llm_provider == "openai" else anthropic

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        query = (args.get("query") or "").strip()
        if len(query) < 2:
            return {"error": "Error: Missing or too-short query"}

        count = int(args.get("count", 8))
        count = max(1, min(30, count))

        mode = args.get("mode", "all")
        if mode not in ("all", "best", "topn"):
            return {"error": f"Invalid mode: {mode}"}

        rerank_default = "none" if mode == "all" else "description"
        rerank = args.get("rerank") or rerank_default
        if rerank not in ("none", "description", "vision"):
            return {"error": f"Invalid rerank: {rerank}"}

        top_n = args.get("top_n")
        if mode == "best":
            top_n = 1
        elif mode == "topn":
            top_n = int(top_n or 3)
            top_n = max(1, min(10, top_n))
        else:
            top_n = count

        intent = (args.get("intent") or "").strip() or None
        download = bool(args.get("download", True))
        expand_pool = bool(args.get("expand_pool", True))

        cache_key = None
        if self.cache:
            cache_key = make_cache_key(
                "image_search_v2", query, count, mode, rerank, top_n, intent, download, expand_pool
            )
            cached = self.cache.get(cache_key)
            if cached:
                cached["_cache_hit"] = True
                return cached

        try:
            candidates = self._fetch_candidates(query, count, expand_pool=expand_pool)
        except Exception as e:
            logger.exception("[image_search] candidate fetch failed")
            return {"error": f"Tavily error: {type(e).__name__}: {e}"}

        if not candidates:
            return self._format(query, [], mode=mode, intent=intent, rerank=rerank, ranking_log="no candidates")

        ranking_log = ""
        candidate_pool: list[dict[str, Any]] = []
        # [2026-06-08] Gate on self.llm_client (not self.anthropic): under the openai provider
        # anthropic=None but llm_client is an openai instance; previously using self.anthropic
        # would make rerank silently not run, degrading to Tavily's original order.
        if rerank == "description" and self.llm_client:
            candidates, ranking_log = self._rerank_by_description(candidates, query, intent)
        elif rerank == "vision" and self.llm_client:
            candidates_dl = self._download_for_vision(candidates[: max(top_n, 5)], query)
            candidates_dl, ranking_log = self._rerank_by_vision(candidates_dl, query, intent)
            candidate_pool = candidates_dl
            for i, item in enumerate(candidates_dl):
                if i < len(candidates):
                    candidates[i] = item
        elif rerank != "none":
            ranking_log = f"rerank='{rerank}' requested but no LLM client; falling back to Tavily order"

        if mode == "all":
            picked = candidates
        else:
            picked = candidates[:top_n]

        if download:
            if mode == "all":
                picked = download_images_parallel(
                    picked, save_dir=self.image_dir, name_prefix=query[:40], max_workers=4, robust=True,
                )
            else:
                picked = download_images_serial_until_n(
                    candidates,
                    n_needed=top_n,
                    save_dir=self.image_dir,
                    name_prefix=query[:40],
                )
                picked = [x for x in picked if x.get("local_path")][:top_n]

        out = self._format(
            query,
            picked,
            mode=mode,
            intent=intent,
            rerank=rerank,
            ranking_log=ranking_log,
            candidate_items=candidate_pool,
        )
        if self.cache and cache_key:
            self.cache.set(cache_key, out)
        return out

    def _fetch_candidates(self, query: str, want: int, expand_pool: bool = True) -> list[dict[str, Any]]:
        seen: set[str] = set()
        pool: list[dict[str, Any]] = []

        def _ingest(images: list) -> None:
            for im in images or []:
                if isinstance(im, dict):
                    url = im.get("url") or ""
                    desc = im.get("description") or ""
                else:
                    url = str(im)
                    desc = ""
                if not url or url in seen:
                    continue
                seen.add(url)
                pool.append({"url": url, "description": desc, "page_url": ""})

        data = self.tavily.search(
            query=query, max_results=1, search_depth=self.search_depth,
            include_answer=False, include_images=True, include_image_descriptions=True,
        )
        _ingest(data.get("images") or [])

        if expand_pool and len(pool) < want:
            for variant in _QUERY_VARIANTS:
                if len(pool) >= want * 3:
                    break
                vq = variant.format(q=query)
                try:
                    d = self.tavily.search(
                        query=vq, max_results=1, search_depth="basic",
                        include_answer=False, include_images=True, include_image_descriptions=True,
                    )
                    _ingest(d.get("images") or [])
                except Exception as e:
                    logger.debug("[image_search] variant '%s' failed: %s", vq, e)

        return pool[: want * 3]

    def _rerank_by_description(
        self, candidates: list[dict], query: str, intent: str | None
    ) -> tuple[list[dict], str]:
        if not candidates:
            return candidates, "no candidates"
        prompt = get_description_rerank_prompt(query, candidates, intent)
        try:
            if self.llm_provider == "openai":
                resp = self.llm_client.chat.completions.create(
                    model=self.rerank_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.0,
                )
                text = (resp.choices[0].message.content or "") if resp.choices else ""
            else:
                resp = self.llm_client.messages.create(
                    model=self.rerank_model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        except Exception as e:
            logger.warning("[image_search] description rerank LLM call failed: %s", e)
            return candidates, f"rerank failed: {type(e).__name__}: {e}"

        scored = _parse_json_array_lenient(text)
        if not scored or not isinstance(scored, list):
            logger.warning("[image_search] description rerank: cannot parse JSON, raw=%r", text[:200])
            return candidates, "rerank parse-failed; using Tavily order"

        score_by_id: dict[int, dict] = {}
        for entry in scored:
            try:
                cid = int(entry.get("id"))
                sc = float(entry.get("score"))
                score_by_id[cid] = {"score": sc, "reason": entry.get("reason") or ""}
            except Exception:
                continue

        ranked: list[tuple[float, int, dict]] = []
        for i, c in enumerate(candidates, 1):
            info = score_by_id.get(i, {})
            sc = float(info.get("score", 0.0))
            reason = info.get("reason", "")
            new_c = dict(c)
            new_c["rerank_score"] = sc
            new_c["rerank_reason"] = reason
            ranked.append((sc, i, new_c))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        out = [r[2] for r in ranked]
        log = f"description-rerank ok ({len(score_by_id)}/{len(candidates)} scored)"
        return out, log

    def _download_for_vision(self, candidates: list[dict], query: str) -> list[dict]:
        return download_images_parallel(
            candidates,
            save_dir=self.image_dir,
            name_prefix=f"_rerank_{query[:30]}",
            max_workers=4,
            robust=True,
        )

    def _rerank_by_vision(
        self, candidates: list[dict], query: str, intent: str | None
    ) -> tuple[list[dict], str]:
        try:
            scored, ranking_log_batch, batch_ok = self._rerank_by_vision_batch(candidates, query, intent)
            if batch_ok:
                return scored, ranking_log_batch
            logger.info("[image_search] batched vision rerank degraded, falling back per-image")
        except Exception as e:
            logger.warning("[image_search] batched vision rerank exception, fallback per-image: %s", e)

        return self._rerank_by_vision_per_image(candidates, query, intent)

    def _rerank_by_vision_batch(
        self, candidates: list[dict], query: str, intent: str | None
    ) -> tuple[list[dict], str, bool]:
        import base64

        with_local: list[tuple[int, dict]] = []
        for i, c in enumerate(candidates, 1):
            if c.get("local_path"):
                with_local.append((i, c))

        if not with_local:
            scored = [dict(c, rerank_score=0.0, rerank_reason="(skipped: download failed)")
                      for c in candidates]
            return scored, "no downloaded candidates to score", True

        content_blocks: list[dict] = []
        for _, c in with_local:
            with open(c["local_path"], "rb") as f:
                img_b64 = base64.standard_b64encode(f.read()).decode()
            if self.llm_provider == "openai":
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })
            else:
                content_blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                })
        batch_text = get_vision_rerank_batch_user_prompt(query, intent, n_images=len(with_local))
        content_blocks.append({"type": "text", "text": batch_text})

        if self.llm_provider == "openai":
            resp = self.llm_client.chat.completions.create(
                model=self.rerank_model,
                messages=[
                    {"role": "system", "content": VISION_RERANK_BATCH_SYSTEM},
                    {"role": "user", "content": content_blocks},
                ],
                max_tokens=800,
                temperature=0.0,
            )
            text = (resp.choices[0].message.content or "") if resp.choices else ""
        else:
            resp = self.llm_client.messages.create(
                model=self.rerank_model,
                max_tokens=800,
                system=VISION_RERANK_BATCH_SYSTEM,
                messages=[{"role": "user", "content": content_blocks}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        m = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
        if not m:
            logger.warning("[image_search] batched rerank: cannot find JSON array; raw=%r", text[:200])
            return candidates, "batched parse failed", False
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            logger.warning("[image_search] batched rerank: JSON parse failed: %s; raw=%r", e, text[:200])
            return candidates, "batched parse failed", False

        if not isinstance(parsed, list):
            return candidates, "batched parse: not a list", False

        batch_id_to_orig: dict[int, int] = {}
        for batch_pos, (orig_pos, _) in enumerate(with_local, 1):
            batch_id_to_orig[batch_pos] = orig_pos

        score_by_orig: dict[int, tuple[float, str]] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                bid = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            if bid not in batch_id_to_orig:
                continue
            score = float(item.get("score", 0.0))
            reason = item.get("rationale") or item.get("reason", "") or ""
            score_by_orig[batch_id_to_orig[bid]] = (score, reason)

        missing = [orig for orig in batch_id_to_orig.values() if orig not in score_by_orig]
        if missing:
            logger.warning(
                "[image_search] batched rerank: %d/%d ids missing in LLM output, fallback per-image",
                len(missing), len(with_local),
            )
            return candidates, f"batched missing {len(missing)} ids", False

        scored: list[dict] = []
        for orig_pos, c in enumerate(candidates, 1):
            new_c = dict(c)
            if orig_pos in score_by_orig:
                s, r = score_by_orig[orig_pos]
                new_c["rerank_score"] = s
                new_c["rerank_reason"] = r
            else:
                new_c["rerank_score"] = 0.0
                new_c["rerank_reason"] = "(skipped: download failed)"
            scored.append(new_c)

        scored.sort(key=lambda x: -float(x.get("rerank_score") or 0.0))
        return scored, f"vision-rerank batched ok ({len(score_by_orig)}/{len(with_local)} scored, 1 LLM call)", True

    def _rerank_by_vision_per_image(
        self, candidates: list[dict], query: str, intent: str | None
    ) -> tuple[list[dict], str]:
        import base64

        scored: list[dict] = []
        success = 0
        for i, c in enumerate(candidates, 1):
            local = c.get("local_path")
            new_c = dict(c)
            if not local:
                new_c["rerank_score"] = 0.0
                new_c["rerank_reason"] = "(skipped: download failed)"
                scored.append(new_c)
                continue
            try:
                with open(local, "rb") as f:
                    img_b64 = base64.standard_b64encode(f.read()).decode()
                if self.llm_provider == "openai":
                    resp = self.llm_client.chat.completions.create(
                        model=self.rerank_model,
                        messages=[
                            {"role": "system", "content": VISION_RERANK_SYSTEM},
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                }},
                                {"type": "text", "text": get_vision_rerank_user_prompt(query, intent, i)},
                            ]},
                        ],
                        max_tokens=300,
                        temperature=0.0,
                    )
                    text = (resp.choices[0].message.content or "") if resp.choices else ""
                else:
                    resp = self.llm_client.messages.create(
                        model=self.rerank_model,
                        max_tokens=300,
                        system=VISION_RERANK_SYSTEM,
                        messages=[{"role": "user", "content": [
                            {"type": "image", "source": {
                                "type": "base64", "media_type": "image/jpeg", "data": img_b64,
                            }},
                            {"type": "text", "text": get_vision_rerank_user_prompt(query, intent, i)},
                        ]}],
                    )
                    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
                m = re.search(r"\{.*?\}", text, re.DOTALL)
                obj = json.loads(m.group(0)) if m else {}
                new_c["rerank_score"] = float(obj.get("score", 0.0))
                new_c["rerank_reason"] = obj.get("rationale") or obj.get("reason", "") or ""
                success += 1
            except Exception as e:
                logger.warning("[image_search] vision rerank %d failed: %s", i, e)
                new_c["rerank_score"] = 0.0
                new_c["rerank_reason"] = f"(vision error: {type(e).__name__})"
            scored.append(new_c)

        scored.sort(key=lambda x: -float(x.get("rerank_score") or 0.0))
        return scored, f"vision-rerank ok ({success}/{len(candidates)} scored)"

    def _format(
        self,
        query: str,
        items: list[dict[str, Any]],
        *,
        mode: str = "all",
        intent: str | None = None,
        rerank: str = "none",
        ranking_log: str = "",
        candidate_items: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        for idx, im in enumerate(items):
            im["rank"] = idx + 1
            im["role"] = "primary" if idx == 0 else "backup"

        picked_urls = {im.get("url") for im in items if im.get("url")}
        candidate_images: list[dict[str, Any]] = []
        for idx, im in enumerate(candidate_items or [], 1):
            local = im.get("local_path") or ""
            if not local:
                continue
            url = im.get("url") or ""
            candidate_images.append({
                "rank": idx,
                "role": "selected" if url and url in picked_urls else "candidate",
                "local_path": local,
                "url": url,
                "description": im.get("description") or "",
                "rerank_score": im.get("rerank_score"),
                "rerank_reason": im.get("rerank_reason") or "",
            })

        lines = [f'Image search results for query: "{query}"']
        lines.append(f"  mode={mode} rerank={rerank}" + (f" intent={intent!r}" if intent else ""))
        if ranking_log:
            lines.append(f"  rerank-log: {ranking_log}")

        if not items:
            lines.append("\nNo images returned.")
        else:
            lines.append(
                f"\n[{len(items)} image{'s' if len(items)!=1 else ''} — #1 is PRIMARY (best), rest are BACKUP]"
            )
            for i, im in enumerate(items, 1):
                url = im.get("url") or ""
                desc = (im.get("description") or "").strip()
                local = im.get("local_path") or ""
                err = im.get("download_error") or ""
                score = im.get("rerank_score")
                reason = im.get("rerank_reason") or ""
                attempts = im.get("download_attempts") or []
                role_tag = "PRIMARY" if i == 1 else "BACKUP"
                lines.append(f"\n{i}. [{role_tag}] {url}")
                if score is not None:
                    lines.append(f"   score: {float(score):.3f}{(' — ' + reason) if reason else ''}")
                if desc:
                    lines.append(f"   desc: {desc[:300]}")
                if local:
                    lines.append(f"   local_path: {local}")
                elif err:
                    lines.append(f"   (download_error: {err})")
                if attempts and not local:
                    lines.append(f"   attempts: {'; '.join(attempts[-3:])}")

        body = "\n".join(lines)
        if len(body) > self.max_result_chars:
            body = body[: self.max_result_chars] + "\n\n[truncated]"

        body += (
            "\n\nREMINDER: Image #1 is the PRIMARY (best) pick; the rest are BACKUP candidates. "
            "When embedding, use markdown ![desc](local_path-or-url) and prefer local_path. "
            "For image-generation use the PRIMARY image; fall back to a BACKUP only if needed."
        )

        return {
            "query": query,
            "mode": mode,
            "intent": intent,
            "rerank": rerank,
            "ranking_log": ranking_log,
            "images": items,
            "candidate_images": candidate_images,
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
