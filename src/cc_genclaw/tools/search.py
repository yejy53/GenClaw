"""SEARCH tool — faithful executor for the queries the main LLM hands us.

[2026-06-08] Rewrite (v3): search no longer nests a ReAct sub-agent that
re-interprets / repackages / guesses answers. It is just a "hand": the main
LLM hands it a few things to search, and it faithfully searches **each one**
and returns the results faithfully. Deciding "what to search and how to hop"
is the job of the main LLM (the brain), not of search.

Why this design:
- The v2 nested agent would package "question 1/2/3" into a single query it
  guessed itself (with pretrained entities); essentially "fill in the blanks
  from priors then confirm", which has confirmation bias and is not truly
  hop-by-hop.
- True hop-by-hop should happen at the [main LLM] level: the main LLM calls
  search once to get facts -> reads -> resolves pronouns into entities ->
  calls search again for the next hop / image search. search only executes
  faithfully.

Execution semantics:
- text_queries (U need_process_problem): each runs web_search **independently**,
  returning facts one by one. Queries must be self-contained (the main LLM is
  responsible for resolving "that game" into "Minecraft" before passing it in);
  search does not interpret pronouns for it.
- image_queries: each runs image_search once (rerank=vision, download+verify).
  final_count is the **per image query** cap; multiple image queries no longer
  share one global truncation budget, so the first query's backup images do not
  crowd out later queries' primary images.

Underneath it uses the bundled _search.WebSearchTool / ImageSearchTool (Tavily
engine + robust download + VLM/description rerank), without going through any
nested agent's LLM loop.

The external interface/contract is unchanged (downstream i2i / format_prompt as
before):
ToolResult.content (JSON):
- reference_paths: List[str] absolute (grouped in image_queries order; best-first per group)
- text_queries / image_queries: List[str] (queries actually run)
- enriched_prompt: str (= user_intent, not rewritten)
- facts: str (per-query web_search fact summary)
- gate_action: compatibility field, fixed 'faithful'
- status: 'success'|'failed'
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from ..runtime.session_dir import tool_output_dir
from ..runtime.artifacts import record_artifact, record_artifacts_in_dir
from .base import BaseTool

# In-process reuse (TavilyClient / TTLCache / OpenAI client / shared across calls by both tools).
# image_dir is overwritten on each execute to the current session's search output directory.
_TOOLS: Dict[str, Any] = {}


def _get_tools() -> Dict[str, Any]:
    """Lazily load the self-contained underlying tools.

    Relies on load_genclaw_config() to inject TAVILY / OPENAI credentials into os.environ.
    """
    if _TOOLS:
        return _TOOLS

    from ._search.cache import TTLCache  # noqa: WPS433
    from ._search.config import Config  # noqa: WPS433
    from ._search.tavily.client import TavilyClient  # noqa: WPS433
    from ._search.image_search import ImageSearchTool  # noqa: WPS433
    from ._search.web_search import WebSearchTool  # noqa: WPS433

    cfg = Config()  # read TAVILY_* / SEARCH_KIT_* defaults

    tavily = TavilyClient(
        api_key=cfg.tavily_api_key,
        base_url=cfg.tavily_base_url,
        timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
        retry_base_delay=cfg.retry_base_delay,
    )
    cache = TTLCache(ttl_seconds=cfg.cache_ttl_seconds)

    # image_search rerank uses the OpenAI protocol (reuses the main LLM's endpoint).
    # Only used to score/sort candidate images (pick the best), not to re-interpret the query.
    from openai import OpenAI  # noqa: WPS433

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    rerank_model = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.4")
    oai = OpenAI(api_key=api_key, base_url=base_url)

    web = WebSearchTool(
        tavily=tavily,
        cache=cache,
        search_depth=cfg.search_depth,
        max_result_chars=cfg.max_result_chars,
    )
    img = ImageSearchTool(
        tavily=tavily,
        anthropic=None,
        rerank_model=rerank_model,
        cache=cache,
        search_depth=cfg.search_depth,
        image_dir=cfg.image_dir,
        max_result_chars=cfg.max_result_chars,
        llm_provider="openai",
        llm_client=oai,
    )

    _TOOLS["web"] = web
    _TOOLS["image"] = img
    return _TOOLS


def _dedup_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for it in items:
        s = (it or "").strip()
        if not s or s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
    return out


def _verify_and_curate(paths: List[str], final_count: int) -> List[str]:
    """Cheap second-pass: filter hidden + PIL verify, truncate (best-first)."""
    try:
        from PIL import Image  # type: ignore
        has_pil = True
    except Exception:
        has_pil = False

    out: List[str] = []
    seen: set[str] = set()
    for p in paths:
        if not p or p in seen:
            continue
        seen.add(p)
        name = os.path.basename(p)
        if name.startswith(".") or name.startswith("._"):
            continue
        if not os.path.isfile(p):
            continue
        if has_pil:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                continue
        out.append(os.path.abspath(p))
        if len(out) >= final_count:
            break
    return out


_REFERENCE_ASPECT_MIN = 0.5
_REFERENCE_ASPECT_MAX = 2.0


def _normalize_reference_for_return(path: str) -> Dict[str, Any]:
    """Return a stable reference path, padding extreme aspect ratios if needed."""
    meta: Dict[str, Any] = {
        "path": os.path.abspath(path),
        "original_path": os.path.abspath(path),
        "normalization": "none",
    }
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            src = im.convert("RGB")
            width, height = src.size
            if width <= 0 or height <= 0:
                return meta

            aspect = width / height
            meta["aspect_ratio"] = aspect
            if _REFERENCE_ASPECT_MIN <= aspect <= _REFERENCE_ASPECT_MAX:
                return meta

            if aspect < _REFERENCE_ASPECT_MIN:
                canvas_w = int(round(height * _REFERENCE_ASPECT_MIN))
                canvas_h = height
            else:
                canvas_w = width
                canvas_h = int(round(width / _REFERENCE_ASPECT_MAX))

            out_dir = Path(path).parent / "_normalized_refs"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{Path(path).stem}_aspect_padded.png"
            if not out_path.is_file():
                canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
                canvas.paste(src, ((canvas_w - width) // 2, (canvas_h - height) // 2))
                canvas.save(out_path, "PNG")

            meta.update({
                "path": os.path.abspath(str(out_path)),
                "normalized_from": os.path.abspath(path),
                "normalization": "aspect_padded",
                "normalized_size": [canvas_w, canvas_h],
            })
            return meta
    except Exception:
        return meta


def _reference_role_from_reason(reason: str, query: str, intent: str) -> str:
    text = f"{reason} {query} {intent}".lower()
    form_terms = (
        "plush", "toy", "keychain", "figurine", "sticker", "costume",
        "packaging", "product mockup", "cake", "sculpture", "accessory",
        "挂件", "毛绒", "玩具", "手办", "钥匙扣",
    )
    identity_terms = ("identity", "outfit", "character", "costume", "身份", "服装", "角色")
    product_terms = ("product", "device", "phone", "camera", "bag", "产品", "手机", "相机", "包")
    if any(term in text for term in form_terms):
        return "form_reference"
    if any(term in text for term in product_terms):
        return "product_reference"
    if any(term in text for term in identity_terms):
        return "identity_reference"
    return "visual_reference"


class SearchTool(BaseTool):
    name = "search"
    description = ""  # filled from yaml
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "text_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Self-contained web-search queries to run, ONE BY ONE. Each must "
                    "stand on its own (resolve pronouns yourself: pass 'Who created "
                    "Minecraft?', NOT 'Who created that game?'). For a fact chain, call "
                    "search again with the next hop after reading this turn's facts."
                ),
            },
            "image_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Image-search subjects to run, ONE BY ONE (e.g. 'flag of Russia'). "
                    "Ground these in facts you ALREADY verified — typically a second "
                    "search call after the text facts come back."
                ),
            },
            "need_process_problem": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Alias for text_queries (dictionary-lookup style questions). "
                    "Merged with text_queries; each runs as its own web_search."
                ),
            },
            "final_count": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 8,
                "description": (
                    "Max reference images to return PER image query (best-first). Default 1."
                ),
            },
            "user_intent": {
                "type": "string",
                "description": "Original user prompt (used as image-rerank intent only; not rewritten).",
            },
        },
        "additionalProperties": False,
    }

    def execute(self, args: Dict[str, Any], signal: Any = None):
        from cc_genclaw.config import load_genclaw_config

        load_genclaw_config()  # push env (TAVILY / OPENAI creds) + add search_kit to path

        text_qs = _dedup_keep_order(
            list(args.get("text_queries") or []) + list(args.get("need_process_problem") or [])
        )
        image_qs = _dedup_keep_order(list(args.get("image_queries") or []))
        final_count = int(args.get("final_count", 1))
        user_intent = str(args.get("user_intent") or "").strip()

        if not text_qs and not image_qs:
            return self.err("search: provide text_queries / need_process_problem and/or image_queries")

        try:
            tools = _get_tools()
        except Exception as e:  # noqa: BLE001
            return self.err(f"search: search_kit tools init failed: {type(e).__name__}: {e}")

        web = tools["web"]
        img = tools["image"]

        out_dir = tool_output_dir("search")
        try:
            img.image_dir = Path(out_dir)
        except Exception:  # noqa: BLE001
            pass

        # ---- 1) Per-query text search (run one by one, independent of each other) ----
        fact_blocks: List[str] = []
        web_trace: List[Dict[str, Any]] = []
        for q in text_qs:
            out = web.call({"query": q, "include_images": False, "max_results": 6})
            if "error" in out:
                fact_blocks.append(f"### {q}\n(search error: {out['error']})")
                web_trace.append({"query": q, "error": out["error"]})
                continue
            answer = (out.get("answer") or "").strip()
            results = out.get("results") or []
            block = [f"### {q}"]
            if answer:
                block.append(answer)
            for r in results[:3]:
                title = r.get("title") or "(untitled)"
                url = r.get("url") or ""
                block.append(f"- [{title}]({url})")
            fact_blocks.append("\n".join(block))
            web_trace.append({
                "query": q,
                "answer": answer,
                "sources": [{"title": r.get("title"), "url": r.get("url")} for r in results[:5]],
            })

        # ---- 2) Per-query image search (run one by one) ----
        all_image_paths: List[str] = []
        curated: List[str] = []
        all_image_results: List[Dict[str, Any]] = []
        all_candidate_results: List[Dict[str, Any]] = []
        image_trace: List[Dict[str, Any]] = []
        mode = "best" if final_count <= 1 else "topn"
        for q in image_qs:
            out = img.call({
                "query": q,
                "mode": mode,
                "top_n": final_count,
                "rerank": "vision",
                "intent": user_intent or None,
                "download": True,
            })
            if "error" in out:
                image_trace.append({"query": q, "error": out["error"]})
                continue
            imgs = out.get("images") or []
            candidate_imgs = out.get("candidate_images") or []
            paths = [im.get("local_path") for im in imgs if im.get("local_path")]
            all_image_paths.extend(paths)
            original_curated = _verify_and_curate(paths, final_count)
            curated_records = [_normalize_reference_for_return(path) for path in original_curated]
            query_curated = [record["path"] for record in curated_records]
            curated.extend(query_curated)
            images_by_path = {
                os.path.abspath(str(im.get("local_path"))): im
                for im in imgs
                if im.get("local_path")
            }
            query_image_results: List[Dict[str, Any]] = []
            for rank, record in enumerate(curated_records, 1):
                path = record["path"]
                original_path = record.get("original_path") or path
                source = images_by_path.get(os.path.abspath(str(original_path)), {})
                score = source.get("rerank_score")
                item: Dict[str, Any] = {
                    "query": q,
                    "rank": rank,
                    "role": "primary" if rank == 1 else "backup",
                    "path": path,
                    "original_path": original_path,
                    "normalization": record.get("normalization"),
                    "url": source.get("url") or "",
                    "description": source.get("description") or "",
                    "rerank_reason": source.get("rerank_reason") or "",
                    "reference_role": _reference_role_from_reason(
                        source.get("rerank_reason") or "",
                        q,
                        user_intent,
                    ),
                    "original_rank": source.get("rank"),
                }
                if record.get("normalized_from"):
                    item["normalized_from"] = record.get("normalized_from")
                if record.get("aspect_ratio") is not None:
                    item["aspect_ratio"] = record.get("aspect_ratio")
                if record.get("normalized_size"):
                    item["normalized_size"] = record.get("normalized_size")
                if score is not None:
                    try:
                        item["rerank_score"] = float(score)
                    except (TypeError, ValueError):
                        pass
                query_image_results.append(item)
            all_image_results.extend(query_image_results)
            query_candidate_results: List[Dict[str, Any]] = []
            for candidate in candidate_imgs:
                if not isinstance(candidate, dict):
                    continue
                path = candidate.get("local_path")
                if not path or not os.path.isfile(str(path)):
                    continue
                record = _normalize_reference_for_return(str(path))
                normalized_path = record["path"]
                score = candidate.get("rerank_score")
                item = {
                    "query": q,
                    "rank": candidate.get("rank"),
                    "role": candidate.get("role") or "candidate",
                    "path": normalized_path,
                    "original_path": record.get("original_path") or os.path.abspath(str(path)),
                    "normalization": record.get("normalization"),
                    "url": candidate.get("url") or "",
                    "description": candidate.get("description") or "",
                    "rerank_reason": candidate.get("rerank_reason") or "",
                    "reference_role": _reference_role_from_reason(
                        candidate.get("rerank_reason") or "",
                        q,
                        user_intent,
                    ),
                }
                if record.get("normalized_from"):
                    item["normalized_from"] = record.get("normalized_from")
                if record.get("aspect_ratio") is not None:
                    item["aspect_ratio"] = record.get("aspect_ratio")
                if record.get("normalized_size"):
                    item["normalized_size"] = record.get("normalized_size")
                if score is not None:
                    try:
                        item["rerank_score"] = float(score)
                    except (TypeError, ValueError):
                        pass
                query_candidate_results.append(item)
            all_candidate_results.extend(query_candidate_results)
            image_trace.append({
                "query": q,
                "returned": len(imgs),
                "downloaded": len(paths),
                "curated": len(query_curated),
                "candidate_count": len(query_candidate_results),
                "downloaded_paths": paths,
                "curated_paths": query_curated,
                "image_results": query_image_results,
                "candidate_image_results": query_candidate_results,
                "ranking_log": out.get("ranking_log"),
            })

        facts = "\n\n".join(fact_blocks).strip()

        # ---- Archive full details; only put a preview into what goes to the LLM ----
        try:
            Path(out_dir, "facts.txt").write_text(facts, encoding="utf-8")
            Path(out_dir, "meta.json").write_text(json.dumps({
                "text_queries": text_qs,
                "image_queries": image_qs,
                "web_trace": web_trace,
                "image_trace": image_trace,
                "downloaded_count": len(all_image_paths),
                "curated_count": len(curated),
                "candidate_count": len(all_candidate_results),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            record_artifacts_in_dir(out_dir, tool_name="search")
            for item in all_candidate_results:
                record_artifact(
                    item["path"],
                    tool_name="search",
                    role="rerank_candidate",
                    kind="image",
                    label="Rerank candidate image",
                    out_dir=out_dir,
                    metadata={
                        "query": item.get("query"),
                        "rank": item.get("rank"),
                        "role": item.get("role"),
                        "reference_role": item.get("reference_role"),
                        "rerank_score": item.get("rerank_score"),
                        "rerank_reason": item.get("rerank_reason"),
                        "url": item.get("url"),
                        "description": item.get("description"),
                        "original_path": item.get("original_path"),
                        "normalized_from": item.get("normalized_from"),
                        "normalization": item.get("normalization"),
                        "aspect_ratio": item.get("aspect_ratio"),
                        "normalized_size": item.get("normalized_size"),
                    },
                )
            for item in all_image_results:
                record_artifact(
                    item["path"],
                    tool_name="search",
                    role="reference",
                    kind="image",
                    label="Primary reference image" if item.get("role") == "primary" else "Backup reference image",
                    out_dir=out_dir,
                    metadata={
                        "query": item.get("query"),
                        "rank": item.get("rank"),
                        "role": item.get("role"),
                        "reference_role": item.get("reference_role"),
                        "is_primary": item.get("role") == "primary",
                        "rerank_score": item.get("rerank_score"),
                        "rerank_reason": item.get("rerank_reason"),
                        "url": item.get("url"),
                        "description": item.get("description"),
                        "original_path": item.get("original_path"),
                        "normalized_from": item.get("normalized_from"),
                        "normalization": item.get("normalization"),
                        "aspect_ratio": item.get("aspect_ratio"),
                        "normalized_size": item.get("normalized_size"),
                    },
                )
        except Exception:  # noqa: BLE001
            pass

        ranking_log = "\n".join(
            f"{trace.get('query')}: {trace.get('ranking_log')}"
            for trace in image_trace
            if trace.get("ranking_log")
        )
        payload: Dict[str, Any] = {
            "reference_paths": curated,
            "image_results": all_image_results,
            "candidate_image_results": all_candidate_results,
            "text_queries": text_qs,
            "image_queries": image_qs,
            "enriched_prompt": user_intent,  # not rewritten; downstream merges facts itself
            "facts": facts[:1800],
            "ranking_log": ranking_log,
            "gate_action": "faithful",
            "downloaded_count": len(all_image_paths),
            "candidate_count": len(all_candidate_results),
            "out_dir": out_dir,
            "status": "success" if (curated or facts) else "failed",
        }
        if payload["status"] != "success":
            payload["error"] = "search returned no curated refs and no facts"

        body = json.dumps(payload, ensure_ascii=False)
        return self.ok(body) if payload["status"] == "success" else self.err(body)
