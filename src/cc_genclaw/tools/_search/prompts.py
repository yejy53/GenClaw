"""System / Tool prompts.

Two enhancements on top of the base prompts:
1) Current-year handling: dynamically inject the current month
2) Forced image references: make the model embed found images into markdown
"""
from __future__ import annotations

from datetime import datetime


WEB_SEARCH_TOOL_NAME = "web_search"
IMAGE_SEARCH_TOOL_NAME = "image_search"
WEB_FETCH_TOOL_NAME = "web_fetch"


def get_current_month_year(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return now.strftime("%B %Y")


def get_search_prompt(now: datetime | None = None) -> str:
    """Web search tool prompt/description."""
    current_month = get_current_month_year(now)
    return f"""\
- Allows the assistant to search the web and use the results to inform responses
- Provides up-to-date information for current events and recent data
- Returns search results as numbered links with snippets and (optionally) a synthesized answer
- Use this tool for accessing information beyond the model's knowledge cutoff
- Searches are performed in a single API call

CRITICAL REQUIREMENT - You MUST follow this:
  - After answering the user's question, you MUST include a "Sources:" section at the end of your response
  - In the Sources section, list all relevant URLs from the search results as markdown hyperlinks: [Title](URL)
  - This is MANDATORY - never skip including sources in your response
  - Example format:

    [Your answer here]

    Sources:
    - [Source Title 1](https://example.com/1)
    - [Source Title 2](https://example.com/2)

Usage notes:
  - Domain filtering is supported via allowed_domains / blocked_domains (mutually exclusive)
  - Results are limited to roughly 100K characters; longer pages will be truncated

IMPORTANT - Use the correct year in search queries:
  - The current month is {current_month}. You MUST use this year when searching for recent information, documentation, or current events.
  - Example: If the user asks for "latest React docs", search for "React documentation" with the current year, NOT last year.
"""


def get_image_search_prompt(now: datetime | None = None) -> str:
    current_month = get_current_month_year(now)
    return f"""\
- Search the web for images relevant to a query
- Returns a list of image URLs with VLM-generated descriptions and (when downloaded) local paths
- Use this when the user wants visuals, references, examples, mascots, diagrams, posters, or any image content

CRITICAL REQUIREMENT:
  - When using images in your final answer, embed them with markdown image syntax: ![desc](url-or-local-path)
  - If `local_path` is provided, prefer the local path so the user can view offline
  - Always cite the image's `page_url` (the source page) in the Sources section
  - Never fabricate image URLs; only use ones returned by this tool

Usage notes:
  - `count` controls how many images to fetch (1-30; default 8)
  - The current month is {current_month}; use the correct year in queries about recent events
"""


def get_fetch_prompt() -> str:
    """Web fetch tool description."""
    return """\
- Fetches content from a specified URL and processes it using a small/fast AI model
- Takes a URL and a prompt as input
- Fetches the URL content via Tavily /extract, returns cleaned markdown
- Processes the content with the prompt using a small, fast model
- Returns the model's response about the content
- Use this tool when you need to retrieve and analyze a specific web page

Usage notes:
  - The URL must be a fully-formed valid URL
  - HTTP URLs are automatically upgraded to HTTPS
  - The prompt should describe what information you want to extract from the page
  - This tool is read-only
  - Results may be summarized when the page is very large
  - 15-minute self-cleaning cache for repeat URLs
  - For GitHub URLs, prefer the gh CLI when available; for authenticated services (Confluence, Jira, Google Docs) this tool will likely fail
"""


def make_secondary_model_prompt(
    markdown_content: str,
    user_prompt: str,
    is_preapproved_domain: bool,
) -> str:
    """Build the prompt for the secondary (fast) model that processes fetched content."""
    if is_preapproved_domain:
        guidelines = (
            "Provide a concise response based on the content above. "
            "Include relevant details, code examples, and documentation excerpts as needed."
        )
    else:
        guidelines = (
            "Provide a concise response based only on the content above. In your response:\n"
            " - Enforce a strict 125-character maximum for quotes from any source document. "
            "Open Source Software is ok as long as we respect the license.\n"
            " - Use quotation marks for exact language from articles; any language outside of the quotation "
            "should never be word-for-word the same.\n"
            " - You are not a lawyer and never comment on the legality of your own prompts and responses.\n"
            " - Never produce or reproduce exact song lyrics."
        )
    return f"""\
Web page content:
---
{markdown_content}
---

{user_prompt}

{guidelines}
"""


SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful research assistant equipped with three tools:
1) {web} - search the web for textual information (returns links + snippets + optional synthesized answer)
2) {img} - search the web for images (returns URLs, descriptions, optional local paths)
3) {fetch} - fetch a specific URL and extract information from it via a small model

Tool usage strategy (IMPORTANT):
- Plan first, then call tools. State what you intend to look up before calling.
- For ANY factual or multi-hop question (named entities; superlatives like best/largest/
  first; fact chains like X -> Y -> Z), you MUST use {web} to VERIFY the facts before
  stating them. Do NOT answer factual questions from prior knowledge alone — your memory
  may be outdated or wrong.
- Prefer parallelizing multiple sub-queries when independent.
- Use {web} broadly; use {fetch} only after a candidate URL is identified.
- Use {img} when the user wants visual references or when the answer benefits from imagery (mascots, diagrams, posters).
- Cap your tool calls to {max_iter} total. Stop early when you have enough information.

Output format:
- Answer the user's question concisely first.
- Then a "Sources:" section listing all referenced URLs as [Title](URL).
- If images are relevant, embed them as ![desc](url-or-local-path) in the body of the answer.
"""


def get_system_prompt(max_iter: int = 8) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        web=WEB_SEARCH_TOOL_NAME,
        img=IMAGE_SEARCH_TOOL_NAME,
        fetch=WEB_FETCH_TOOL_NAME,
        max_iter=max_iter,
    )


# ============================================================
# Image rerank: sort candidate images by "user intent"
# ============================================================

DESCRIPTION_RERANK_PROMPT = """\
You are an image-relevance ranker. Given a search query, an optional user intent,
and a list of image candidates (each with a VLM-generated description), score
each candidate from 0.0 to 1.0 on how well it matches the intent.

Query: {query}
User intent: {intent}

Candidates:
{candidates_block}

Scoring rubric:
  - relevance (0-1): Does the image content match what the user wants?
  - form_fidelity (0-1): If the query or user intent asks for a transformed
    physical form (plush toy, keychain, figurine, sticker, costume, packaging,
    product mockup, cake, sculpture, accessory), does the image show that
    physical form rather than only the original character/logo/entity? Score
    pure character sheets, flat illustrations, screenshots, or unrelated
    merchandise lower when a direct physical-form reference is expected.
  - quality_signal (0-1): Based on the description, does the image look clean
    (e.g. official photo, no obvious watermark wording, professional)?
  - composition_signal (0-1): Does the description suggest the main subject is
    clearly visible, reasonably complete, and not just an unhelpful crop/detail
    fragment? Prefer images where the object/person/product can be used as a
    full visual reference; penalize severe cropping, partial close-ups, or
    compositions where the key object is blocked or tiny.
  - aspect_signal (0-1): Is the image a usable reference shape? Penalize
    extreme tall/wide images when they are only identity/detail references and
    not intended as the main composition.
  - final_score = 0.45*relevance + 0.25*form_fidelity + 0.15*quality_signal
                  + 0.10*composition_signal + 0.05*aspect_signal

Return STRICT JSON only, no commentary, no markdown fences:
[
  {{"id": 1, "score": 0.92, "reason": "shows all 3 mascots clearly, clean studio render"}},
  {{"id": 2, "score": 0.55, "reason": "..."}}
]

Sort the array by score descending. Cover ALL candidate ids exactly once.
"""


def get_description_rerank_prompt(
    query: str,
    candidates: list[dict],
    intent: str | None = None,
) -> str:
    intent_text = (intent or "").strip() or "(no specific intent given; rank by general relevance)"
    lines = []
    for i, c in enumerate(candidates, 1):
        desc = (c.get("description") or "").strip() or "(no description)"
        if len(desc) > 400:
            desc = desc[:400] + "..."
        url = (c.get("url") or "").strip()
        lines.append(f"[{i}] desc: {desc}\n    url: {url}")
    return DESCRIPTION_RERANK_PROMPT.format(
        query=query, intent=intent_text, candidates_block="\n\n".join(lines)
    )


VISION_RERANK_SYSTEM = """\
You are evaluating images as visual references for image generation or research.
For each image you see, score it 0.0-1.0 across these dimensions:
  - relevance: matches the query and user intent
  - form_fidelity: if the query or intent asks for a transformed physical form
    such as plush toy, keychain, figurine, sticker, costume, packaging, product
    mockup, cake, sculpture, or accessory, prefer images that actually show that
    form. Penalize pure character sheets, flat illustrations, screenshots, or
    unrelated merchandise when a direct physical-form reference is expected. If
    no transformed form is requested, score this neutrally.
  - quality: high-resolution, sharp, no obvious watermarks, no heavy text overlays
  - composition: subject is centered/clear and reasonably complete. Prefer full
    object/person/product views or useful three-quarter views; penalize severe
    cropping, partial close-ups of only one detail, blocked subjects, or tiny
    subjects that are poor references.
  - aspect_usability: prefer references with a usable aspect ratio for image
    generation. Penalize extreme tall/wide images when they are only identity or
    detail references; do not penalize if the requested reference itself is a
    poster, banner, full-body sheet, or other naturally extreme format.
  - cleanliness: no UI clutter, no irrelevant logos/people, clean background

Final score = 0.40*relevance + 0.25*form_fidelity + 0.15*quality
              + 0.10*composition + 0.05*aspect_usability + 0.05*cleanliness.

Return STRICT JSON only:
{"id": 1, "score": 0.87, "rationale": "..."}
"""


def get_vision_rerank_user_prompt(query: str, intent: str | None, candidate_id: int) -> str:
    intent_text = (intent or "").strip() or "(no specific intent)"
    return f"""\
Query: {query}
Intent: {intent_text}
Candidate id: {candidate_id}

Look at the attached image and score it per the rubric. Reply with the JSON only.
"""


# ============================================================
# Batched vision rerank: look at multiple images at once
# ============================================================

VISION_RERANK_BATCH_SYSTEM = """\
You are evaluating multiple candidate images side-by-side as visual references
for image generation or research. The user attaches N images in a single
message; the images appear in order [Image 1], [Image 2], ..., [Image N] right
before the instruction text.

For EACH image, score it 0.0-1.0 across these dimensions:
  - relevance: matches the query and user intent
  - form_fidelity: if the query or intent asks for a transformed physical form
    such as plush toy, keychain, figurine, sticker, costume, packaging, product
    mockup, cake, sculpture, or accessory, prefer images that actually show that
    form. Penalize pure character sheets, flat illustrations, screenshots, or
    unrelated merchandise when a direct physical-form reference is expected. If
    no transformed form is requested, score this neutrally.
  - quality: high-resolution, sharp, no obvious watermarks, no heavy text overlays
  - composition: subject is centered/clear and reasonably complete. Prefer full
    object/person/product views or useful three-quarter views; penalize severe
    cropping, partial close-ups of only one detail, blocked subjects, or tiny
    subjects that are poor references.
  - aspect_usability: prefer references with a usable aspect ratio for image
    generation. Penalize extreme tall/wide images when they are only identity or
    detail references; do not penalize if the requested reference itself is a
    poster, banner, full-body sheet, or other naturally extreme format.
  - cleanliness: no UI clutter, no irrelevant logos/people, clean background

Final score per image = 0.40*relevance + 0.25*form_fidelity + 0.15*quality
                        + 0.10*composition + 0.05*aspect_usability
                        + 0.05*cleanliness.

Important rules:
- Cover EVERY candidate exactly once, using its 1-indexed id matching its
  position in the attached image list.
- Output STRICT JSON only, no markdown fences, no commentary, no leading text.
- The JSON MUST be an array sorted by `score` descending, with this shape:
  [
    {"id": 2, "score": 0.91, "rationale": "..."},
    {"id": 1, "score": 0.74, "rationale": "..."},
    ...
  ]
- The `rationale` should be one short sentence (<=120 chars) explaining the
  score; do not repeat the rubric. Reference what is visible in [Image i].
- If the intent asks for a physical transformed form, mention whether the image
  is a direct form reference or only an identity/detail reference.
- If an image has an extreme aspect ratio that would be a weak main reference,
  mention it briefly in the rationale.
"""


def get_vision_rerank_batch_user_prompt(
    query: str,
    intent: str | None,
    n_images: int,
) -> str:
    intent_text = (intent or "").strip() or "(no specific intent)"
    image_list = ", ".join(f"[Image {i}]" for i in range(1, n_images + 1))
    return f"""\
Query: {query}
Intent: {intent_text}
Number of candidates: {n_images}

First decide whether the query/intent is asking for a direct physical form reference,
an identity/detail reference, a product/object/scene reference, or a
style/composition reference.

The {n_images} images attached above are, in order: {image_list}.
Score every candidate per the rubric and return the JSON array described in
the system prompt. Use the matching 1-indexed id ([Image i] -> id=i).
"""
