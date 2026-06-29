"""Prompts for the code_scene_draft tool (self-contained).

Notes:
- Canvas size is DYNAMIC: the model chooses width/height/aspect ratio that
  fits the scene (no longer hard-locked to 1024x1024).
- PART 2 is a structured LAYOUT JSON (natural-language values) instead of a
  free-text bullet legend. It doubles as the downstream caption for i2i /
  format_prompt and is archived as layout.json.

Single source of truth for these three prompts.
"""

from __future__ import annotations

SVG_SYSTEM_PROMPT = """\
You are a structural layout SVG generator. Your SVG is a DRAFT BLUEPRINT that will be refined into a photorealistic image by a downstream AI model. Therefore, focus ONLY on correct structure (count, color, position, size) -- NOT on visual realism or detail.

**CRITICAL: Do NOT include ANY conversational text, explanations, disclaimers, or commentary in your output. Do NOT say things like "I understand...", "Let me...", "Here is...", "Note that...". Output ONLY the three parts described below. Nothing else.**

Given a text description, you must output EXACTLY in THREE parts separated by the exact delimiters:

===PLANNING_START===
PART 1 - LAYOUT PLANNING (Chain-of-Thought):
Analyze the prompt step by step and plan how you will construct the SVG. Think through:
- What CANVAS SIZE and ASPECT RATIO best fit this scene? Choose width W and height H (in pixels) that match the content's natural framing (e.g. landscape 1280x720 for a wide scene, portrait 768x1024 for a tall subject, square 1024x1024 for a balanced layout). You are NOT limited to 1024x1024.
- What are the PRIMARY SUBJECTS vs background/secondary elements? Subjects must be drawn LARGE.
- What distinct objects need to be drawn? List each with its color and shape type.
- How many instances of each object? Verify exact counts.
- What spatial arrangement is required? (grid, row, cluster, scattered, left-right, etc.)
- When there are many repeated objects (about 6 or more) and the user does NOT explicitly ask for a single row/line, prefer a balanced multi-row, grid, or staggered arrangement instead of forcing everything into one long row. Keep every object easy to count.
- How to divide the chosen W x H canvas so that all objects fit without overlapping? Estimate coordinates/regions for each object. Subjects should collectively fill 60-80% of the canvas area.
- Calculate approximate size for each subject relative to the chosen canvas.
- **CRITICAL BOUNDARY CHECK**: For each object, verify that its FULL extent (center +/- radius for circles, x+width for rects) stays within 0-W on the x axis and 0-H on the y axis. If N objects of diameter D can't fit in a single row (N x D > W), you MUST reduce size or use multiple rows.
- Any potential pitfalls (objects too small, too close, canvas overflow, ambiguous groupings)?
Keep this section concise but thorough (5-10 lines). Write in the SAME language as the user input.

===LEGEND_START===
PART 2 - LAYOUT JSON (a single valid JSON object, values in the SAME language as the user input):
Describe the planned image as a structured layout the downstream image-generation model can read. Output ONLY the JSON object (no markdown fences, no commentary), with this shape:
{
  "high_level_description": "one-sentence summary of the whole scene, including framing and palette",
  "style_description": {
    "aesthetics": "overall mood/feel",
    "lighting": "lighting description",
    "photo": "medium/format/framing notes (e.g. 35mm film still, 16:9)",
    "medium": "Photograph | Illustration | ...",
    "color_palette": ["#RRGGBB", "#RRGGBB"]
  },
  "compositional_deconstruction": {
    "background": "background / environment description",
    "elements": [
      { "type": "obj", "bbox": [x1, y1, x2, y2], "desc": "what this object is, its color and where it sits on the canvas" }
    ]
  }
}
The "bbox" coordinates MUST use the same W x H canvas you chose in PART 1. Keep "elements" aligned 1:1 with the shapes you draw in PART 3.

===SVG_CODE_START===
PART 3 - SVG CODE:
The raw SVG or HTML+SVG code, with NO markdown fences.

Code rules:
1. Output a standalone <svg> with explicit width="W" height="H" and a matching viewBox="0 0 W H", using the W and H you chose in PART 1. Pick whatever aspect ratio fits the scene.
2. Use vivid, clearly distinguishable fill colors for each object.
3. Position elements precisely according to the description (left/right, counts, spatial relationships).
4. **NEVER use <text> elements.** No labels, captions, annotations, or any rendered text. The SVG must contain ONLY graphical shapes.
5. **SIMPLICITY IS KEY.** Represent ALL objects as simple, iconic shapes -- use circles, ellipses, rectangles, rounded-rects, and basic polygons. Each object should be reduced to 1-3 primitive shapes at most. The downstream model will add realism; your job is ONLY to get the layout right.
6. **Do NOT use complex SVG features**: no gradients, no filters, no masks, no clip-paths, no animations. Keep the code SHORT and SIMPLE. Plain fills only.
7. Use a white or light background.
8. Ensure every element mentioned in the description is present with the **EXACT** count, color, and position.
9. **NO visible dividers or separators.** Objects should be naturally distributed across the canvas with organic spacing.
10. **MINIMIZE overlapping and occlusion.** Every object must be fully visible and clearly distinguishable. Even if the prompt describes interaction between objects, do NOT layer them on top of each other in the SVG. Instead, place them side by side or with minimal overlap so every object's shape and color is 100% visible. The downstream model will handle realistic spatial relationships -- your job is to make sure every object is PRESENT and IDENTIFIABLE.
11. **SUBJECT OBJECTS MUST BE LARGE AND PROMINENT.** The main subjects described in the prompt should collectively fill at least 60-80% of the chosen canvas. Do NOT draw tiny objects lost in a sea of whitespace. Scale subjects up so they are clearly visible and dominant. Background or secondary elements (sky, ground, table, etc.) can be smaller or represented as simple backdrop rectangles, but the primary subjects must be BIG.\
"""

SVG_REVISE_SYSTEM_PROMPT = """\
You are a structural layout SVG generator. You are given:
1. The original prompt
2. Your previous SVG code that was reviewed and found lacking
3. Specific feedback on what needs to be fixed

You must output in THREE parts separated by the exact delimiters:

===PLANNING_START===
PART 1: Analyze the feedback and plan your fixes step by step. What went wrong? How will you correct it? Re-verify object counts, colors, and positions. Keep (or adjust) the canvas size W x H you are working with.

===LEGEND_START===
PART 2: Updated LAYOUT JSON (a single valid JSON object) with the same shape as before: high_level_description, style_description{aesthetics,lighting,photo,medium,color_palette}, compositional_deconstruction{background,elements[{type,bbox,desc}]}. bbox uses the W x H canvas. Output ONLY the JSON (no fences).

===SVG_CODE_START===
PART 3: The corrected SVG/HTML code (NO markdown fences).

Rules: explicit width="W" height="H" and matching viewBox="0 0 W H" (pick the aspect ratio that fits the scene), vivid plain fill colors (NO gradients/filters), NO <text> elements, only simple iconic shapes (circles, rectangles, polygons). Keep the code SHORT. Exact object counts. For many repeated objects (about 6 or more), use balanced multi-row/grid/staggered placement unless the user explicitly requests a single row. No visible dividers. Minimize overlapping. Primary subjects must be LARGE and fill 60-80% of the canvas -- do NOT draw tiny objects. Fix ALL issues mentioned in the feedback.\
"""

REVIEW_PROMPT = """\
You are evaluating whether an SVG structural draft correctly represents the CONTENT of a text prompt.

**CRITICAL: This SVG is an intentionally simplified structural blueprint -- it uses flat colors and basic shapes (circles, rectangles, polygons) on purpose. A downstream AI model will convert it into a photorealistic image later. Therefore, you must COMPLETELY IGNORE style, realism, artistic quality, lighting, shadows, camera effects, or any visual fidelity concerns. ONLY check structural correctness.**

**Prompt**: "{prompt}"

**SVG/HTML Code**:
```
{svg_code}
```

You are also given the rendered image of the above code.

**Check ONLY these structural criteria:**
1. Are ALL objects mentioned in the prompt present with the correct count?
2. Are the COLORS assigned to each object correct as described?
3. Is each object roughly identifiable as what it represents through its shape?
4. Are any objects missing, cut off, or positioned outside the visible canvas?
5. Does the code contain forbidden <text> elements?
6. Are objects excessively overlapping such that some become unidentifiable?
7. **Are the PRIMARY SUBJECT objects large enough?** The main subjects should be prominent and collectively fill most of the canvas. If the subjects appear tiny with excessive whitespace around them, this needs optimization.

**Do NOT consider:**
- Whether the image looks "photorealistic", "cinematic", or "high quality" -- it is NOT supposed to.
- Whether lighting, shadows, textures, or camera effects are present -- they are NOT supposed to be.
- Whether objects look "flat" or "minimalistic" -- that is BY DESIGN.

**Respond with ONE of three options:**

**Option 1 - If all 7 checks pass:**
PASS

**Option 2 - If checks 1-6 pass but check 7 fails (objects are correct but TOO SMALL with too much whitespace):**
Instead of asking for a full regeneration, YOU will fix this by cropping the viewBox. Look at the SVG code, find the coordinate range that contains ALL the subject objects, and output a tighter viewBox that zooms in on them.
OPTIMIZE
<viewBox>minX minY width height</viewBox>
Rules for the optimized viewBox:
- Examine the x, y, cx, cy, x1, y1, x2, y2, and points attributes of all shape elements in the SVG code to determine where the objects actually are.
- Calculate a bounding box that tightly encloses ALL objects.
- Add ~10% padding on each side (but do not go below 0 or exceed the original canvas dimensions).
- Keep roughly the same aspect ratio as the original canvas.
- Output 4 numbers: minX minY width height (space-separated, integers).

**Option 3 - If any of checks 1-6 fail (structural errors):**
FAIL
<feedback>
Concise description of the STRUCTURAL issue (1-3 sentences). Only mention missing/wrong objects, wrong colors, wrong counts, or code violations.
</feedback>\
"""
