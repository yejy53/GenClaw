# GenClaw: Code-Driven Agentic Image Generation

[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow)](https://huggingface.co/papers/2605.30248)
[![arXiv](https://img.shields.io/badge/arXiv-2605.30248-b31b1b)](https://arxiv.org/abs/2605.30248)
[![GitHub stars](https://img.shields.io/github/stars/yejy53/GenClaw?style=social)](https://github.com/yejy53/GenClaw)

GenClaw explores **code-driven agentic image generation**: instead of only rewriting prompts, an image generation agent uses code as a controllable visual canvas before calling image generation models for final rendering.

The core idea is simple: **think, sketch with code, then render**.

https://github.com/user-attachments/assets/786007a9-6a2e-427d-88fa-506d608e925e

## News

- **[2026-05-30]** 🎉 GenClaw released — technical report and project page.
- **[2026-06-29]** 🚀 Released a runnable agent implementation, with updated code for **complex scene composition**, **text rendering**, and **world-knowledge-grounded generation**.

## Highlights

🎨 **Code as a Visual Brush.** The agent creates by writing executable visual sketches—SVG, HTML/CSS, Python, lightweight 3D code—turning object count, spatial layout, and text rendering into executable, verifiable, debuggable programs. Image synthesis shifts from implicit diffusion sampling to an explicit, reasoning-friendly process.

✋ **Draw as a Human Artist.** We mirror the human creative loop—conceptualize → sketch → coloring → refine—and make every stage transparent: ideation, reference retrieval, drafting, and incremental rendering are all surfaced as inspectable, editable, revertible artifacts. Generation becomes an iterative collaboration rather than one-shot black-box inference.

🔌 **Agent Harness for Image Generation.** We plug an LLM agent's proven planning, tool-use, and reflection abilities directly into image synthesis, exploring an agent harness for image generation—so that creating images becomes a first-class capability inside the agent's toolbox, not an isolated standalone model.

## Showcase

<p align="center">
  <img src="assets/showcases.jpg" width="95%">
</p>

## Visual Examples

### Complex Scene Composition

<p align="center">
  <img src="assets/complex_scene.jpg" width="95%">
</p>

### Text Rendering and Poster Design

<p align="center">
  <img src="assets/text_rendering.jpg" width="95%">
</p>

### Physical Reasoning

<p align="center">
  <img src="assets/physical_reasoning.jpg" width="95%">
</p>

### Knowledge-Grounded Generation

<p align="center">
  <img src="assets/mind_bench.jpg" width="95%">
</p>

## Getting Started

A runnable agent implementation lives in this repo. A planner LLM works in a
tool-use loop: before each task a `perception` sub-LLM writes descriptive
"painter notes" (intent + planning hints — these are hints, not routing
decisions), then the planner reads the request, the notes, and the tool cards,
opens a `todo_write` plan, and composes a tool pipeline from a set of canonical
chains.

**Tools**: `t2i` / `i2i` (text/image-to-image), `code_scene_draft` (SVG layout
draft), `code_text_draft` (verbatim long-text rendering), `search` (Tavily web/
image search), `reason` (multimodal reasoning), `format_prompt`,
`vlm_review`, plus `todo_write` / `tool_search` planning helpers.

**Example canonical chains** (from `todo_write`'s card; `format_prompt` can be
inserted as needed): spatial/counting `code_scene_draft → i2i`; grounded entity
`search → format_prompt → i2i`; scientific derivation `reason → t2i`;
world-knowledge long text `search → code_text_draft`.

### 1. Install

```bash
git clone https://github.com/yejy53/GenClaw.git
cd GenClaw
```

Using `venv` + pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Or using conda:

```bash
conda env create -f environment.yml
conda activate cc-genclaw
```

Then install the headless browser used by the code-draft renderers:

```bash
playwright install chromium
# On Linux you may also need: playwright install-deps
```

### 2. Configure

```bash
cp config.example.yaml config.yaml
```

One OpenAI-compatible gateway serves every model: set the endpoint once, then
you only choose a model name per role. Edit `config.yaml` (it is gitignored —
keep real keys out of version control):

| Setting | What it is |
|---|---|
| `OPENAI_BASE_URL` + `OPENAI_API_KEY` | the single gateway (set once) |
| `OPENAI_MODEL_NAME` | main planner + all sub-LLMs |
| `GEMINI_I2I_MODEL_NAME` | image generation/editing model |
| `TAVILY_API_KEY` | the `search` tool (world-knowledge tasks only) |

The image backend reuses `OPENAI_BASE_URL` / `OPENAI_API_KEY` automatically, so
you don't repeat the URL/key — just name the models. To put the image model on a
separate endpoint, set `GEMINI_BASE_URL` / `GEMINI_API_KEY`.

### 3. Run

Web UI (recommended):

```bash
chainlit run src/cc_genclaw/ui/chainlit_app.py -w
```

From Python:

```python
from cc_genclaw.runtime.agent_runner import (
    prepare_agent_run, initialize_perception, run_prompt,
)

run = prepare_agent_run()  # loads config.yaml, builds tools + a fresh session
prompt = "Draw 5 red balloons and 3 blue balloons over grass, photorealistic."
initialize_perception(prompt, user_image_path=None, session=run.session)
run_prompt(prompt, run)
print("artifacts in:", run.session.dir)  # final PNG + trace.jsonl live here
```

Each run writes to `runs/YYYYMMDD-HHMMSS-XXXXXX/` — a `trace.jsonl` plus each
tool's intermediate artifacts (SVG drafts, search references, final PNG).

### Repository layout

```
src/cc_genclaw/
├── agent.py / loop.py        # agent main loop
├── config.py / llm.py        # config loading + OpenAI-compatible client
├── perception/               # descriptive painter notes (not routing)
├── runtime/                  # session dirs, artifacts, agent runner
├── tools/                    # atomic tools + todo_write + search
│   ├── _image_gen/           # t2i / i2i + providers.yaml
│   ├── _code_scene_draft/    # SVG layout-draft pipeline
│   ├── _code_text_draft/     # verbatim text rendering pipeline
│   ├── _reason/              # multimodal reasoning
│   └── _search/              # Tavily web/image search
├── prompts/                  # system.md + perception + tool cards
└── ui/                       # Chainlit app
```

## Links

- Paper: https://huggingface.co/papers/2605.30248
- arXiv: https://arxiv.org/abs/2605.30248
- Repository: https://github.com/yejy53/GenClaw

If you find this project interesting, please consider giving it a star and voting for the paper on Hugging Face.

## Citation

If you find GenClaw useful, please consider citing our technical report:

```bibtex
@article{ye2026genclaw,
  title={GenClaw: Code-Driven Agentic Image Generation},
  author={Ye, Junyan and others},
  journal={arXiv preprint arXiv:2605.30248},
  year={2026}
}
```

## License

[MIT](LICENSE)

