# Kiyosi Deep Research v1

Multi-agent research pipeline built with LangGraph + LangChain. It takes a prompt, runs iterative research, and generates a citation-backed report.

## What this project does

- Converts a user prompt into a structured research brief and report plan
- Runs a supervisor + researcher agent loop for evidence gathering
- Produces a final markdown report
- Optionally exports source metadata and a research trace

Core entrypoint: `run_research.py`

## Requirements

- Python 3.11+ recommended
- One LLM provider API key (Gemini/OpenAI/Anthropic/Cerebras)
- Tavily API key for web search (recommended for normal operation)

Dependencies are listed in `requirements.txt`.

## 1) Setup

### Option A: `pip` + `requirements.txt` (standard)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option B: `uv` (optional)

```powershell
uv venv
.\venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

## 2) Configure environment variables

Create a `.env` file at the project root.

`run_research.py` loads this exact file (`./.env`), and `.env` is already gitignored in this repo, so your secrets will not be committed.

### Required keys for current default config

Current `deep_research/config.py` defaults to Gemini fallback models, so you need:

- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) for model calls
- `TAVILY_API_KEY` for core web-search tool use

Use this template and replace placeholder values with your real keys:

```dotenv
GOOGLE_API_KEY=<YOUR_GOOGLE_OR_GEMINI_API_KEY>
TAVILY_API_KEY=<YOUR_TAVILY_API_KEY>
```

If you prefer the Gemini alias, this also works:

```dotenv
GEMINI_API_KEY=<YOUR_GOOGLE_OR_GEMINI_API_KEY>
TAVILY_API_KEY=<YOUR_TAVILY_API_KEY>
```

### If you change providers in `deep_research/config.py`

Set the provider key that matches your chosen models:

```dotenv
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
# or
ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
# or
CEREBRAS_API_KEY=<YOUR_CEREBRAS_API_KEY>
```

Tooling note: `PERPLEXITY_KEY` is required for Substack/Perplexity-based tool flows. If you want full tool coverage across web + Substack discovery, set both `TAVILY_API_KEY` and `PERPLEXITY_KEY`.

Optional (needed for Substack discovery tools):

```dotenv
PERPLEXITY_KEY=<YOUR_PERPLEXITY_API_KEY>
```

## 3) Run the project

### Prompt as CLI arg

```powershell
python run_research.py --prompt "What are the latest developments in AI safety?"
```

### Prompt from file

```powershell
python run_research.py --prompt-file input.txt
```

### Custom output directory

```powershell
python run_research.py --prompt-file input.txt --output-dir outputs
```

## Output files

By default (`OUTPUT_MODE = "file"` in `deep_research/config.py`), the run writes:

- `research_<timestamp>.md` (main report)
- `research_data_<timestamp>.json` (sources metadata, if collected)
- `trace_<timestamp>.md` (compressed research process trace, when available)
- `error_<timestamp>.txt` (only when a run fails)

## Config knobs

Main runtime config lives in `deep_research/config.py`:

- `DEFAULT_MODEL` and `MODEL_FALLBACK_CHAIN`
- `RESEARCH_TIME_MIN_MINUTES` / `RESEARCH_TIME_MAX_MINUTES`
- `OUTPUT_MODE` (`file`, `db`, `both`)
- `SAVE_REPORT_TO_FILE`
- `ENABLE_SUBTOPIC_GENERATION`

Recommended model setting for this repo:

- `DEFAULT_MODEL = "gemini-3-flash-preview"` for strong cost efficiency, speed, and depth-per-minute.

## Troubleshooting

### 1) Import errors for `deep_research`

Run commands from repository root so `run_research.py` can resolve imports correctly.

### 2) LLM auth/provider errors

Check:

- your API key exists in `.env`
- `DEFAULT_MODEL` in `deep_research/config.py` matches the provider key you supplied

### 3) Long stall at draft-generation stage

This repo includes a fix that makes draft generation async + timeout based, and avoids structured output wrapping for long drafts.

## Quick architecture

Note: `clarify_with_user` is currently not enabled for real clarification behavior and acts as a dummy pass-through call to `write_research_brief`.

Pipeline order:

1. `clarify_with_user`
2. `write_research_brief`
3. `plan_report`
4. `write_draft_report`
5. `supervisor_subgraph`
6. `final_report_generation`
7. optional: `subtopic_evaluation` -> `subtopic_generation`
