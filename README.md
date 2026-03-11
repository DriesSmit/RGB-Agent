# Read-Grep-Bash Agent

An agent for [ARC-AGI-3](https://three.arcprize.org/) that completes all three preview games in 1,069 actions, the lowest publicly reported count.

For details on approach and findings, see our [blog post](https://blog.alexisfox.dev/arcagi3).

![Architecture](assets/architecture.png)

## Setup

Requires Python (3.12 recommended) and Docker.

```bash
git clone git@github.com:alexisfox7/RGB-Agent.git
cd RGB-Agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
cd docker/opencode-sandbox && bash build.sh   # build analyzer sandbox image
```

Create a `.env` file:

```
ARC_API_KEY=...
ANTHROPIC_API_KEY=...
LOCAL_ANALYZER_BASE_URL=http://host.docker.internal:1234/v1
LOCAL_ANALYZER_MODEL_ID=qwen3.5-0.8b
```

## Usage

```bash
arcgym-swarm --suite all --max-actions 500
arcgym-swarm --game ls20,ft09
arcgym-swarm --env-source re_arc --game taps --max-actions 200
arcgym-swarm --env-source re_arc --game memory-0001 --analyzer-model opus
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--suite` | — | Predefined game suites (e.g. `ls20`, `vc33`, `ft09`, or `all`) |
| `--game` | — | Comma-separated game names or IDs (alternative to `--suite`) |
| `--max-actions` | 500 | Max actions per game |
| `--env-source` | `arc_agi` | Environment backend: `arc_agi` (online/offline ARC API) or `re_arc` (local games like `taps`) |
| `--analyzer-interval` | 10 | Actions per analyzer batch plan |
| `--analyzer-model` | `local-qwen` | Analyzer model or alias (see below) |
| `--operation-mode` | `online` | `online` / `offline` / `normal` |

### Analyzer models

The default analyzer preset is `local-qwen`, which expects a local OpenAI-compatible endpoint reachable from the Docker sandbox. Anthropic models can still be passed without a prefix. For other providers, use `provider/model`.

| Model | `--analyzer-model` value |
|-------|--------------------------|
| Local Qwen 3.5 0.8B | `local-qwen` (default) |
| Claude Opus 4.6 | `opus` or `claude-opus-4-6` |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` |
| GPT 5.2 | `openai/gpt-5.2` |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` |

Any model available via OpenRouter can also be used with the `openrouter/` prefix (e.g. `openrouter/google/gemini-3.1-pro-preview`).

Set the matching API key in `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `OPENROUTER_API_KEY`). For the local preset, set `LOCAL_ANALYZER_BASE_URL` and `LOCAL_ANALYZER_MODEL_ID`; if your local server requires auth, also set `LOCAL_ANALYZER_API_KEY`.

The analyzer runs inside Docker. On Linux, your local model server usually needs to listen on a non-loopback interface so `host.docker.internal` can reach it from the sandbox.

Results are saved to `evaluation_results/`.

## Architecture

Every N actions, the planner (a coding agent ([OpenCode](https://github.com/opencode-ai/opencode))) running in a sandboxed Docker container reads the agent's prompt log with Read, Grep, and Python, then outputs a JSON action plan. The agent queues these actions and drains them one per step with zero LLM calls. When the queue empties or the score changes, the planner re-fires.

```
arcgym/agents/
├── rgb_agent.py   # Agent + action queue
├── planner.py     # OpenCode-in-Docker planner
├── prompts.py     # Planner prompt templates
arcgym/evaluation/
├── swarm.py       # CLI entry point
├── runner.py      # Per-game episode loop
```
