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
make install
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
make run
make run MODEL=opus
rgb-swarm --suite all --max-actions 500
rgb-swarm --game ls20,ft09
rgb-swarm --env-source re_arc --game memory-0001 --max-actions 200
rgb-swarm --env-source re_arc --game memory-0001 --analyzer-model opus
```

For the local-model path, `make server` now launches the repo's built-in local Qwen server by default, and `make run` depends on it:

```bash
make server
make run
```

The first `make server` run downloads the Qwen weights, so expect startup to take a few minutes before `/v1/models` becomes healthy. Server logs go to `/tmp/rgb-agent-local-server.log`.

Useful overrides:

```bash
make server LOCAL_SERVER_MODEL=Qwen/Qwen3.5-0.8B
make server LOCAL_SERVER_DEVICE=cpu
make run MODEL=opus
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--suite` | — | Predefined game suites (e.g. `ls20`, `vc33`, `ft09`, or `all`) |
| `--game` | — | Comma-separated game names or IDs (alternative to `--suite`) |
| `--max-actions` | 500 | Max actions per game |
| `--env-source` | `arc_agi` | Environment backend: `arc_agi` or `re_arc` |
| `--interval`, `-n`, `--analyzer-interval` | 10 | Actions per analyzer batch plan |
| `--model`, `-m`, `--analyzer-model` | `local-qwen` | Analyzer model or alias (see below) |
| `--operation-mode` | `online` | `online` / `offline` / `normal` |

### Models

The default analyzer preset is `local-qwen`, which expects a local OpenAI-compatible endpoint reachable from the Docker sandbox. Anthropic models can still be passed without a prefix. For other providers, use `provider/model`.

| Model | `--model` value |
|-------|-----------------|
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

The analyzer agent ([OpenCode](https://github.com/opencode-ai/opencode)) runs in a sandboxed Docker container, reads the game's prompt log with Read, Grep, and Python, and outputs a JSON action plan. The action queue drains these one per step with zero LLM calls. When the queue empties or the score changes, the analyzer re-fires.

```
rgb_agent/
├── agent/              
│   ├── opencode_agent.py # Runs OpenCode in Docker to produce action plans
│   ├── action_queue.py # Drains one action per step (to support batched action plans + score-change flush)
│   ├── game_state.py   # Formatting
│   └── prompts.py      
├── environment/        
│   ├── arcagi3.py      # ARC-AGI-3 API wrapper (reset, step, scoring)
│   ├── runner.py       # Per-game orchestration loop
│   ├── swarm.py        # Runs multiple games in parallel on a scorecard
│   └── config.py     
├── metrics/            
└── utils/            
```
