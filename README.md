# tau2-purple

Purple agent for the [tau2-bench](https://github.com/RDI-Foundation/tau2-agentbeats) benchmark on [AgentBeats](https://agentbeats.dev).

## How it works

This agent acts as a customer service AI that:
- Receives domain policies and available tools from the green agent
- Detects the active domain (airline, retail, or telecom) and applies domain-specific reasoning
- Uses an LLM via any OpenAI-compatible API to decide on tool calls or user replies
- Returns decisions as a single raw JSON object per turn
- Maintains per-context conversation history with automatic window trimming

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_API_KEY` | Yes | — | API key for the LLM provider |
| `LLM_BASE_URL` | No | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL |
| `AGENT_MODEL` | No | `qwen/qwen3.6-plus:free` | Model identifier passed to the API |
| `LOG_LEVEL` | No | `INFO` | Python logging level |

Copy `.env` and fill in your key:

```bash
cp .env .env.local
# edit .env.local, then:
export $(cat .env.local | xargs)
```

## Local development

```bash
uv sync
uv run src/server.py --host 127.0.0.1 --port 9009
```

## Docker

```bash
docker build -t tau2-purple .
docker run -p 9009:9009 --env-file .env tau2-purple
```
