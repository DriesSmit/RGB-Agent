# Local Server

This folder contains the repo-local OpenAI-compatible server used by `make server`.

Default behavior:
- `make server` starts `Qwen/Qwen3.5-0.8B`
- The server listens on `0.0.0.0:1234`
- The analyzer connects through `http://host.docker.internal:1234/v1`
- The first startup downloads model weights from Hugging Face, so it can take a few minutes.

Useful overrides:

```bash
make server LOCAL_SERVER_MODEL=Qwen/Qwen2.5-0.5B-Instruct
make server LOCAL_SERVER_DEVICE=cpu
make run MODEL=opus
```

Logs:

```bash
tail -f /tmp/rgb-agent-local-server.log
```
