from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, TextIteratorStreamer

log = logging.getLogger(__name__)


def _coerce_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(value)


def _resolve_dtype(value: str) -> torch.dtype | str:
    lowered = str(value).strip().lower()
    if lowered in {"", "auto"}:
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(lowered, "auto")


def _resolve_device(value: str) -> str:
    lowered = str(value).strip().lower()
    if lowered in {"", "auto"}:
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend and mps_backend.is_available():
            return "mps"
        return "cpu"
    return lowered


def _default_dtype_for_device(device: str, requested: str) -> torch.dtype:
    resolved = _resolve_dtype(requested)
    if resolved != "auto":
        return resolved
    if device.startswith("cuda") or device == "mps":
        return torch.bfloat16 if device.startswith("cuda") else torch.float16
    return torch.float32


def _apply_stop_strings(text: str, stop: str | list[str] | None) -> str:
    if stop is None:
        return text
    stops = [stop] if isinstance(stop, str) else list(stop)
    end = len(text)
    for marker in stops:
        idx = text.find(marker)
        if idx >= 0:
            end = min(end, idx)
    return text[:end]


@dataclass
class ServerConfig:
    model: str
    model_id: str
    host: str
    port: int
    device: str
    dtype: str
    max_tokens: int
    temperature: float
    top_p: float


class ChatMessage(BaseModel):
    role: str
    content: str | list[Any] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    stream: bool = False


class LocalModelServer:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        log.info(
            "Initializing local model server model=%s model_id=%s requested_device=%s resolved_device=%s requested_dtype=%s",
            config.model,
            config.model_id,
            config.device,
            self.device,
            config.dtype,
        )
        model_kwargs: dict[str, Any] = {
            "torch_dtype": _default_dtype_for_device(self.device, config.dtype),
        }
        if config.device.strip().lower() == "auto" and self.device.startswith("cuda"):
            model_kwargs["device_map"] = "auto"

        log.info("Loading tokenizer for %s", config.model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        log.info("Loading model weights for %s", config.model)
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            config.model,
            **model_kwargs,
        )
        if "device_map" not in model_kwargs:
            self.model.to(self.device)
        self.model.eval()
        log.info(
            "Model loaded: runtime_device=%s dtype=%s",
            self._input_device(),
            getattr(self.model, "dtype", "unknown"),
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.generation_config, "pad_token_id", None) is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self._lock = threading.Lock()

    def _input_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _normalize_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            normalized.append(
                {
                    "role": message.role,
                    "content": [
                        {
                            "type": "text",
                            "text": _coerce_content(message.content or ""),
                        }
                    ],
                }
            )
        return normalized

    def _build_prompt(self, messages: list[ChatMessage]) -> str:
        normalized = self._normalize_messages(messages)
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
            )
        lines: list[str] = []
        for message in normalized:
            text = _coerce_content(message["content"])
            lines.append(f"{message['role'].upper()}: {text}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    def _build_inputs(self, prompt: str) -> tuple[dict[str, torch.Tensor], int]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_device = self._input_device()
        tensor_inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
            if torch.is_tensor(value)
        }
        prompt_tokens = int(tensor_inputs["input_ids"].shape[-1])
        return tensor_inputs, prompt_tokens

    def _generation_kwargs(
        self,
        *,
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        streamer: TextIteratorStreamer | None = None,
    ) -> dict[str, Any]:
        resolved_max_tokens = int(max_tokens or self.config.max_tokens)
        resolved_temperature = float(self.config.temperature if temperature is None else temperature)
        resolved_top_p = float(self.config.top_p if top_p is None else top_p)
        do_sample = resolved_temperature > 0.0

        kwargs: dict[str, Any] = {
            "max_new_tokens": resolved_max_tokens,
            "do_sample": do_sample,
            "top_p": resolved_top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = max(resolved_temperature, 1e-5)
        if streamer is not None:
            kwargs["streamer"] = streamer
        return kwargs

    def generate_chat(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[str, dict[str, int]]:
        prompt = self._build_prompt(request.messages)
        encoded, prompt_tokens = self._build_inputs(prompt)

        with self._lock, torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                **self._generation_kwargs(
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                ),
            )

        new_tokens = outputs[0][prompt_tokens:]
        new_token_ids = new_tokens.tolist()
        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        text = _apply_stop_strings(text, request.stop)
        completion_tokens = int(len(new_token_ids))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return text, usage

    def stream_chat(self, request: ChatCompletionRequest):
        prompt = self._build_prompt(request.messages)
        encoded, prompt_tokens = self._build_inputs(prompt)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        def _run_generate() -> None:
            with self._lock, torch.inference_mode():
                self.model.generate(
                    **encoded,
                    **self._generation_kwargs(
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        streamer=streamer,
                    ),
                )

        threading.Thread(target=_run_generate, daemon=True).start()

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        def _iter_events():
            emitted = ""
            buffer = ""
            for chunk in streamer:
                buffer += chunk
                trimmed = _apply_stop_strings(buffer, request.stop)
                new_text = trimmed[len(emitted):]
                if new_text:
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.config.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": new_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                emitted = trimmed
                if len(trimmed) != len(buffer):
                    break
            final_payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.config.model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_payload)}\n\n"
            yield "data: [DONE]\n\n"

        return prompt_tokens, StreamingResponse(_iter_events(), media_type="text/event-stream")


def create_app(config: ServerConfig) -> FastAPI:
    runtime = LocalModelServer(config)
    app = FastAPI(title="RGB Agent Local Server", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model_id,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if request.model != config.model_id:
            raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")
        if request.stream:
            _, response = runtime.stream_chat(request)
            return response

        text, usage = runtime.generate_chat(request)
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": config.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            }
        )

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        if request.model != config.model_id:
            raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")

        prompt = request.prompt[0] if isinstance(request.prompt, list) else request.prompt
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=[ChatMessage(role="user", content=prompt)],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            stream=request.stream,
        )

        if request.stream:
            _, response = runtime.stream_chat(chat_request)
            return response

        text, usage = runtime.generate_chat(chat_request)
        return JSONResponse(
            {
                "id": f"cmpl-{uuid.uuid4().hex}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": config.model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            }
        )

    return app


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="Serve a local Qwen model through an OpenAI-compatible API.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model-id", default="qwen3.5-0.8b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()
    return ServerConfig(
        model=args.model,
        model_id=args.model_id,
        host=args.host,
        port=args.port,
        device=args.device,
        dtype=args.dtype,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = parse_args()
    log.info("Starting FastAPI server host=%s port=%d", config.host, config.port)
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
