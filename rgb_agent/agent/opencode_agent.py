"""OpenCodeAgent: runs OpenCode in a sandboxed Docker container to produce action plans."""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import requests
import shutil
import shlex
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Literal, Optional
from urllib.parse import urlparse, urlunparse

from rgb_agent.agent.prompts import (
    INITIAL_PROMPT,
    RESUME_PROMPT,
    COMPACT_INITIAL_PROMPT,
    COMPACT_RESUME_PROMPT,
    ACTIONS_ADDENDUM,
    COMPACT_ACTIONS_ADDENDUM,
    PYTHON_ADDENDUM,
    COMPACT_PYTHON_ADDENDUM,
    SMALL_MODEL_ADDENDUM,
)

log = logging.getLogger(__name__)

_DOCKER_IMAGE = os.environ.get("OPENCODE_DOCKER_IMAGE", "rgb-agent/opencode-sandbox:latest")
_LOCAL_ANALYZER_PROVIDER = os.environ.get("LOCAL_ANALYZER_PROVIDER", "local")
_LOCAL_ANALYZER_MODEL_ID = os.environ.get("LOCAL_ANALYZER_MODEL_ID", "qwen3.5-0.8b")
_LOCAL_ANALYZER_BASE_URL = os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://host.docker.internal:1234/v1")
_LOCAL_ANALYZER_DISPLAY_NAME = os.environ.get("LOCAL_ANALYZER_DISPLAY_NAME", "Local Qwen 3.5 0.8B")
_DEFAULT_ANALYZER_MODEL = os.environ.get("RGB_ANALYZER_MODEL", os.environ.get("ARCGYM_ANALYZER_MODEL", "local-qwen"))


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


_LOCAL_ANALYZER_CONTEXT_WINDOW = _get_env_int("LOCAL_ANALYZER_CONTEXT_WINDOW", 32768)
_LOCAL_ANALYZER_MAX_OUTPUT = _get_env_int("LOCAL_ANALYZER_MAX_OUTPUT", 2048)


@dataclass(frozen=True)
class _AnalyzerModelSpec:
    oc_model: str
    provider_config: dict[str, dict]
    compact_prompt: bool = False
    fast_by_default: bool = False
    docker_network_mode: Literal["bridge", "host"] = "bridge"
    opencode_print_logs: bool = False
    opencode_log_level: str | None = None
    harden_container: bool = True


def _local_model_discovery_urls(base_url: str) -> list[str]:
    candidates: list[str] = []

    def _add(url: str) -> None:
        normalized = url.rstrip("/")
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    _add(base_url)

    parsed = urlparse(base_url)
    hostname = parsed.hostname or ""
    if hostname == "host.docker.internal":
        for alt_host in ("127.0.0.1", "localhost"):
            replacement = parsed._replace(netloc=f"{alt_host}:{parsed.port}" if parsed.port else alt_host)
            _add(urlunparse(replacement))

    return candidates


def _discover_local_model_id(base_url: str) -> str | None:
    headers: dict[str, str] = {}
    api_key = os.environ.get("LOCAL_ANALYZER_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for candidate in _local_model_discovery_urls(base_url):
        models_url = f"{candidate}/models"
        try:
            response = requests.get(models_url, headers=headers, timeout=3)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("data", [])
            for model in models:
                model_id = str(model.get("id", "")).strip()
                if model_id:
                    return model_id
        except Exception as exc:
            log.debug("local model discovery failed via %s: %s", models_url, exc)
    return None


def _container_local_base_url(base_url: str) -> tuple[str, Literal["bridge", "host"]]:
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").strip().lower()
    if hostname not in {"host.docker.internal", "localhost", "127.0.0.1"}:
        return base_url, "bridge"

    netloc = "127.0.0.1"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc)), "host"


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _wait_for_container_tcp_port(container_name: str, port: int, timeout_seconds: int = 10) -> bool:
    probe = (
        "import socket; "
        f"s=socket.create_connection(('127.0.0.1',{port}), timeout=1); "
        "s.close(); "
        "print('ok')"
    )
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "exec", container_name, "python3", "-c", probe],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and "ok" in result.stdout:
            return True
        time.sleep(0.5)
    return False


def _resolve_analyzer_model(model: str) -> _AnalyzerModelSpec:
    requested = (model or "").strip()
    lowered = requested.lower()

    if lowered in {"local", "local-qwen", "qwen-local", "qwen"}:
        provider_id = os.environ.get("LOCAL_ANALYZER_PROVIDER", _LOCAL_ANALYZER_PROVIDER).strip() or "local"
        base_url = os.environ.get("LOCAL_ANALYZER_BASE_URL", _LOCAL_ANALYZER_BASE_URL).strip()
        if not base_url:
            raise ValueError("LOCAL_ANALYZER_BASE_URL must be set for the local analyzer preset.")

        configured_model_id = os.environ.get("LOCAL_ANALYZER_MODEL_ID", _LOCAL_ANALYZER_MODEL_ID).strip()
        discovered_model_id = _discover_local_model_id(base_url)
        model_id = discovered_model_id or configured_model_id
        if not model_id:
            raise ValueError("LOCAL_ANALYZER_MODEL_ID must be set for the local analyzer preset.")
        if discovered_model_id and discovered_model_id != configured_model_id:
            log.info(
                "Local analyzer model id mismatch: configured=%s discovered=%s; using discovered",
                configured_model_id or "<unset>",
                discovered_model_id,
            )

        container_base_url, docker_network_mode = _container_local_base_url(base_url)
        provider_options: dict[str, str] = {"baseURL": container_base_url}
        if os.environ.get("LOCAL_ANALYZER_API_KEY"):
            provider_options["apiKey"] = "{env:LOCAL_ANALYZER_API_KEY}"

        model_options: dict[str, int] = {}
        if _LOCAL_ANALYZER_CONTEXT_WINDOW > 0:
            model_options["contextWindow"] = _LOCAL_ANALYZER_CONTEXT_WINDOW
        if _LOCAL_ANALYZER_MAX_OUTPUT > 0:
            model_options["maxOutput"] = _LOCAL_ANALYZER_MAX_OUTPUT

        provider_config = {
            provider_id: {
                "npm": "@ai-sdk/openai-compatible",
                "name": os.environ.get("LOCAL_ANALYZER_PROVIDER_NAME", "Local OpenAI-Compatible"),
                "options": provider_options,
                "models": {
                    model_id: {
                        "name": os.environ.get("LOCAL_ANALYZER_DISPLAY_NAME", _LOCAL_ANALYZER_DISPLAY_NAME),
                        "options": model_options,
                    },
                },
            },
        }
        return _AnalyzerModelSpec(
            oc_model=f"{provider_id}/{model_id}",
            provider_config=provider_config,
            compact_prompt=True,
            fast_by_default=True,
            docker_network_mode=docker_network_mode,
            opencode_print_logs=True,
            opencode_log_level="DEBUG",
            harden_container=False,
        )

    if lowered == "opus":
        requested = "claude-opus-4-6"
    elif lowered == "sonnet":
        requested = "claude-sonnet-4-6"

    oc_model = requested if "/" in requested else f"anthropic/{requested}"
    provider_id = oc_model.split("/", 1)[0]
    return _AnalyzerModelSpec(
            oc_model=oc_model,
            provider_config={provider_id: {}},
            compact_prompt=False,
            fast_by_default=False,
            opencode_print_logs=False,
            opencode_log_level=None,
            harden_container=True,
        )


def _docker_image_exists(image: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


class _EventStreamParser:
    """Parses nd-JSON events from OpenCode and writes to an analyzer log."""

    def __init__(self, f: IO[str]):
        self._f = f
        self.accumulated_text = ""
        self.session_id: str | None = None

    def _write(self, label: str, content: str) -> None:
        if content:
            self._f.write(f"[{label}]\n{content}\n\n")
            self._f.flush()

    def _write_tool(self, name: str, state: dict) -> None:
        status = state.get("status", "?")
        if status in ("running", "completed", "done"):
            input_data = state.get("input", {})
            input_str = json.dumps(input_data, indent=2) if isinstance(input_data, dict) else str(input_data)
            self._write(f"TOOL CALL: {name}", input_str)
        if status in ("completed", "done"):
            output = state.get("output", state.get("result", ""))
            is_error = state.get("is_error", False) or state.get("error", False)
            label = "TOOL RESULT ERROR" if is_error else "TOOL RESULT"
            self._write(label, str(output)[:4000])

    def handle(self, event: dict) -> None:
        etype = event.get("type")
        log.debug("event type=%s", etype)

        if etype == "step_start":
            sid = event.get("sessionID")
            if sid and not self.session_id:
                self.session_id = sid

        elif etype == "text":
            text = event.get("part", {}).get("text", "")
            if text:
                self.accumulated_text += text
                self._write("ASSISTANT", text)

        elif etype == "tool_use":
            part = event.get("part", {})
            self._write_tool(part.get("tool", "?"), part.get("state", {}))

        elif etype == "message.part.updated":
            part = event.get("part", {})
            ptype = part.get("type")
            if ptype in ("thinking", "reasoning"):
                self._write("THINKING", part.get("text", ""))
            elif ptype == "tool":
                name = part.get("name", "?")
                pstate = part.get("state", "?")
                if pstate == "running":
                    input_data = part.get("input", {})
                    input_str = json.dumps(input_data, indent=2) if isinstance(input_data, dict) else str(input_data)
                    self._write(f"TOOL CALL: {name}", input_str)
                elif pstate in ("completed", "done"):
                    result = part.get("result", part.get("output", ""))
                    text = result if isinstance(result, str) else str(result)
                    is_error = part.get("is_error", False) or part.get("error", False)
                    label = "TOOL RESULT ERROR" if is_error else "TOOL RESULT"
                    self._write(label, text[:4000])

        elif etype == "error":
            err = event.get("error", {})
            name = err.get("name", "UnknownError")
            msg = err.get("data", {}).get("message", str(err))
            self._write(f"ERROR: {name}", msg)
            log.error("API error: %s: %s", name, msg)
            if "overflow" in name.lower() or "too long" in msg.lower():
                self.session_id = None

        elif etype == "step_finish":
            cost = event.get("part", {}).get("cost")
            self._write("RESULT", f"cost=${cost}")

        elif etype == "assistant":
            for block in event.get("message", {}).get("content", []):
                btype = block.get("type")
                if btype == "thinking":
                    self._write("THINKING", block.get("thinking", ""))
                elif btype == "text":
                    text = block["text"]
                    self.accumulated_text += text
                    self._write("ASSISTANT", text)
                elif btype == "tool_use":
                    self._write(f"TOOL CALL: {block['name']}", json.dumps(block.get("input", {}), indent=2))

        elif etype == "user":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, list):
                        text = "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
                    elif isinstance(content, str):
                        text = content
                    else:
                        text = str(content)
                    is_error = block.get("is_error", False)
                    label = "TOOL RESULT ERROR" if is_error else "TOOL RESULT"
                    self._write(label, text[:4000])

        elif etype == "result":
            result_text = event.get("result", "").strip()
            if result_text and not self.accumulated_text.strip():
                self.accumulated_text = result_text
            cost = event.get("total_cost_usd")
            self._write("RESULT", f"cost=${cost}")

        else:
            self._f.write(f"[RAW:{etype}] {json.dumps(event)[:500]}\n")
            self._f.flush()


class _ContainerPool:
    """Manages persistent Docker containers running `opencode serve`."""

    def __init__(
        self,
        config_path: Path,
        permission: dict,
        docker_image: str,
        sandbox_prefix: str,
        network_mode: Literal["bridge", "host"] = "bridge",
        harden_container: bool = True,
    ):
        self._config_path = config_path
        self._permission = permission
        self._image = docker_image
        self._prefix = sandbox_prefix
        self._network_mode = network_mode
        self._harden_container = harden_container
        self._containers: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[str, int, str]:
        with self._lock:
            if key in self._containers:
                info = self._containers[key]
                check = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", info["name"]],
                    capture_output=True, text=True, timeout=5,
                )
                if check.returncode == 0 and "true" in check.stdout.lower():
                    return info["name"], info["port"], info["sandbox_dir"]
                log.warning("server container %s died, recreating", info["name"])
                subprocess.run(["docker", "rm", "-f", info["name"]], capture_output=True, timeout=10)
                del self._containers[key]

            return self._create(key)

    def _create(self, key: str) -> tuple[str, int, str]:
        sandbox = tempfile.mkdtemp(prefix=self._prefix)
        os.chmod(sandbox, 0o777)
        name = f"oc_{uuid.uuid4().hex[:12]}"
        port = _find_free_tcp_port() if self._network_mode == "host" else 4096

        shutil.copy2(self._config_path, Path(sandbox) / "opencode.json")

        env_flags: list[str] = []
        for key_name in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
            "LOCAL_ANALYZER_API_KEY",
        ):
            val = os.environ.get(key_name)
            if val:
                env_flags.extend(["-e", f"{key_name}={val}"])

        cmd = [
            "docker", "run", "-d",
            "--name", name,
        ]
        if self._harden_container:
            cmd.extend([
                "--read-only",
                "--user", "1000:1000",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges:true",
                "--memory=4g", "--cpus=2",
                "--pids-limit=128",
                "--shm-size=8m",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,uid=1000,gid=1000",
                "--tmpfs", "/home/opencode:rw,noexec,nosuid,size=128m,uid=1000,gid=1000",
            ])
        cmd.extend([
            "-v", f"{os.path.realpath(sandbox)}:/workspace:rw",
            "-e", "OPENCODE_CONFIG=/workspace/opencode.json",
            *env_flags,
        ])
        if self._network_mode == "host":
            cmd.extend(["--network", "host"])
        else:
            cmd.extend(["--add-host", "host.docker.internal:host-gateway"])
        cmd.append(self._image)
        cmd.extend(["serve", "--port", str(port), "--hostname", "127.0.0.1" if self._network_mode == "host" else "0.0.0.0"])

        subprocess.run(cmd, check=True, capture_output=True, timeout=30)

        for _ in range(45):
            time.sleep(1)
            state = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", name],
                capture_output=True, text=True, timeout=15,
            )
            if state.returncode != 0 or "true" not in state.stdout.lower():
                logs = subprocess.run(
                    ["docker", "logs", name], capture_output=True, text=True, timeout=15,
                )
                raise RuntimeError(
                    f"opencode server container {name} exited during startup: "
                    f"{(logs.stderr or logs.stdout).strip() or 'no logs captured'}"
                )
            logs = subprocess.run(
                ["docker", "logs", name], capture_output=True, text=True, timeout=15,
            )
            if "listening" in logs.stdout or "listening" in logs.stderr:
                break
        else:
            log.warning("server %s may not be ready (timeout)", name)
        if not _wait_for_container_tcp_port(name, port):
            raise RuntimeError(f"opencode server container {name} did not open port {port} in time")

        self._containers[key] = {"name": name, "port": port, "sandbox_dir": sandbox}
        log.info("container ready: %s", name)
        return name, port, sandbox

    def cleanup(self) -> None:
        with self._lock:
            for info in self._containers.values():
                try:
                    log.info("stopping container: %s", info["name"])
                    subprocess.run(["docker", "stop", "-t", "3", info["name"]], capture_output=True, timeout=10)
                    subprocess.run(["docker", "rm", "-f", info["name"]], capture_output=True, timeout=10)
                except Exception as e:
                    log.warning("failed to cleanup container %s: %s", info["name"], e)
                if info.get("sandbox_dir"):
                    shutil.rmtree(info["sandbox_dir"], ignore_errors=True)
            self._containers.clear()


class OpenCodeAgent:
    """Runs OpenCode in a sandboxed Docker container to analyze game logs and produce action plans."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_ANALYZER_MODEL,
        interval: int = 0,
        timeout: Optional[int] = None,
        allow_bash: bool = True,
        action_mode: Optional[Literal["move", "click", "all"]] = "all",
        plan_size: int = 5,
        allow_self_read: bool = False,
        fast: bool = False,
        resume_session: bool = True,
    ) -> None:
        if not shutil.which("docker"):
            raise FileNotFoundError("'docker' CLI not found. Install Docker Desktop to use the analyzer.")
        if not _docker_image_exists(_DOCKER_IMAGE):
            raise FileNotFoundError(
                f"Docker image '{_DOCKER_IMAGE}' not found. Build with:\n"
                f"  cd docker/opencode-sandbox && bash build.sh"
            )
        log.info("using Docker sandbox: %s", _DOCKER_IMAGE)

        self._model_spec = _resolve_analyzer_model(model)
        self._oc_model = self._model_spec.oc_model
        self._interval = interval
        self._timeout = timeout
        self._allow_bash = allow_bash
        self._action_mode = action_mode
        self._plan_size = plan_size
        self._allow_self_read = allow_self_read
        self._fast = fast
        self._resume_session = resume_session

        if self._model_spec.harden_container:
            permission: dict = {
                "*": "deny",
                "read": "allow",
                "grep": "allow",
                "bash": {
                    "*": "deny",
                    "python3 *": "allow",
                    "python *": "allow",
                } if allow_bash else "deny",
                "external_directory": "deny",
                "doom_loop": "allow",
                "question": "deny",
                "edit": "deny",
                "write": "deny",
                "patch": "deny",
                "glob": "deny",
                "list": "deny",
                "lsp": "deny",
                "skill": "deny",
                "webfetch": "deny",
                "websearch": "deny",
                "todowrite": "deny",
                "todoread": "deny",
            }
        else:
            permission = {
                "*": "deny",
                "read": "allow",
                "grep": "allow",
            }

        config = {
            "model": self._oc_model,
            "provider": self._model_spec.provider_config,
            "permission": permission,
        }

        config_dir = tempfile.mkdtemp(prefix="opencode_analyzer_")
        config_path = Path(config_dir) / "opencode.json"
        config_path.write_text(json.dumps(config, indent=2))
        atexit.register(shutil.rmtree, config_dir, True)

        self._pool = _ContainerPool(
            config_path,
            permission,
            _DOCKER_IMAGE,
            f"oc_sandbox_{uuid.uuid4().hex[:8]}_",
            network_mode=self._model_spec.docker_network_mode,
            harden_container=self._model_spec.harden_container,
        )
        atexit.register(self._pool.cleanup)

        self._session_ids: dict[str, str] = {}
        self._session_lock = threading.Lock()

    def _build_prompt(self, log_name: str, analyzer_log_name: str,
                      analyzer_log_exists: bool, is_first: bool) -> str:
        use_compact = self._model_spec.compact_prompt
        if self._resume_session and not is_first:
            base_prompt = COMPACT_RESUME_PROMPT if use_compact else RESUME_PROMPT
            prompt = base_prompt.format(log_path=log_name)
        else:
            base_prompt = COMPACT_INITIAL_PROMPT if use_compact else INITIAL_PROMPT
            prompt = base_prompt.format(log_path=log_name)
            if self._allow_self_read and analyzer_log_exists:
                prompt += (
                    f"\n\nYour previous analysis output is at: {analyzer_log_name}\n"
                    "Read it to see what you concluded last time and build on it. "
                    "Avoid repeating strategies that didn't work."
                )
        if self._allow_bash:
            python_addendum = COMPACT_PYTHON_ADDENDUM if use_compact else PYTHON_ADDENDUM
            prompt += python_addendum.format(log_path=log_name)
        if use_compact:
            prompt += SMALL_MODEL_ADDENDUM
        if self._action_mode:
            action_addendum = COMPACT_ACTIONS_ADDENDUM if use_compact else ACTIONS_ADDENDUM
            prompt += action_addendum.format(plan_size=self._plan_size)
        return prompt

    def _try_recover_text(self, container_name: str, sid: str, sandbox_dir: str) -> str:
        export_path = Path(sandbox_dir) / "_export.json"
        try:
            subprocess.run(
                ["docker", "exec", container_name, "sh", "-c",
                 f"opencode export {sid} > /workspace/_export.json 2>/dev/null"],
                capture_output=True, text=True, timeout=30,
            )
            if not export_path.exists():
                return ""
            data = json.loads(export_path.read_text())
            recovered = ""
            for msg in reversed(data.get("messages", [])):
                role = msg.get("info", {}).get("role")
                if role == "assistant":
                    for part in msg.get("parts", []):
                        if part.get("type") == "text":
                            candidate = part.get("text", "").strip()
                            if candidate and "[ACTIONS]" in candidate:
                                return candidate
                            if candidate and not recovered:
                                recovered = candidate
                    if recovered and "[ACTIONS]" in recovered:
                        return recovered
            return recovered
        except Exception as e:
            log.debug("export recovery failed: %s", e)
            return ""

    def analyze(
        self,
        log_path: Path,
        action_num: int,
        transcript_path: Path | None = None,
        analysis_step: int | None = None,
        retry_nudge: str = "",
        retry_attempt: int | None = None,
        retry_total: int | None = None,
    ) -> Optional[str]:
        """Analyze the game log and return the agent's response text, or None on failure."""
        if self._interval > 0 and action_num % self._interval != 0:
            return None
        if not log_path.exists():
            return None

        analyzer_log = transcript_path or (log_path.parent / (log_path.stem + "_analyzer.txt"))
        path_key = str(log_path)

        is_first = True
        current_sid = None
        if self._resume_session:
            with self._session_lock:
                if path_key in self._session_ids:
                    current_sid = self._session_ids[path_key]
                    is_first = False

        container_name, server_port, sandbox_dir = self._pool.get(path_key)
        sandbox = Path(sandbox_dir)

        try:
            shutil.copy2(log_path, sandbox / log_path.name)
            if self._allow_self_read and analyzer_log.exists():
                shutil.copy2(analyzer_log, sandbox / analyzer_log.name)

            prompt = self._build_prompt(log_path.name, analyzer_log.name, analyzer_log.exists(), is_first)
            if retry_nudge:
                prompt += f"\n\n{retry_nudge}"

            oc_args: list[str] = []
            if self._model_spec.opencode_print_logs:
                oc_args.append("--print-logs")
            if self._model_spec.opencode_log_level:
                oc_args.extend(["--log-level", self._model_spec.opencode_log_level])
            oc_args.extend(["run", "--attach", f"http://127.0.0.1:{server_port}"])
            if self._resume_session and not is_first and current_sid:
                oc_args.extend(["--session", current_sid, "--continue"])
            oc_args.extend(["--model", self._oc_model])
            if self._fast or self._model_spec.fast_by_default:
                oc_args.extend(["--variant", "minimal"])
            oc_args.extend(["--format", "json", "--dir", "/workspace"])
            prompt_file_name = f"_opencode_prompt_{time.time_ns()}.txt"
            prompt_path = sandbox / prompt_file_name
            prompt_path.write_text(prompt, encoding="utf-8")
            shell_cmd = (
                f"{shlex.join(['opencode', *oc_args])} "
                f'"$(cat {shlex.quote(f"/workspace/{prompt_file_name}")})"'
            )

            cmd = ["docker", "exec", container_name, "sh", "-lc", shell_cmd]
            log.info("exec %s model=%s%s", container_name, self._oc_model,
                     f" session={current_sid}" if current_sid else "")

            proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )

            stderr_lines: list[str] = []
            def drain_stderr():
                for line in proc.stderr:
                    stderr_lines.append(line.rstrip("\n"))
                    log.debug("STDERR: %s", line[:300].rstrip())

            stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
            stderr_thread.start()

            with open(analyzer_log, "a", encoding="utf-8") as f:
                step_label = f"analysis_step={analysis_step} | " if analysis_step is not None else ""
                retry_label = (
                    f" | analyzer_retry={retry_attempt}/{retry_total}"
                    if retry_attempt is not None and retry_total is not None
                    else ""
                )
                f.write(
                    f"\n--- {step_label}action={action_num}{retry_label} | "
                    f"{datetime.now().strftime('%H:%M:%S')} | opencode ---\n"
                )
                f.write(f"[SYSTEM PROMPT]\n{prompt}\n\n")
                f.flush()

                parser = _EventStreamParser(f)
                deadline = time.monotonic() + self._timeout if self._timeout is not None else None
                assert proc.stdout is not None
                stdout_queue: queue.Queue[str | None] = queue.Queue()

                def drain_stdout() -> None:
                    for stdout_line in proc.stdout:
                        stdout_queue.put(stdout_line)
                    stdout_queue.put(None)

                stdout_thread = threading.Thread(target=drain_stdout, daemon=True)
                stdout_thread.start()
                stdout_open = True

                while True:
                    if deadline is not None and time.monotonic() > deadline:
                        proc.kill()
                        f.write("[TIMEOUT]\n")
                        log.warning("timed out at action %d", action_num)
                        return None

                    try:
                        line = stdout_queue.get(timeout=1.0)
                    except queue.Empty:
                        if proc.poll() is not None and not stdout_open:
                            break
                        continue

                    if line is None:
                        stdout_open = False
                        if proc.poll() is not None:
                            break
                        continue

                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    try:
                        parser.handle(json.loads(line))
                    except json.JSONDecodeError:
                        f.write(f"[RAW] {line}\n")
                        f.flush()

                proc.wait()
                stdout_thread.join(timeout=5)
                stderr_thread.join(timeout=5)
                if stderr_lines:
                    f.write(f"\n--- STDERR ---\n{''.join(l + chr(10) for l in stderr_lines)}")
                    f.flush()

                needs_recovery = (
                    not parser.accumulated_text.strip()
                    or (self._action_mode and "[ACTIONS]" not in parser.accumulated_text)
                )
                if needs_recovery and parser.session_id:
                    recovered = self._try_recover_text(container_name, parser.session_id, sandbox_dir)
                    if recovered:
                        parser.accumulated_text = recovered
                        log.info("recovered %d chars via session export", len(recovered))

                if self._resume_session and parser.session_id is None and not is_first:
                    log.warning("context overflow — clearing session for %s", path_key)
                    with self._session_lock:
                        self._session_ids.pop(path_key, None)

                f.flush()

            hint = parser.accumulated_text.strip() or None
            missing_actions = bool(hint and self._action_mode and "[ACTIONS]" not in hint)

            if proc.returncode != 0 or not hint:
                with open(analyzer_log, "a", encoding="utf-8") as f:
                    f.write("[ANALYZER STATUS]\n")
                    f.write(f"return_code: {proc.returncode}\n")
                    f.write(f"assistant_output_chars: {len(parser.accumulated_text.strip())}\n")
                    if not hint:
                        f.write("message: No assistant text or action plan was captured.\n")
                    f.write("\n")
                log.warning("action=%d failed: rc=%d, hint_len=%d",
                            action_num, proc.returncode, len(hint) if hint else 0)
                if self._resume_session:
                    with self._session_lock:
                        self._session_ids.pop(path_key, None)
                return None

            if missing_actions and self._resume_session:
                log.info("clearing session for %s because response omitted [ACTIONS]", path_key)
                with self._session_lock:
                    self._session_ids.pop(path_key, None)

            if self._resume_session and parser.session_id and not missing_actions:
                with self._session_lock:
                    self._session_ids[path_key] = parser.session_id

            log.info("action=%d OK (%d chars)", action_num, len(hint))
            return hint

        except Exception as e:
            log.error("unexpected error: %s", e, exc_info=True)
            return None
        finally:
            if 'prompt_path' in locals() and prompt_path.exists():
                prompt_path.unlink(missing_ok=True)
