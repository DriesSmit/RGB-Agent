"""
OpenCode-based prompt log analyzer.

Spawns `opencode run` inside a Docker sandbox to analyze the agent's
prompt log. All write/web/edit tools are explicitly denied via a generated
opencode.json config file.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (shared constants)
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """\
You are a strategic advisor for an AI agent playing a grid-based puzzle game.
The agent's full prompt log for this run is at this ABSOLUTE path: {log_path}

You may only access this single file (use its absolute path directly with Read and Grep).

Most games have some form of timer mechanism. A score increase means a level was solved.

Deeply analyze this log to understand what the agent has been doing, what has worked,
what hasn't, and what patterns explain the game's behavior.

Your response MUST contain ALL sections below — the agent cannot act without [ACTIONS]:
1. A detailed strategic briefing (explain your reasoning, be specific with coordinates)
2. Followed by exactly this separator and a 2-3 sentence action plan:

[PLAN]
<concise action plan the agent should follow until the next analysis>
"""

_RESUME_FOLLOW_UP_PROMPT = """\
The prompt log has grown since your last analysis. The log file is at: {log_path}

Re-read the latest actions (from where you left off) and update your strategic briefing.
Focus on what changed: new moves, score transitions, and whether the agent followed
your previous plan or diverged. Parse the board programmatically from the file using
section markers ([POST-ACTION BOARD STATE], etc.) — do NOT visually copy the grid.

Your response MUST contain ALL three sections below — the agent cannot act without [ACTIONS]:
1. A detailed strategic briefing (explain your reasoning, be specific with coordinates)
2. Followed by exactly this separator and a 2-3 sentence action plan:

[PLAN]
<concise action plan the agent should follow until the next analysis>
"""

_ACTIONS_ADDENDUM = """
3. Followed by exactly this separator and a JSON action plan (REQUIRED — the agent cannot act without this):

[ACTIONS]
{{"plan": [{{"action": "ACTION1"}}, {{"action": "ACTION6", "x": 3, "y": 7}}, ...], "reasoning": "why these steps"}}

Available actions: ACTION1-4 (moves), ACTION6 (click at x,y), ACTION5 (no-op), RESET.
Each action MUST be a JSON object: {{"action": "ACTION6", "x": <row>, "y": <col>}} for clicks, {{"action": "ACTION1"}} for moves. Never use string shorthand like "ACTION6(x,y)".
Plan 1–{plan_size} actions. IMPORTANT: shorter plans (3-5 steps) are strongly preferred
because the agent can re-evaluate sooner. Only use more than 5 if you have very high
confidence AND the extra steps are critical. Even on a clear straight path, prefer
stopping early so the agent can observe the game's response and adapt.
\
"""

_PYTHON_ADDENDUM = (
    "\n\nBash (and therefore Python) is available to you. **Always** use Python to "
    "parse the board — do NOT try to visually read the ASCII grid.\n\n"
    "The log file uses section markers to delimit board grids:\n"
    "  [INITIAL BOARD STATE]   — the grid at the start (after Action 0 header)\n"
    "  [POST-ACTION BOARD STATE] — the grid after each action\n"
    "\n"
    "To extract the latest board into a matrix:\n"
    "```python\n"
    "import re\n"
    "data = open('{log_path}').read()\n"
    "# Find the last board state section\n"
    "boards = re.split(r'\\[(?:POST-ACTION|INITIAL) BOARD STATE\\]', data)\n"
    "last_board = boards[-1].strip()\n"
    "# Skip 'Score: N' line if present, then parse rows into a 2-D list\n"
    "lines = last_board.split('\\n')\n"
    "if lines[0].startswith('Score:'):\n"
    "    lines = lines[1:]\n"
    "grid = [list(row) for row in lines if row.strip()]\n"
    "# Now slice, count, compare programmatically\n"
    "```\n"
    "Run Python inline."
)

_HAS_DOCKER = shutil.which("docker") is not None
_DOCKER_IMAGE = os.environ.get("OPENCODE_DOCKER_IMAGE", "arcgym/opencode-sandbox:latest")


def _docker_image_exists(image: str) -> bool:
    """Check if the Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def make_opencode_analyzer(
    interval: int = 5,
    timeout: Optional[int] = None,
    use_subscription: bool = False,
    allow_bash: bool = False,
    action_mode: Optional[Literal["move", "click", "all"]] = None,
    plan_size: int = 5,
    allow_self_read: bool = False,
    model: str = "claude-opus-4-6",
    fast: bool = False,
    resume_session: bool = False,
) -> Callable[[Path, int], Optional[str]]:
    """
    Returns a hook function that calls OpenCode to analyze the prompt log.

    Args:
        interval: Maximum actions between analyzer invocations. 0 = every action.
        timeout: Max seconds to wait for opencode subprocess.
        use_subscription: Ignored for OpenCode (always uses ANTHROPIC_API_KEY).
        allow_bash: If True, allow Bash/Python for board analysis.
        action_mode: If set, output [ACTIONS] batch plan. "move"/"click"/"all".
        plan_size: Max actions per batch plan (default: 5).
        allow_self_read: If True, analyzer can read its own previous output.
        model: Model name (default: claude-opus-4-6). Auto-prefixed with
            "anthropic/" if no provider prefix present.
        fast: Ignored for OpenCode (no equivalent feature).
        resume_session: If True, reuse OpenCode session across invocations.
    The returned hook has signature: hook(log_path, action_num) -> hint_str | None
    """
    # --- Verify Docker is available ---
    if not _HAS_DOCKER:
        raise FileNotFoundError(
            "'docker' CLI not found in PATH. Install Docker Desktop to use the OpenCode analyzer."
        )
    if not _docker_image_exists(_DOCKER_IMAGE):
        raise FileNotFoundError(
            f"Docker image '{_DOCKER_IMAGE}' not found. Build it with:\n"
            f"  cd docker/opencode-sandbox && bash build.sh"
        )
    log.info(f"[opencode_analyzer] using Docker sandbox: {_DOCKER_IMAGE}")

    prompt_template = ANALYSIS_PROMPT

    # Normalize model name to provider/model format
    # Supports: "anthropic/claude-opus-4-6", "openai/gpt-4o", "openai/o3", etc.
    oc_model = model if "/" in model else f"anthropic/{model}"
    oc_provider = oc_model.split("/")[0]  # e.g. "anthropic", "openai"

    # --- Build OpenCode config file (once per analyzer instance) ---
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
        # Deny interactive/write/web tools explicitly
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

    config = {
        "model": oc_model,
        "provider": {
            oc_provider: {}
        },
        "permission": permission,
        "agent": {
            "build": {
                "steps": 50,
            }
        },
    }

    _config_dir = tempfile.mkdtemp(prefix="opencode_analyzer_")
    _config_path = Path(_config_dir) / "opencode.json"
    _config_path.write_text(json.dumps(config, indent=2))
    log.info(f"[opencode_analyzer] config written to {_config_path}")

    # Clean up temp dir on exit
    atexit.register(shutil.rmtree, _config_dir, True)

    # Unique sandbox prefix per analyzer instance (safe for concurrent runs)
    _sandbox_prefix = f"oc_sandbox_{uuid.uuid4().hex[:8]}_"

    # Session resume state (thread-safe: one session per log_path)
    _session_ids: dict[str, str] = {}
    _session_lock = threading.Lock()

    # --- Persistent Docker server containers (one per game / log_path) ---
    # Maps log_path_key → {"container_name": str, "port": int, "sandbox_dir": str}
    _server_containers: dict[str, dict] = {}
    _server_lock = threading.Lock()

    def _ensure_server(log_path_key: str) -> tuple[str, int, str]:
        """
        Start (or reuse) a persistent Docker container running `opencode serve`.
        Returns (container_name, port, sandbox_dir).
        The sandbox_dir is the persistent bind-mounted directory for /workspace.
        """
        with _server_lock:
            if log_path_key in _server_containers:
                info = _server_containers[log_path_key]
                # Verify container is still running
                check = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", info["container_name"]],
                    capture_output=True, text=True, timeout=5,
                )
                if check.returncode == 0 and "true" in check.stdout.lower():
                    return info["container_name"], info["port"], info["sandbox_dir"]
                else:
                    # Container died — remove entry, will recreate below
                    log.warning(f"[opencode_analyzer] server container {info['container_name']} died, recreating")
                    try:
                        subprocess.run(["docker", "rm", "-f", info["container_name"]],
                                       capture_output=True, timeout=10)
                    except Exception:
                        pass
                    del _server_containers[log_path_key]

            # Create a persistent sandbox dir for this server container
            # chmod 777 so container user (1000:1000) can write on Linux
            sbox_dir = tempfile.mkdtemp(prefix=_sandbox_prefix)
            os.chmod(sbox_dir, 0o777)
            container_name = f"oc_{uuid.uuid4().hex[:12]}"
            port = 4096
            real_sandbox = os.path.realpath(sbox_dir)

            # Copy config into sandbox so container can read it
            shutil.copy2(_config_path, Path(sbox_dir) / "opencode.json")

            # Collect API keys
            env_flags: list[str] = []
            for key_name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
                val = os.environ.get(key_name)
                if val:
                    env_flags.extend(["-e", f"{key_name}={val}"])

            cmd = [
                "docker", "run", "-d",
                "--name", container_name,
                "--read-only",
                "--user", "1000:1000",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges:true",
                "--memory=4g", "--cpus=2",
                "--pids-limit=128",
                "--shm-size=8m",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,uid=1000,gid=1000",
                "--tmpfs", "/home/opencode:rw,noexec,nosuid,size=128m,uid=1000,gid=1000",
                "-v", f"{real_sandbox}:/workspace:rw",
                "-e", "OPENCODE_CONFIG=/workspace/opencode.json",
                "-e", f"OPENCODE_PERMISSION={json.dumps(permission)}",
                *env_flags,
                _DOCKER_IMAGE,
                "serve", "--port", str(port), "--hostname", "0.0.0.0",
            ]

            log.debug(f"[opencode_analyzer] starting server container: {container_name}")
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)

            # Wait for server to be ready (poll logs for "listening")
            for _ in range(15):
                time.sleep(1)
                logs_result = subprocess.run(
                    ["docker", "logs", container_name],
                    capture_output=True, text=True, timeout=15,
                )
                if "listening" in logs_result.stdout or "listening" in logs_result.stderr:
                    break
            else:
                log.warning(f"[opencode_analyzer] server {container_name} may not be ready (timeout waiting for 'listening')")

            _server_containers[log_path_key] = {
                "container_name": container_name,
                "port": port,
                "sandbox_dir": sbox_dir,
            }
            log.info(f"[opencode_analyzer] container ready: {container_name}")
            return container_name, port, sbox_dir

    def _cleanup_servers():
        """Stop and remove all persistent server containers and their sandbox dirs."""
        with _server_lock:
            for key, info in list(_server_containers.items()):
                name = info["container_name"]
                sbox = info.get("sandbox_dir")
                try:
                    log.info(f"[opencode_analyzer] stopping server container: {name}")
                    subprocess.run(["docker", "stop", "-t", "3", name],
                                   capture_output=True, timeout=10)
                    subprocess.run(["docker", "rm", "-f", name],
                                   capture_output=True, timeout=10)
                except Exception as e:
                    log.warning(f"[opencode_analyzer] failed to cleanup container {name}: {e}")
                if sbox:
                    shutil.rmtree(sbox, ignore_errors=True)
            _server_containers.clear()

    atexit.register(_cleanup_servers)

    def hook(log_path: Path, action_num: int, retry_nudge: str = "") -> Optional[str]:
        if interval > 0 and action_num % interval != 0:
            return None
        if not log_path.exists():
            return None

        analyzer_log = log_path.parent / (log_path.stem + "_analyzer.txt")

        # --- Session resume: detect first-call vs follow-up ---
        log_path_key = str(log_path)
        is_first_call = True
        current_session_id = None
        if resume_session:
            with _session_lock:
                if log_path_key in _session_ids:
                    current_session_id = _session_ids[log_path_key]
                    is_first_call = False

        # --- Sandbox: reuse the persistent sandbox dir from _ensure_server ---
        _container_name, _server_port, sandbox_dir = _ensure_server(log_path_key)
        sandbox = Path(sandbox_dir)

        try:
            # Copy fresh files into sandbox (overwrite for Docker persistent dir)
            shutil.copy2(log_path, sandbox / log_path.name)
            local_log = log_path.name

            # Copy analyzer log if self-read is enabled and it exists
            local_analyzer_log = analyzer_log.name
            if allow_self_read and analyzer_log.exists():
                shutil.copy2(analyzer_log, sandbox / analyzer_log.name)

            # --- Build prompt ---
            if resume_session and not is_first_call:
                prompt = _RESUME_FOLLOW_UP_PROMPT.format(log_path=local_log)
            else:
                prompt = prompt_template.format(log_path=local_log)
                if allow_self_read and analyzer_log.exists():
                    prompt += (
                        f"\n\nYour previous analysis output is at: {local_analyzer_log}\n"
                        "Read it to see what you concluded last time and build on it. "
                        "Avoid repeating strategies that didn't work."
                    )
            if allow_bash:
                prompt += _PYTHON_ADDENDUM.format(log_path=local_log)
            if action_mode:
                prompt += _ACTIONS_ADDENDUM.format(plan_size=plan_size)
            if retry_nudge:
                prompt += f"\n\n{retry_nudge}"

            session_label = ""
            if resume_session and current_session_id:
                session_label = f" session={current_session_id} ({'new' if is_first_call else 'resume'})"

            # --- Docker sandbox: persistent server + docker exec ---
            oc_args = ["run", "--attach", f"http://127.0.0.1:{_server_port}"]
            if resume_session and not is_first_call and current_session_id:
                oc_args.extend(["--session", current_session_id, "--continue"])
            oc_args.extend(["--model", oc_model])
            if fast:
                oc_args.extend(["--variant", "minimal"])
            oc_args.extend(["--format", "json", "--dir", "/workspace"])
            oc_args.append(prompt)

            cmd = ["docker", "exec", _container_name, "opencode", *oc_args]
            popen_env = None
            popen_cwd = None
            log.info(f"[opencode_analyzer] exec {_container_name} model={oc_model}{session_label}")
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=popen_env,
                cwd=popen_cwd,
            )
            log.debug(f"[opencode_analyzer] process started: pid={proc.pid}")

            # Stream-parse stdout line-by-line (nd-JSON), writing to sidecar log
            accumulated_text = ""
            session_id_captured = None

            # Stream stderr in background so we see errors in real-time
            _stderr_lines: list[str] = []

            def _drain_stderr():
                for err_line in proc.stderr:
                    err_line = err_line.rstrip("\n")
                    _stderr_lines.append(err_line)
                    log.debug(f"[opencode_analyzer] STDERR: {err_line[:300]}")

            stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()

            with open(analyzer_log, "a", encoding="utf-8") as f:
                f.write(f"\n--- action={action_num} | {datetime.now().strftime('%H:%M:%S')} | opencode ---\n")
                if is_first_call or not resume_session:
                    f.write(f"[SYSTEM PROMPT]\n{prompt}\n\n")
                f.flush()

                deadline = time.monotonic() + timeout if timeout is not None else None

                _line_count = 0
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break  # EOF
                    if deadline is not None and time.monotonic() > deadline:
                            proc.kill()
                            f.write("[TIMEOUT]\n")
                            f.flush()
                            log.warning(f"[opencode_analyzer] timed out at action {action_num}")
                            return None

                    _line_count += 1
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        etype = event.get("type")
                        log.debug(f"[opencode_analyzer] event type={etype}")

                        if etype == "step_start":
                            # Capture session ID for resume
                            sid = event.get("sessionID")
                            if sid and not session_id_captured:
                                session_id_captured = sid

                        elif etype == "text":
                            text = event.get("part", {}).get("text", "")
                            if text:
                                accumulated_text += text
                                f.write(f"[ASSISTANT]\n{text}\n\n")

                        elif etype == "tool_use":
                            part = event.get("part", {})
                            tool_name = part.get("tool", "?")
                            state = part.get("state", {})
                            status = state.get("status", "?")
                            if status in ("running", "completed", "done"):
                                input_data = state.get("input", {})
                                input_str = json.dumps(input_data, indent=2) if isinstance(input_data, dict) else str(input_data)
                                f.write(f"[TOOL CALL: {tool_name}]\n{input_str}\n\n")
                            if status in ("completed", "done"):
                                output_data = state.get("output", "")
                                is_error = state.get("is_error", False) or state.get("error", False)
                                label = "[TOOL RESULT ERROR]" if is_error else "[TOOL RESULT]"
                                f.write(f"{label}\n{str(output_data)[:4000]}\n\n")

                        elif etype == "message.part.updated":
                            # Fallback for alternative event format
                            part = event.get("part", {})
                            ptype = part.get("type")
                            if ptype == "thinking" or ptype == "reasoning":
                                f.write(f"[THINKING]\n{part.get('text', '')}\n\n")
                            elif ptype == "tool":
                                name = part.get("name", "?")
                                state = part.get("state", "?")
                                if state == "running":
                                    input_data = part.get("input", {})
                                    if isinstance(input_data, dict):
                                        f.write(f"[TOOL CALL: {name}]\n{json.dumps(input_data, indent=2)}\n\n")
                                    else:
                                        f.write(f"[TOOL CALL: {name}]\n{input_data}\n\n")
                                elif state in ("completed", "done"):
                                    result_text = part.get("result", part.get("output", ""))
                                    if isinstance(result_text, str):
                                        text = result_text
                                    else:
                                        text = str(result_text)
                                    is_error = part.get("is_error", False) or part.get("error", False)
                                    label = "[TOOL RESULT ERROR]" if is_error else "[TOOL RESULT]"
                                    f.write(f"{label}\n{text[:4000]}\n\n")

                        elif etype == "error":
                            err = event.get("error", {})
                            err_name = err.get("name", "UnknownError")
                            err_data = err.get("data", {})
                            err_msg = err_data.get("message", str(err))
                            f.write(f"[ERROR: {err_name}]\n{err_msg}\n\n")
                            log.error(f"[opencode_analyzer] API error: {err_name}: {err_msg}")

                            # Context overflow → kill session so next call starts fresh
                            if "overflow" in err_name.lower() or "too long" in err_msg.lower():
                                log.warning(
                                    f"[opencode_analyzer] context overflow at action {action_num} "
                                    f"— clearing session ID to force fresh start"
                                )
                                with _session_lock:
                                    _session_ids.pop(log_path_key, None)
                                session_id_captured = None

                        elif etype == "step_finish":
                            part = event.get("part", {})
                            cost = part.get("cost")
                            f.write(f"[RESULT] cost=${cost}\n\n")

                        # --- Fallback: Claude-style events (in case OpenCode
                        #     mirrors Claude's format for some providers) ---
                        elif etype == "assistant":
                            for block in event.get("message", {}).get("content", []):
                                if block.get("type") == "thinking":
                                    f.write(f"[THINKING]\n{block.get('thinking', '')}\n\n")
                                elif block.get("type") == "text":
                                    text = block["text"]
                                    accumulated_text += text
                                    f.write(f"[ASSISTANT]\n{text}\n\n")
                                elif block.get("type") == "tool_use":
                                    f.write(f"[TOOL CALL: {block['name']}]\n{json.dumps(block.get('input', {}), indent=2)}\n\n")
                        elif etype == "user":
                            msg = event.get("message", {})
                            for block in msg.get("content", []):
                                if block.get("type") == "tool_result":
                                    content = block.get("content", "")
                                    if isinstance(content, str):
                                        text = content
                                    elif isinstance(content, list):
                                        text = "\n".join(
                                            c.get("text", "") for c in content
                                            if isinstance(c, dict) and c.get("type") == "text"
                                        )
                                    else:
                                        text = str(content)
                                    is_error = block.get("is_error", False)
                                    label = "[TOOL RESULT ERROR]" if is_error else "[TOOL RESULT]"
                                    f.write(f"{label}\n{text[:4000]}\n\n")
                        elif etype == "result":
                            # Use result text only as fallback — text events are the
                            # primary source across multi-step agent runs
                            result_text = event.get("result", "").strip()
                            if result_text and not accumulated_text.strip():
                                accumulated_text = result_text
                            cost = event.get("total_cost_usd")
                            f.write(f"[RESULT] cost=${cost}\n\n")

                        else:
                            # Unknown event type — log raw for debugging
                            f.write(f"[RAW:{etype}] {line[:500]}\n")

                        f.flush()
                    except json.JSONDecodeError:
                        f.write(f"[RAW] {line}\n")
                        f.flush()

                returncode = proc.wait()
                stderr_thread.join(timeout=5)
                stderr_text = "\n".join(_stderr_lines)
                if stderr_text:
                    f.write(f"\n--- STDERR ---\n{stderr_text}\n")
                    f.flush()

                # Fallback: export session to recover text lost to race condition.
                # Fires when: (1) no text at all, or (2) text exists but lacks [ACTIONS]
                _needs_recovery = (
                    not accumulated_text.strip()
                    or (action_mode and "[ACTIONS]" not in accumulated_text)
                )
                if _needs_recovery and session_id_captured:
                    try:
                        _export_container_path = "/workspace/_export.json"
                        _export_host_path = Path(sandbox_dir) / "_export.json"
                        _export = subprocess.run(
                            ["docker", "exec", _container_name, "sh", "-c",
                             f"opencode export {session_id_captured} > {_export_container_path} 2>/dev/null"],
                            capture_output=True, text=True, timeout=30,
                        )
                        if _export.returncode == 0 and _export_host_path.exists():
                            _export_data = json.loads(_export_host_path.read_text())
                            _msgs = _export_data.get("messages", [])
                            _recovered_text = ""
                            for _msg in reversed(_msgs):
                                _role = _msg.get("info", {}).get("role")
                                if _role == "assistant":
                                    for _part in _msg.get("parts", []):
                                        if _part.get("type") == "text":
                                            _candidate = _part.get("text", "").strip()
                                            if _candidate and "[ACTIONS]" in _candidate:
                                                _recovered_text = _candidate
                                                break
                                            elif _candidate and not _recovered_text:
                                                _recovered_text = _candidate
                                    if _recovered_text and "[ACTIONS]" in _recovered_text:
                                        break
                            if _recovered_text:
                                accumulated_text = _recovered_text
                                log.info(f"[opencode_analyzer] recovered {len(_recovered_text)} chars via session export")
                        else:
                            log.debug(f"[opencode_analyzer] export recovery: export failed rc={_export.returncode}")
                    except json.JSONDecodeError as _je:
                        log.debug(f"[opencode_analyzer] export recovery: JSON parse error: {_je}")
                    except Exception as _e:
                        log.debug(f"[opencode_analyzer] export recovery failed: {_e}")

                f.flush()

            hint = accumulated_text.strip() if accumulated_text.strip() else None

            if returncode != 0 or not hint:
                log.warning(
                    f"[opencode_analyzer] action={action_num} failed: "
                    f"rc={returncode}, hint_len={len(hint) if hint else 0}"
                )
                # Clear session on any failure so next call starts fresh
                if resume_session:
                    log.warning(
                        f"[opencode_analyzer] clearing session for "
                        f"{log_path_key} (was {'resume' if not is_first_call else 'fresh'})"
                    )
                    with _session_lock:
                        _session_ids.pop(log_path_key, None)
                return None

            # Store session ID for resume only after confirmed success
            if resume_session and session_id_captured:
                with _session_lock:
                    _session_ids[log_path_key] = session_id_captured

            log.info(f"[opencode_analyzer] action={action_num} OK ({len(hint)} chars)")
            return hint

        except Exception as e:
            log.error(f"[opencode_analyzer] unexpected error: {e}", exc_info=True)
            return None

    return hook
