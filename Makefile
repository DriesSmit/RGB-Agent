.DEFAULT_GOAL := run

.PHONY: install server stop-server check-server run

-include .env

UV ?= uv
GAME ?= memory-0001
MAX_ACTIONS ?= 200
MODEL ?= local-qwen
ENV_SOURCE ?= re_arc

SERVER_HOST ?= 0.0.0.0
SERVER_PORT ?= 1234
SERVER_START_TIMEOUT ?= 900
SERVER_STATUS_EVERY ?= 10
SERVER_LOG_TAIL_LINES ?= 20
SERVER_HEALTH_URL ?= http://127.0.0.1:$(SERVER_PORT)/v1/models

LOCAL_SERVER_MODEL ?= Qwen/Qwen3.5-0.8B
LOCAL_ANALYZER_BASE_URL ?= http://host.docker.internal:$(SERVER_PORT)/v1
LOCAL_ANALYZER_MODEL_ID ?= qwen3.5-0.8b

LOCAL_SERVER_DEVICE ?= auto
LOCAL_SERVER_DTYPE ?= auto
LOCAL_SERVER_MAX_TOKENS ?= 2048
LOCAL_SERVER_TEMPERATURE ?= 0.2
LOCAL_SERVER_TOP_P ?= 0.95

LOCAL_SERVER_LOG ?= /tmp/rgb-agent-local-server.log
LOCAL_SERVER_PID ?= /tmp/rgb-agent-local-server.pid
LOCAL_SERVER_REQUIREMENTS ?= local_server/requirements.txt

export LOCAL_ANALYZER_BASE_URL
export LOCAL_ANALYZER_MODEL_ID

install:
	$(UV) sync
	$(UV) pip install -r $(LOCAL_SERVER_REQUIREMENTS)
	cd docker/opencode-sandbox && bash build.sh

server:
	@if curl -fsS "$(SERVER_HEALTH_URL)" >/dev/null 2>&1; then \
		echo "Local model server already running at $(SERVER_HEALTH_URL)"; \
	else \
		echo "Starting local Qwen server for $(LOCAL_SERVER_MODEL)"; \
		echo "Logs: $(LOCAL_SERVER_LOG)"; \
		nohup $(UV) run --with-requirements $(LOCAL_SERVER_REQUIREMENTS) python -u -m local_server.server \
			--host "$(SERVER_HOST)" \
			--port "$(SERVER_PORT)" \
			--model "$(LOCAL_SERVER_MODEL)" \
			--model-id "$(LOCAL_ANALYZER_MODEL_ID)" \
			--device "$(LOCAL_SERVER_DEVICE)" \
			--dtype "$(LOCAL_SERVER_DTYPE)" \
			--max-tokens "$(LOCAL_SERVER_MAX_TOKENS)" \
			--temperature "$(LOCAL_SERVER_TEMPERATURE)" \
			--top-p "$(LOCAL_SERVER_TOP_P)" \
			>"$(LOCAL_SERVER_LOG)" 2>&1 & echo $$! >"$(LOCAL_SERVER_PID)"; \
		echo "PID: $$(cat "$(LOCAL_SERVER_PID)")"; \
	fi
	@for i in $$(seq 1 $(SERVER_START_TIMEOUT)); do \
		if curl -fsS "$(SERVER_HEALTH_URL)" >/dev/null 2>&1; then \
			echo "Local model server is ready at $(SERVER_HEALTH_URL)"; \
			exit 0; \
		fi; \
		if [ $$i -eq 1 ] || [ $$((i % $(SERVER_STATUS_EVERY))) -eq 0 ]; then \
			echo "[wait $$i/$(SERVER_START_TIMEOUT)s] waiting for $(SERVER_HEALTH_URL)"; \
			if [ -f "$(LOCAL_SERVER_LOG)" ]; then \
				echo "--- recent server log ---"; \
				tail -n $(SERVER_LOG_TAIL_LINES) "$(LOCAL_SERVER_LOG)" || true; \
				echo "--- end server log ---"; \
			fi; \
		fi; \
		sleep 1; \
	done; \
	echo "Timed out waiting for local model server. Check $(LOCAL_SERVER_LOG)"; \
	if [ -f "$(LOCAL_SERVER_LOG)" ]; then \
		echo "--- final server log ---"; \
		tail -n $(SERVER_LOG_TAIL_LINES) "$(LOCAL_SERVER_LOG)" || true; \
		echo "--- end server log ---"; \
	fi; \
	exit 1

stop-server:
	@if [ -f "$(LOCAL_SERVER_PID)" ]; then \
		if kill "$$(cat "$(LOCAL_SERVER_PID)")" >/dev/null 2>&1; then \
			rm -f "$(LOCAL_SERVER_PID)" && echo "Stopped local model server"; \
		else \
			rm -f "$(LOCAL_SERVER_PID)" && echo "Removed stale PID file"; \
		fi; \
	else \
		echo "No PID file at $(LOCAL_SERVER_PID)"; \
	fi

check-server:
	@if ! curl -fsS "$(SERVER_HEALTH_URL)" >/dev/null 2>&1; then \
		echo "Local model server is not reachable at $(SERVER_HEALTH_URL)"; \
		echo "Run 'make server' and inspect $(LOCAL_SERVER_LOG) if it fails."; \
		exit 1; \
	fi

run: server
	$(UV) run rgb-swarm --env-source $(ENV_SOURCE) --game $(GAME) --max-actions $(MAX_ACTIONS) --model $(MODEL)
