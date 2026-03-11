.PHONY: install run

UV ?= uv
GAME ?= memory-0001
MAX_ACTIONS ?= 200
MODEL ?= local-qwen
ENV_SOURCE ?= re_arc

install:
	$(UV) sync
	cd docker/opencode-sandbox && bash build.sh

run:
	$(UV) run rgb-swarm --env-source $(ENV_SOURCE) --game $(GAME) --max-actions $(MAX_ACTIONS) --model $(MODEL)
