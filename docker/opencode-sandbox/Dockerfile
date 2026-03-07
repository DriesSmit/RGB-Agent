# Minimal sandbox image for running the opencode analyzer CLI.
# Build:  docker build -t arcgym/opencode-sandbox:latest .

# ---- Stage 1: fetch the opencode binary ----
FROM python:3.12-slim AS fetcher

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://opencode.ai/install | bash \
    && mv /root/.opencode/bin/opencode /usr/local/bin/opencode \
    && chmod +x /usr/local/bin/opencode

# ---- Stage 2: minimal runtime ----
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 -s /bin/false opencode

COPY --from=fetcher /usr/local/bin/opencode /usr/local/bin/opencode

USER opencode
WORKDIR /workspace
ENTRYPOINT ["opencode"]
