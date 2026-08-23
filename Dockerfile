# NOTE: this Dockerfile was written but could not be built or run against a
# real Docker daemon in the environment this project was developed in (no
# `docker` binary was available — see docs/security.md / SECURITY.md for
# what that means for the shell tool's sandboxing). It has not been tested.
# Treat it as a reasonable starting point, not a verified build.

FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock* README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

RUN uv pip install -e .

COPY benchmarks ./benchmarks
COPY experiments ./experiments
COPY scripts ./scripts

ENV ARL_DB_PATH=/data/arl.db
VOLUME ["/data"]

EXPOSE 8000

ENTRYPOINT ["uv", "run", "arl"]
CMD ["--help"]
