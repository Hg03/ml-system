# --- Stage 1: Base Setup (Alpine) ---
FROM python:3.13-alpine AS python_base
# Python optimizations
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
WORKDIR /ml_system
# --- Stage 2: Builder ---
FROM python_base AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev
# --- Stage 3: Dev Environment ---
FROM python_base AS dev
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Add common debug tools for local troubleshooting
RUN apk add --no-cache \
    curl \
    git \
    vim \
    wget \
    bind-tools \
    netcat-openbsd \
    procps \
    bash
WORKDIR /ml_system
COPY --from=builder /ml_system/.venv /ml_system/.venv
ENV PATH="/ml_system/.venv/bin:$PATH"
ENV PYTHONPATH="/ml_system/src"
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project
COPY . .
CMD ["/bin/bash"]
# --- Stage 4: Production (The Tiny Image) ---
FROM python_base AS prod
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apk add --no-cache bash git
WORKDIR /ml_system
COPY --from=builder /ml_system/.venv /ml_system/.venv
COPY . .
ENV PATH="/ml_system/.venv/bin:$PATH"
ENV PYTHONPATH="/ml_system/src"
CMD ["/bin/bash"]