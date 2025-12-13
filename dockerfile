FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /ml_system

# Copy project files
COPY . .

# Configure uv to use a .venv in the workdir and add it to PATH
ENV UV_PROJECT_ENVIRONMENT=.venv \
    PATH="/ml_system/.venv/bin:${PATH}"

# Install Python dependencies from pyproject.toml
RUN uv sync --no-dev

# Expose application port
EXPOSE 8000

# Run FastAPI with uvicorn
# CMD ["fastapi", "dev", "src/stack/main.py", "--host", "0.0.0.0", "--port", "8000"]
CMD ["/bin/bash"]