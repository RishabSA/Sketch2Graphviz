# docker.io/rishabsa/sketch2graphviz:latest

# docker buildx build --platform linux/amd64 -t rishabsa/sketch2graphviz:latest .
# docker push rishabsa/sketch2graphviz:latest

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:/usr/local/bin:$PATH"

# System dependencies (postgresql install asks for timezone data so disable interactions so it runs on Docker build)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    graphviz \
    build-essential \
    postgresql \
    postgresql-contrib \
    postgresql-server-dev-all \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && make -C /tmp/pgvector \
    && make -C /tmp/pgvector install \
    && rm -rf /tmp/pgvector

RUN dot -V

# uv: copied from the official image (pin the tag for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependency layer: only invalidated when lockfile / pyproject / python pin changes.
# --no-install-project because the project has no package layout; app runs from /app directly.
COPY pyproject.toml uv.lock .python-version /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Application source (includes lora_checkpoints — kept in the image per project requirement)
COPY . /app

EXPOSE 8000

ENTRYPOINT ["bash", "-lc", "set -e; \
service postgresql start; \
service postgresql status; \
su - postgres -c \"psql -c \\\"CREATE ROLE root WITH LOGIN SUPERUSER;\\\"\"; \
su - postgres -c \"psql -c \\\"CREATE DATABASE sketch2graphvizdb;\\\"\"; \
su - postgres -c \"psql -d sketch2graphvizdb -c \\\"CREATE EXTENSION IF NOT EXISTS vector;\\\"\"; \
su - postgres -c \"psql -d sketch2graphvizdb -c \\\"\\\\dx\\\"\"; \
su - postgres -c \"psql -d sketch2graphvizdb -f /app/postgreSQL_data/sketch2graphvizdb.sql\"; \
exec \"$@\"", "bash"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
