# docker.io/rishabsa/sketch2graphviz:lates

# docker buildx build --platform linux/amd64 -t rishabsa/sketch2graphviz:latest .
# docker push rishabsa/sketch2graphviz:latest

FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

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

WORKDIR /app

# Python depenencies
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

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
