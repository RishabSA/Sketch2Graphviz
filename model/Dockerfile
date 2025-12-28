# docker buildx build --platform linux/amd64 -t rishabsa/sketch2graphviz:latest .
# docker push rishabsa/sketch2graphviz:latest

FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python depenencies
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

COPY . /app

# FastAPI
ENV HOST=0.0.0.0 PORT=8000

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]