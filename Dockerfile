FROM python:3.12-slim

# ffmpeg is required to extract audio
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# CPU engine. MLX (Apple GPU) cannot run inside Docker.
RUN pip install --no-cache-dir '.[faster]'

# Models are cached here; mount a volume so they download only once
ENV HF_HOME=/models
WORKDIR /work
ENTRYPOINT ["transcribe"]
CMD ["--help"]
