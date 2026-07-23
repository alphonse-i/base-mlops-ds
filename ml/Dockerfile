FROM python:3.12-slim

WORKDIR /app

# Installation des dépendances système (OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==2.2.1

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev \
    && rm -rf /root/.cache/pypoetry

COPY src/ /app/src/
COPY config/ /app/config/

ENTRYPOINT ["python", "src/main.py"]
CMD ["--config", "config/detection_yolov8_config.yaml"]