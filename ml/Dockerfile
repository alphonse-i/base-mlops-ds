# MLOps_DS/Dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Installation de Poetry
RUN pip install --no-cache-dir poetry==2.2.1

# Copie des fichiers de configuration du package
COPY pyproject.toml poetry.lock* ./

# Installation des dépendances du projet
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copie du code métier
COPY src/ /app/src/
COPY config/ /app/config/

ENTRYPOINT ["python", "src/main.py"]
CMD ["--config", "config/detection_yolov8_config.yaml"]