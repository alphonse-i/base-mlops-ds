#!/bin/bash

echo "🚀 Création de l'arborescence ml pipeline en cours..."

# 1. Création des dossiers et sous-dossiers (Ajout des accolades pour config et src)
mkdir -p {config,src/{core,tracking,data,training}}

# 2. Création des fichiers à la racine (Ajout des accolades)
touch {Dockerfile,pyproject.toml}

# 3. Création des fichiers de configuration
touch config/{classification_resnext_iconfig.yaml,detection_yolov8_config.yaml}

# 4. Création des fichiers sources
touch src/{__init__.py,main.py}
touch src/core/{__init__.py,config_parser.py,factory.py}
touch src/tracking/{__init__.py,base_tracker.py,picsellia_tracker.py,local_tracker.py}
touch src/data/{__init__.py,base_data_builder.py,yolo_data_builder.py,pytorch_data_builder.py}
touch src/training/{__init__.py,base_trainer.py,yolo_trainer.py,pytorch_trainer.py}

echo "✅ Arborescence ml_pipeline créée avec succès !"
