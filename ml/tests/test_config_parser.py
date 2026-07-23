"""
Ce fichier va s'assurer que ton parser lit bien les fichiers corrects, 
mais surtout qu'il bloque immédiatement toute erreur de syntaxe ou valeur inattendue (comme un framework non supporté)

"""

import pytest
import yaml
from pydantic import ValidationError
from src.core.config_parser import load_config

def test_load_valid_config(tmp_path):
    """Vérifie qu'une configuration YAML valide est correctement parsée en objet."""
    # 1. Arrange : Création d'un YAML temporaire valide
    config_data = {
        "task": "classification",
        "framework": "pytorch",
        "model_name": "resnext50",
        "tracking": {"mode": "local"},
        "data": {"local_path": "/tmp/data"},
        "hyperparameters": {"epochs": 10}
    }
    file_path = tmp_path / "valid_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config_data, f)
    
    # 2. Act : Chargement via notre parser
    config = load_config(str(file_path))
    
    # 3. Assert : Vérification des attributs typés
    assert config.framework == "pytorch"
    assert config.tracking.mode == "local"
    assert config.hyperparameters["epochs"] == 10

def test_load_invalid_framework_raises_error(tmp_path):
    """Vérifie que Pydantic bloque un framework non autorisé (ex: tensorflow)."""
    # 1. Arrange : Un YAML avec une erreur intentionnelle (tensorflow n'est pas permis par la Regex)
    config_data = {
        "task": "classification",
        "framework": "tensorflow", # <-- L'erreur est ici
        "model_name": "resnext50",
        "tracking": {"mode": "local"},
        "data": {"local_path": "/tmp/data"},
        "hyperparameters": {}
    }
    file_path = tmp_path / "invalid_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config_data, f)
    
    # 2 & 3. Act & Assert : On s'attend à ce qu'une erreur de validation explose
    with pytest.raises(ValidationError):
        load_config(str(file_path))

def test_missing_file_raises_error():
    """Vérifie qu'une erreur propre est levée si le fichier YAML n'existe pas."""
    with pytest.raises(FileNotFoundError):
        load_config("chemin/qui/n/existe/pas.yaml")