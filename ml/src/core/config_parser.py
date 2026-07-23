import yaml
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

"""
Rôle :
Lire le fichier YAML et le transformer en objets Python fortement typés 
(Dataclasses Pydantic).

"""

class TrackingConfig(BaseModel):
    """Configuration liée à l'outil de tracking et de MLOps."""
    mode: str = Field(
        ..., 
        pattern="^(picsellia|local)$", 
        description="Backend de tracking à utiliser."
    )

class DataConfig(BaseModel):
    """Configuration liée aux chemins et sources de données."""
    local_path: Optional[str] = Field(
        default=None, 
        description="Chemin local du dataset (utilisé principalement en mode local)."
    )

class PipelineConfig(BaseModel):
    """
    Modèle principal de configuration du pipeline d'entraînement.
    Valide l'ensemble des paramètres fournis dans le YAML.
    """
    task: str = Field(
        ..., 
        pattern="^(classification|detection)$", 
        description="Type de tâche ML à effectuer."
    )
    framework: str = Field(
        ..., 
        pattern="^(pytorch|yolo)$", 
        description="Framework d'entraînement cible."
    )
    model_name: str = Field(
        ..., 
        description="Nom de l'architecture ou des poids pré-entraînés (ex: yolov8s.pt, resnext50)."
    )
    tracking: TrackingConfig
    data: DataConfig
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Dictionnaire des hyperparamètres spécifiques au modèle."
    )

def load_config(config_path: str) -> PipelineConfig:
    """
    Charge, parse et valide le fichier de configuration YAML.
    
    Args:
        config_path (str): Le chemin vers le fichier .yaml
        
    Returns:
        PipelineConfig: L'objet de configuration validé.
        
    Raises:
        ValueError: Si le fichier est introuvable.
        ValidationError: Si le format YAML ne respecte pas le schéma attendu.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)
            
        if not raw_config:
            raise ValueError(f"Le fichier de configuration {config_path} est vide.")
            
        # Pydantic validera automatiquement les types et les valeurs (Regex)
        return PipelineConfig(**raw_config)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")