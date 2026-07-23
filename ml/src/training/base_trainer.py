"""
Rôle : Interface abstraite pour les algorithmes d'entraînement. 
Elle définit les trois actions fondamentales que tout modèle (YOLO, ResNeXt, etc.) 
doit savoir faire : s'entraîner, s'évaluer et se sauvegarder.

"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTrainer(ABC):
    """
    Interface abstraite pour encapsuler la logique d'entraînement d'un modèle ML.
    """
    
    @abstractmethod
    def train(self, data_path: str) -> None:
        """Déclenche la boucle d'entraînement du modèle."""
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Évalue le modèle et retourne un dictionnaire de métriques."""
        pass

    @abstractmethod
    def save_model(self) -> str:
        """Sauvegarde les poids du modèle et retourne le chemin physique de l'artefact."""
        pass