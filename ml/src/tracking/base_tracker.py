"""
Rôle :
Ce fichier définit le "contrat" (Interface) que tous les trackers devront respecter. 
Il garantit que le script main.py et les différents Trainers pourront appeler log_metrics 
ou update_status de la même manière, que l'on soit connecté à Picsellia ou hors-ligne.

"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTracker(ABC):
    """
    Interface abstraite définissant le comportement standard d'un outil 
    de tracking (Picsellia, MLflow, Local, etc.).
    """

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Enregistre un dictionnaire de métriques."""
        pass

    @abstractmethod
    def store_artifact(self, name: str, path: str) -> None:
        """Sauvegarde un fichier (ex: poids du modèle) comme artefact."""
        pass

    @abstractmethod
    def update_status(self, status: str) -> None:
        """Met à jour le statut du run (ex: SUCCESS, FAILED)."""
        pass