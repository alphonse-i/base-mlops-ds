"""
Rôle :
Indispensable pour la phase de test et de développement. Ce tracker permet à un Data Scientist de lancer le code sur sa machine 
ou directement sur le DGX Spark sans dépendre de Picsellia ni nécessiter un accès internet. 
Il loggue simplement tout dans la console standard.

"""

import logging
import json
from typing import Dict, Any

from src.tracking.base_tracker import BaseTracker

logger = logging.getLogger(__name__)

class LocalTracker(BaseTracker):
    """
    Tracker de développement. N'envoie aucune donnée sur le réseau.
    Affiche les résultats dans la console (stdout).
    """
    def __init__(self):
        logger.info("✅ Tracker Local initialisé. (Mode hors-ligne)")

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        logger.info(f"📊 MÉTRIQUES LOCALES :\n{json.dumps(metrics, indent=2)}")

    def store_artifact(self, name: str, path: str) -> None:
        logger.info(f"📦 ARTEFACT LOCAL : Le modèle '{name}' est disponible ici -> {path}")

    def update_status(self, status: str) -> None:
        logger.info(f"🔄 STATUT DU RUN : {status}")