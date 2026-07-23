"""
Rôle :
C'est ici que réside la véritable intégration avec ton Datalake et tes expériences Picsellia. 
Ce tracker récupère les identifiants injectés par Myoboku ou Airflow et communique avec l'API.

"""

import os
import logging
from typing import Dict, Any
from picsellia import Client

from src.tracking.base_tracker import BaseTracker

logger = logging.getLogger(__name__)

class PicselliaTracker(BaseTracker):
    """
    Implémentation du tracker pour la plateforme Picsellia.
    Se connecte à l'expérience en cours via les variables d'environnement.
    """
    def __init__(self):
        # Myoboku injecte ces variables automatiquement dans le conteneur
        api_token = os.environ.get("api_token")
        experiment_id = os.environ.get("experiment_id")
        
        if not api_token or not experiment_id:
            raise ValueError(
                "Impossible d'initialiser PicselliaTracker : "
                "Variables d'environnement 'api_token' ou 'experiment_id' manquantes."
            )
        
        self.client = Client(api_token=api_token)
        self.experiment = self.client.get_experiment_by_id(experiment_id)
        logger.info(f"✅ Connecté à Picsellia. Expérience ID : {experiment_id}")

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Envoie les métriques à l'expérience Picsellia."""
        for key, value in metrics.items():
            # Picsellia attend généralement des flottants natifs pour les courbes
            self.experiment.log(key, float(value), "line")
        logger.info("📊 Métriques envoyées à Picsellia.")

    def store_artifact(self, name: str, path: str) -> None:
        """Attache un fichier physique (ex: best.pt) à l'expérience."""
        if os.path.exists(path):
            self.experiment.store_artifact(name, path)
            logger.info(f"📦 Artefact '{name}' sauvegardé sur Picsellia.")
        else:
            logger.error(f"⚠️ Artefact introuvable au chemin : {path}")

    def update_status(self, status: str) -> None:
        """Met à jour le statut final de l'expérience."""
        self.experiment.update(status=status)
        logger.info(f"🔄 Statut de l'expérience mis à jour : {status}")