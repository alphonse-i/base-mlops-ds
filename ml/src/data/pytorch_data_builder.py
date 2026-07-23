import os
import logging
from src.data.base_data_builder import BaseDataBuilder
from src.core.config_parser import PipelineConfig

# Import optionnel pour éviter le crash en mode local pur
try:
    from picsellia import Client
except ImportError:
    Client = None

logger = logging.getLogger(__name__)

class PyTorchDataBuilder(BaseDataBuilder):
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tracking_mode = config.tracking.mode
        self.local_path = config.data.local_path

    def prepare_data(self) -> str:
        """
        Récupère les données et retourne le chemin vers le dossier racine
        prêt à être consommé par torchvision.datasets.ImageFolder.
        """
        dataset_dir = self._get_data_directory()
        logger.info(f"✅ Données PyTorch prêtes dans le dossier : {dataset_dir}")
        return dataset_dir

    def _get_data_directory(self) -> str:
        """Détermine d'où proviennent les données (Picsellia ou Local)."""
        if self.tracking_mode == "picsellia":
            logger.info("Récupération des données depuis Picsellia (Format Classification)...")
            # Myoboku injecte ces variables
            api_token = os.environ.get("api_token")
            experiment_id = os.environ.get("experiment_id")
            
            if not api_token or not experiment_id:
                raise ValueError("Variables 'api_token' ou 'experiment_id' manquantes pour le mode Picsellia.")
            
            client = Client(api_token=api_token)
            experiment = client.get_experiment_by_id(experiment_id)
            
            # Picsellia télécharge les données et reconstitue l'arborescence (train, val, test)
            return experiment.download_data(destination_path="/data")
            
        elif self.tracking_mode == "local":
            logger.info(f"Utilisation du dataset local PyTorch : {self.local_path}")
            if not self.local_path or not os.path.exists(self.local_path):
                raise FileNotFoundError(f"Chemin local invalide : {self.local_path}")
            return self.local_path
        else:
            raise ValueError(f"Mode de tracking inconnu : {self.tracking_mode}")