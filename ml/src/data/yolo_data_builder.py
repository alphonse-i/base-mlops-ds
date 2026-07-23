"""
Rôle : Ce composant a deux responsabilités (Adapter Pattern). 
Premièrement, il récupère les données de manière intelligente : via le SDK Picsellia si on est sur le cloud, 
ou via le volume local si on est en mode hors-ligne sur le DGX Spark. 
Deuxièmement, il génère dynamiquement le fichier dataset.yaml qui est indispensable au fonctionnement d'Ultralytics YOLOv8.

"""

import os
import yaml
import logging
from src.data.base_data_builder import BaseDataBuilder
from src.core.config_parser import PipelineConfig

# Import optionnel pour éviter le crash en mode local pur
try:
    from picsellia import Client
except ImportError:
    Client = None

logger = logging.getLogger(__name__)

class YoloDataBuilder(BaseDataBuilder):
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tracking_mode = config.tracking.mode
        self.local_path = config.data.local_path

    def prepare_data(self) -> str:
        """Récupère les données et génère le fichier YAML pour YOLOv8."""
        dataset_dir = self._get_data_directory()
        yaml_path = self._generate_yolo_yaml(dataset_dir)
        return yaml_path

    def _get_data_directory(self) -> str:
        """Détermine d'où proviennent les données (Picsellia ou Local)."""
        if self.tracking_mode == "picsellia":
            logger.info("Récupération des données depuis Picsellia...")
            # Myoboku injecte ces variables
            api_token = os.environ.get("api_token")
            experiment_id = os.environ.get("experiment_id")
            
            client = Client(api_token=api_token)
            experiment = client.get_experiment_by_id(experiment_id)
            
            # Picsellia télécharge les données et reconstitue l'arborescence
            return experiment.download_data(destination_path="/data")
            
        elif self.tracking_mode == "local":
            logger.info(f"Utilisation du dataset local : {self.local_path}")
            if not self.local_path or not os.path.exists(self.local_path):
                raise FileNotFoundError(f"Chemin local invalide : {self.local_path}")
            return self.local_path

    def _generate_yolo_yaml(self, dataset_dir: str) -> str:
        """Génère le fichier dataset.yaml requis par Ultralytics."""
        logger.info("Génération du fichier dataset.yaml pour YOLOv8...")
        
        # Structure type attendue par YOLO. 
        yolo_config = {
            "path": dataset_dir,
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            # En production, ces noms pourraient être extraits dynamiquement
            # depuis Picsellia ou via la configuration YAML.
            "names": {
                0: "helmet",
                1: "person",
                2: "head"
            }
        }
        
        yaml_path = os.path.join(dataset_dir, "dataset.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(yolo_config, f, sort_keys=False)
            
        logger.info(f"✅ Fichier YOLO généré : {yaml_path}")
        return yaml_path