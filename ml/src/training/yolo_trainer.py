"""
Rôle : C'est l'encapsulation (Wrapper) de la librairie ultralytics. 
Cette classe instancie le modèle YOLOv8, lit les hyperparamètres depuis la configuration, et surtout, 
elle injecte notre Tracker directement dans les callbacks natifs de YOLO pour monitorer l'entraînement en temps réel sans polluer le code.

"""

import logging
from typing import Dict, Any
from ultralytics import YOLO

from src.training.base_trainer import BaseTrainer
from src.core.config_parser import PipelineConfig
from src.tracking.base_tracker import BaseTracker

logger = logging.getLogger(__name__)

class YoloTrainer(BaseTrainer):
    def __init__(self, config: PipelineConfig, tracker: BaseTracker):
        self.config = config
        self.tracker = tracker
        self.model_name = config.model_name
        self.model = None

    def train(self, data_path: str) -> None:
        """Entraîne le modèle YOLOv8."""
        logger.info(f"Initialisation du modèle YOLO avec les poids : {self.model_name}")
        self.model = YOLO(self.model_name)
        
        # Injection du tracker via les callbacks d'Ultralytics
        self._setup_callbacks()
        
        # Extraction des hyperparamètres depuis la configuration
        epochs = self.config.hyperparameters.get("epochs", 20)
        batch_size = self.config.hyperparameters.get("batch_size", 16)
        imgsz = self.config.hyperparameters.get("imgsz", 640)
        
        logger.info(f"Démarrage de l'entraînement YOLOv8 ({epochs} epochs)...")
        self.model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project="runs", # Dossier de sortie standard de YOLO
            name="yolo_train"
        )
        logger.info("✅ Entraînement terminé.")

    def evaluate(self) -> Dict[str, Any]:
        """Évalue le modèle sur le jeu de validation et extrait les métriques clés."""
        logger.info("Évaluation du modèle...")
        # L'appel à model.val() retourne un objet metrics contenant les résultats
        metrics = self.model.val()
        
        # Formatage des métriques pour le tracking (Picsellia ou Local)
        results = {
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
        }
        logger.info("✅ Évaluation terminée.")
        return results

    def save_model(self) -> str:
        """Retourne le chemin des meilleurs poids sauvegardés par YOLO."""
        weights_path = "runs/yolo_train/weights/best.pt"
        logger.info(f"Poids du modèle prêts pour la sauvegarde : {weights_path}")
        return weights_path

    def _setup_callbacks(self) -> None:
        """
        Configure les callbacks natifs de YOLO pour envoyer les métriques 
        à Picsellia à la fin de chaque epoch.
        """
        def on_train_epoch_end(trainer):
            # Ultralytics stocke les pertes de l'epoch dans trainer.loss_items
            # Note: selon la version d'Ultralytics, l'extraction exacte peut varier
            if hasattr(trainer, "loss_items"):
                metrics = {
                    "train_loss": float(trainer.loss_items[0])
                }
                self.tracker.log_metrics(metrics)
                
        self.model.add_callback("on_train_epoch_end", on_train_epoch_end)