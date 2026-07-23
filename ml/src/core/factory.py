"""
Rôle :
Ce fichier contient les "Fabriques" (Factories). Son rôle est d'analyser la configuration (PipelineConfig) et 
de retourner les bonnes classes (Tracker, DataBuilder, Trainer) 
sans que le point d'entrée n'ait besoin de connaître les détails d'implémentation. 
C'est le cœur de l'approche "Agnostique".
"""

from typing import Any
from src.core.config_parser import PipelineConfig

# Imports des futurs Trackers
from src.tracking.picsellia_tracker import PicselliaTracker
from src.tracking.local_tracker import LocalTracker

# Imports des futurs Data Builders
from src.data.yolo_data_builder import YoloDataBuilder
from src.data.pytorch_data_builder import PyTorchDataBuilder

# Imports des futurs Trainers
from src.training.yolo_trainer import YoloTrainer
from src.training.pytorch_trainer import PyTorchTrainer

class TrackerFactory:
    @staticmethod
    def get_tracker(config: PipelineConfig) -> Any:
        if config.tracking.mode == "picsellia":
            return PicselliaTracker()
        elif config.tracking.mode == "local":
            return LocalTracker()
        else:
            raise ValueError(f"Mode de tracking inconnu : {config.tracking.mode}")

class DataFactory:
    @staticmethod
    def get_builder(config: PipelineConfig) -> Any:
        if config.framework == "yolo":
            return YoloDataBuilder(config)
        elif config.framework == "pytorch":
            return PyTorchDataBuilder(config)
        else:
            raise ValueError(f"Framework Data inconnu : {config.framework}")

class TrainerFactory:
    @staticmethod
    def get_trainer(config: PipelineConfig, tracker: Any) -> Any:
        if config.framework == "yolo":
            return YoloTrainer(config, tracker)
        elif config.framework == "pytorch":
            return PyTorchTrainer(config, tracker)
        else:
            raise ValueError(f"Framework ML inconnu : {config.framework}")