"""
Ce fichier garantit que le "chef d'orchestre" (main.py) 
recevra toujours les bons composants selon la configuration demandée.
"""

import pytest
from src.core.config_parser import PipelineConfig, TrackingConfig, DataConfig
from src.core.factory import TrackerFactory, TrainerFactory, DataFactory
from src.tracking.local_tracker import LocalTracker
from src.training.yolo_trainer import YoloTrainer
from src.data.yolo_data_builder import YoloDataBuilder

@pytest.fixture
def yolo_local_config():
    """Fixture retournant une configuration YOLO en mode Local prête à l'emploi."""
    return PipelineConfig(
        task="detection",
        framework="yolo",
        model_name="yolov8s.pt",
        tracking=TrackingConfig(mode="local"),
        data=DataConfig(local_path="/tmp/data"),
        hyperparameters={}
    )

def test_tracker_factory_returns_local_tracker(yolo_local_config):
    """Vérifie que le mode 'local' retourne bien la classe LocalTracker."""
    tracker = TrackerFactory.get_tracker(yolo_local_config)
    assert isinstance(tracker, LocalTracker)

def test_trainer_factory_returns_yolo_trainer(yolo_local_config):
    """Vérifie que le framework 'yolo' retourne bien la classe YoloTrainer."""
    # Le Trainer a besoin d'un tracker pour s'instancier
    tracker = TrackerFactory.get_tracker(yolo_local_config)
    trainer = TrainerFactory.get_trainer(yolo_local_config, tracker)
    
    assert isinstance(trainer, YoloTrainer)
    assert trainer.model_name == "yolov8s.pt"

def test_data_factory_returns_yolo_builder(yolo_local_config):
    """Vérifie que le framework 'yolo' retourne bien le constructeur de données YOLO."""
    data_builder = DataFactory.get_builder(yolo_local_config)
    assert isinstance(data_builder, YoloDataBuilder)