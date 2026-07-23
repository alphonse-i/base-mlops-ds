"""
Rôle : Encapsuler la boucle d'entraînement personnalisée (forward pass, backward pass, optimisation) 
pour le modèle de classification ResNeXt50. Contrairement à YOLO qui masque cette complexité, 
ce script gère le passage des tenseurs sur le GPU (ou CPU).

"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from typing import Dict, Any

from src.training.base_trainer import BaseTrainer
from src.core.config_parser import PipelineConfig
from src.tracking.base_tracker import BaseTracker

logger = logging.getLogger(__name__)

class PyTorchTrainer(BaseTrainer):
    def __init__(self, config: PipelineConfig, tracker: BaseTracker):
        self.config = config
        self.tracker = tracker
        self.model_name = config.model_name
        self.model = None
        # Détection automatique du matériel disponible (GPU NVIDIA DGX ou CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_model_path = "/tmp/resnext50_best.pth"
        self.last_metrics = {}

    def train(self, data_path: str) -> None:
        """Déclenche la boucle d'entraînement PyTorch."""
        # 1. Extraction des hyperparamètres depuis la configuration YAML
        epochs = int(self.config.hyperparameters.get("epochs", 10))
        batch_size = int(self.config.hyperparameters.get("batch_size", 16))
        im_size = int(self.config.hyperparameters.get("im_size", 224))
        learning_rate = float(self.config.hyperparameters.get("learning_rate", 0.001))

        # 2. Préparation des données (Data Pipeline)
        logger.info(f"Chargement des données PyTorch depuis {data_path}...")
        tfs = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = datasets.ImageFolder(root=data_path, transform=tfs)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 3. Initialisation du modèle
        logger.info(f"Initialisation du modèle {self.model_name} sur : {self.device}")
        # On peut rendre le choix du modèle dynamique basé sur self.model_name
        self.model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        
        # Adaptation de la couche de sortie au nombre de classes du dataset
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
        self.model = self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 4. Boucle d'entraînement
        logger.info(f"Démarrage de l'entraînement ({epochs} epochs)...")
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_dl:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_dl)
            epoch_acc = correct / total
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
            
            # Monitoring via la couche d'abstraction (PicselliaTracker ou LocalTracker)
            metrics = {
                "train_loss": float(epoch_loss),
                "train_accuracy": float(epoch_acc)
            }
            self.tracker.log_metrics(metrics)
            self.last_metrics = metrics

        # Sauvegarde de l'artefact à la fin de l'entraînement
        torch.save(self.model.state_dict(), self.best_model_path)
        logger.info("✅ Entraînement terminé.")

    def evaluate(self) -> Dict[str, Any]:
        """Évalue le modèle et retourne un dictionnaire de métriques."""
        # Note MLOps : L'ancien script ne possédait pas de boucle de validation dédiée.
        # Dans un environnement de production complet, on implémenterait ici une boucle 
        # model.eval() avec torch.no_grad() sur un validation_dataloader.
        # Pour maintenir la parité avec ton code actuel, nous renvoyons les métriques de la dernière epoch.
        logger.info("Récupération des métriques finales...")
        return self.last_metrics

    def save_model(self) -> str:
        """Retourne le chemin absolu du modèle sauvegardé."""
        return self.best_model_path