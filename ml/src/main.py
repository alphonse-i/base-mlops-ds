"""
Rôle :
C'est le chef d'orchestre. 
Il ne contient aucune logique métier (pas de PyTorch, pas de YOLO, pas d'API Picsellia). 
Il définit uniquement les étapes du pipeline MLOps.
"""

import argparse
import sys
import logging

from src.core.config_parser import load_config
from src.core.factory import TrackerFactory, DataFactory, TrainerFactory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # 1. Récupération du chemin de la configuration via la ligne de commande
    parser = argparse.ArgumentParser(description="Lancement du pipeline MLOps")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Chemin vers le fichier YAML de configuration"
    )
    args = parser.parse_args()

    try:
        # 2. Chargement et validation de la configuration
        logger.info(f"Chargement de la configuration : {args.config}")
        config = load_config(args.config)

        # 3. Initialisation des composants via les Factories
        logger.info("Initialisation des composants du pipeline...")
        tracker = TrackerFactory.get_tracker(config)
        data_builder = DataFactory.get_builder(config)
        trainer = TrainerFactory.get_trainer(config, tracker)

        # 4. Exécution du Pipeline Standard
        logger.info("Démarrage de la préparation des données...")
        dataset_path = data_builder.prepare_data()

        logger.info("Démarrage de l'entraînement...")
        trainer.train(dataset_path)

        logger.info("Démarrage de l'évaluation...")
        metrics = trainer.evaluate()
        tracker.log_metrics(metrics)

        logger.info("Sauvegarde des artefacts...")
        model_path = trainer.save_model()
        tracker.store_artifact("model-weights", model_path)

        # 5. Clôture
        tracker.update_status("SUCCESS")
        logger.info("Pipeline exécuté avec succès.")

    except Exception as e:
        logger.error(f"Erreur fatale lors de l'exécution du pipeline : {e}")
        # Si le tracker a eu le temps d'être instancié, on prévient Picsellia de l'échec
        if 'tracker' in locals() and hasattr(tracker, 'update_status'):
            tracker.update_status("FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()