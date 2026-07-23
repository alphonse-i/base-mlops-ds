"""


"""

import os
from abc import ABC, abstractmethod
from picsellia import Client
from picsellia.exceptions import ResourceNotFoundError

class BasePicselliaIngestor(ABC):
    """
    Classe de base pour l'ingestion de datasets vers Picsellia.
    Gère l'initialisation du client, la création/récupération des datasets 
    et l'upload massif des images.
    """
    def __init__(self, dataset_name: str, pipeline_run_id: str, description: str):
        self.api_token = os.environ.get("PICSELLIA_API_TOKEN")
        self.organization_name = os.environ.get("PICSELLIA_ORG_NAME")
        
        self.dataset_name = dataset_name
        self.version_name = f"v-{pipeline_run_id}"
        self.description = description
        
        if not self.api_token or not self.organization_name:
            raise ValueError("Variables d'environnement PICSELLIA_API_TOKEN ou PICSELLIA_ORG_NAME manquantes.")
            
        self.client = Client(api_token=self.api_token, organization_name=self.organization_name)
        self.dataset = None
        self.dataset_version = None

    def setup_dataset(self):
        """Idempotence : récupère ou crée le dataset et sa version."""
        try:
            self.dataset = self.client.get_dataset(self.dataset_name)
            print(f"Dataset '{self.dataset_name}' existant récupéré.")
        except ResourceNotFoundError:
            self.dataset = self.client.create_dataset(name=self.dataset_name, description=self.description)
            print(f"Dataset '{self.dataset_name}' créé.")

        try:
            self.dataset_version = self.dataset.get_version(self.version_name)
            print(f"DatasetVersion '{self.version_name}' existante récupérée.")
        except ResourceNotFoundError:
            print(f"Création de la DatasetVersion : {self.version_name}")
            self.dataset_version = self.dataset.create_version(
                version=self.version_name,
                description=self.description
            )

    def upload_images_to_datalake(self, image_paths: list):
        """Upload les images physiquement vers le Datalake et les attache à la version."""
        print(f"Upload total de {len(image_paths)} images...")
        if image_paths:
            lake = self.client.get_datalake()
            print("Upload des images vers le Datalake Picsellia...")
            uploaded_data = lake.upload_data(filepaths=image_paths)
            
            print("Liaison des images à la DatasetVersion...")
            self.dataset_version.add_data(uploaded_data)

    @abstractmethod
    def process_data(self):
        """
        Méthode abstraite. Doit découvrir les images, créer les labels,
        déclencher l'upload et appliquer les annotations spécifiques.
        """
        pass

    def run(self) -> str:
        """Point d'entrée principal (Template Method Pattern)."""
        self.setup_dataset()
        self.process_data()
        print(f"✅ Ingestion terminée avec succès pour {self.dataset_name} !")
        return self.dataset_version.id