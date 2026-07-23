import os
import glob
from picsellia.exceptions import ResourceNotFoundError
from scripts.base_ingestion import BasePicselliaIngestor

class PyTorchClassificationIngestor(BasePicselliaIngestor):
    def __init__(self, pipeline_run_id: str, base_path: str):
        super().__init__(
            dataset_name="Retail_Alcohol_Classifier",
            pipeline_run_id=pipeline_run_id,
            description="Dataset pour la classification d'alcools (PyTorch Format)"
        )
        self.base_path = base_path

    def process_data(self):
        local_mapping = {} 
        all_image_paths = []
        
        # 1. Découverte dynamique
        dataset_folders = [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]
        
        for dataset_folder in dataset_folders:
            dataset_folder_path = os.path.join(self.base_path, dataset_folder)
            label_folders = [d for d in os.listdir(dataset_folder_path) if os.path.isdir(os.path.join(dataset_folder_path, d))]
            
            for label_name in label_folders:
                try:
                    self.dataset_version.get_label(label_name)
                except ResourceNotFoundError:
                    self.dataset_version.create_label(name=label_name)
                    
                images_in_folder = glob.glob(os.path.join(dataset_folder_path, label_name, "*.*"))
                all_image_paths.extend(images_in_folder)
                
                for img_path in images_in_folder:
                    local_mapping[os.path.basename(img_path)] = label_name

        # 2. Upload
        self.upload_images_to_datalake(all_image_paths)

        # 3. Annotations (Classification)
        print("Application des classifications en base de données...")
        assets = self.dataset_version.list_assets()
        for asset in assets:
            label_name = local_mapping.get(asset.filename)
            if label_name:
                picsellia_label = self.dataset_version.get_label(label_name)
                annotation = asset.create_annotation()
                annotation.create_classification(label=picsellia_label)

def ingest_classification_dataset(**kwargs):
    """Fonction appelée par le PythonOperator d'Airflow."""
    run_id = kwargs.get('run_id', 'manual-run')
    base_path = "/opt/airflow/datasets/classification"
    
    ingestor = PyTorchClassificationIngestor(pipeline_run_id=run_id, base_path=base_path)
    return ingestor.run()