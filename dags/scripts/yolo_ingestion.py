import os
import glob
from picsellia.exceptions import ResourceNotFoundError
from scripts.base_ingestion import BasePicselliaIngestor

class YoloDetectionIngestor(BasePicselliaIngestor):
    def __init__(self, pipeline_run_id: str, base_path: str):
        # Récupère le nom du dataset depuis le .env (ou valeur par défaut)
        dataset_name = os.environ.get("PICSELLIA_DATASET_NAME", "Helmet_Detection_Dataset")
        
        super().__init__(
            dataset_name=dataset_name,
            pipeline_run_id=pipeline_run_id,
            description="Dataset de détection d'objets (Format YOLO) ingéré via Airflow"
        )
        self.images_path = os.path.join(base_path, "images")
        self.labels_path = os.path.join(base_path, "annotations")

    def process_data(self):
        # 1. Récupération de toutes les images brutes (.png, .jpg, etc.)
        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG")
        all_image_paths = []
        for ext in valid_exts:
            all_image_paths.extend(glob.glob(os.path.join(self.images_path, ext)))

        if not all_image_paths:
            raise FileNotFoundError(
                f"❌ Aucune image trouvée dans {self.images_path}. "
                f"Extensions vérifiées : {valid_exts}"
            )

        print(f"📤 Upload de {len(all_image_paths)} images vers le Datalake Picsellia...")
        self.upload_images_to_datalake(all_image_paths)

        # 2. Application des Bounding Boxes (Annotations)
        print("🏷️ Application des annotations depuis le dossier 'annotations'...")
        assets = self.dataset_version.list_assets()
        
        for asset in assets:
            base_filename = os.path.splitext(asset.filename)[0]
            txt_filepath = os.path.join(self.labels_path, f"{base_filename}.txt")
            
            if os.path.exists(txt_filepath):
                annotation = asset.create_annotation()
                with open(txt_filepath, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center = float(parts[1]), float(parts[2])
                        norm_w, norm_h = float(parts[3]), float(parts[4])
                        
                        # Création/récupération dynamique de la classe
                        label_name = f"class_{class_id}"
                        try:
                            picsellia_label = self.dataset_version.get_label(label_name)
                        except ResourceNotFoundError:
                            picsellia_label = self.dataset_version.create_label(name=label_name)
                            
                        # Conversion des coordonnées normalisées YOLO -> Pixels
                        img_w, img_h = asset.width, asset.height
                        box_w = int(norm_w * img_w)
                        box_h = int(norm_h * img_h)
                        box_x = int((x_center * img_w) - (box_w / 2))
                        box_y = int((y_center * img_h) - (box_h / 2))
                        
                        annotation.create_rectangle(
                            x=box_x, y=box_y, w=box_w, h=box_h, label=picsellia_label
                        )

        # 3. SPLIT AUTOMATIQUE PAR PICSELLIA (80% Train / 10% Val / 10% Test)
        print("📊 Application du split automatique (80/10/10) sur Picsellia...")
        self.dataset_version.split_repartition([0.8, 0.1, 0.1])
        print("✅ Ingestion et découpage terminés avec succès !")


def ingest_detection_dataset(**kwargs):
    """Fonction appelée directement par le PythonOperator d'Airflow."""
    run_id = kwargs.get('run_id', 'manual-run')
    # Chemin vers le dossier brut de données
    base_path = "/opt/airflow/datasets/detection"
    
    ingestor = YoloDetectionIngestor(pipeline_run_id=run_id, base_path=base_path)
    return ingestor.run()