import os
import glob
from picsellia.exceptions import ResourceNotFoundError
from scripts.base_ingestion import BasePicselliaIngestor

class YoloDetectionIngestor(BasePicselliaIngestor):
    def __init__(self, pipeline_run_id: str, base_path: str):
        super().__init__(
            dataset_name="Retail_Alcohol_Detector",
            pipeline_run_id=pipeline_run_id,
            description="Dataset pour la détection d'alcools (YOLO Format)"
        )
        self.images_path = os.path.join(base_path, "image")
        self.labels_path = os.path.join(base_path, "labels")
        
        # Mapping des classes YOLO
        self.yolo_classes = {
            0: "Aperol",
            1: "Campari",
            2: "Heineken",
        }

    def process_data(self):
        # 1. Création des labels
        for label_name in self.yolo_classes.values():
            try:
                self.dataset_version.get_label(label_name)
            except ResourceNotFoundError:
                self.dataset_version.create_label(name=label_name)

        # 2. Récupération & Upload des images
        all_image_paths = glob.glob(os.path.join(self.images_path, "*.jpg"))
        self.upload_images_to_datalake(all_image_paths)

        # 3. Annotations (Bounding Boxes)
        print("Application des annotations (Bounding Boxes)...")
        assets = self.dataset_version.list_assets()
        
        for asset in assets:
            txt_filename = asset.filename.replace('.jpg', '.txt')
            txt_filepath = os.path.join(self.labels_path, txt_filename)
            
            if os.path.exists(txt_filepath):
                annotation = asset.create_annotation()
                with open(txt_filepath, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center = float(parts[1]), float(parts[2])
                        norm_width, norm_height = float(parts[3]), float(parts[4])
                        
                        label_name = self.yolo_classes.get(class_id)
                        if label_name:
                            picsellia_label = self.dataset_version.get_label(label_name)
                            
                            # Conversion YOLO -> Pixels Picsellia
                            img_w, img_h = asset.width, asset.height
                            box_w = int(norm_width * img_w)
                            box_h = int(norm_height * img_h)
                            box_x = int((x_center * img_w) - (box_w / 2))
                            box_y = int((y_center * img_h) - (box_h / 2))
                            
                            annotation.create_rectangle(
                                x=box_x, y=box_y, w=box_w, h=box_h, label=picsellia_label
                            )
            else:
                print(f"⚠️ Aucun fichier de label trouvé pour {asset.filename}")

def ingest_detection_dataset(**kwargs):
    """Fonction appelée par le PythonOperator d'Airflow."""
    run_id = kwargs.get('run_id', 'manual-run')
    base_path = "/opt/airflow/datasets/detection"
    
    ingestor = YoloDetectionIngestor(pipeline_run_id=run_id, base_path=base_path)
    return ingestor.run()