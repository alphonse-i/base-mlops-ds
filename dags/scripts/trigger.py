import os
from picsellia import Client

def trigger_training(**kwargs):
    # --- 1. RÉCUPÉRATION DU CONTEXTE AIRFLOW ---
    # L'ID du dataset est passé par le DAG via "op_kwargs"
    dataset_version_id = kwargs.get("dataset_version_id")
    # L'ID du run est passé automatiquement par Airflow car "provide_context=True"
    run_id = kwargs.get("run_id", "manual-run")
    
    # NOUVEAU : On récupère le chemin de la config depuis le DAG (ou on met une valeur par défaut)
    config_file_path = kwargs.get("config_file_path", "config/detection_yolov8_config.yaml")

    if not dataset_version_id:
        raise ValueError("L'ID de la DatasetVersion est introuvable. Vérifie le XCom de la tâche d'ingestion.")

    # --- 2. CONFIGURATION PICSELLIA ---
    api_token = os.environ.get("PICSELLIA_API_TOKEN")
    organization_name = os.environ.get("PICSELLIA_ORG_NAME")
    
    client = Client(api_token=api_token, organization_name=organization_name)
    project = client.get_project("Retail_Alcohol_Detection")

    # --- 3. LECTURE DU TAG DOCKER (MÉTHODE GITOPS) ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    version_file_path = os.path.join(current_dir, "image_version.txt")
    
    try:
        with open(version_file_path, "r") as f:
            docker_image_tag = f.read().strip()
        print(f"✅ Image Docker dynamique lue depuis GitOps : {docker_image_tag}")
    except FileNotFoundError:
        print("⚠️ Fichier image_version.txt introuvable. Vérifie ton pipeline GitHub Actions. Tag 'latest' appliqué par défaut.")
        docker_image_tag = "latest"

    # --- 4. CRÉATION DE L'EXPÉRIENCE ---
    # NOUVEAU : Nom générique car cela peut être YOLO ou ResNeXt
    experiment_name = f"Auto-Pipeline-Run-{run_id}"
    experiment = project.create_experiment(name=experiment_name)
    # Dans le cas où on cible spécifiquement le DGX Spark ! on le commente si on utilise le compute picsellia
    experiment.attach_compute_block("dgx-spark-cluster")
    print(f"Création de l'expérience '{experiment_name}' (ID: {experiment.id}).")

    # --- 5. ATTACHEMENT DU LIVRABLE DATA ---
    dataset_version = client.get_dataset_version_by_id(dataset_version_id)
    experiment.attach_dataset(name="train", dataset_version=dataset_version)
    print(f"Dataset {dataset_version_id} correctement attaché.")

    # --- 6. LOG DES PARAMÈTRES POUR LE CONTENEUR ---
    # NOUVEAU : On n'envoie plus les hyperparamètres ML.
    # On indique juste au conteneur quelle configuration YAML utiliser.
    experiment.log_parameters({
        "config_file": config_file_path,
        "docker_image": docker_image_tag  # Instruction essentielle pour Myoboku
    })

    # --- 7. DÉCLENCHEMENT DE L'ENTRAÎNEMENT ---
    experiment.update(status="PENDING")
    print(f"🚀 L'expérience {experiment.id} est en PENDING. En attente d'un GPU sur l'agent Myoboku...")
    
    return experiment.id