from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Import de tes scripts (assure-toi qu'ils sont dans le PYTHONPATH d'Airflow)
from scripts.pytorch_ingestion import ingest_classification_dataset
from scripts.trigger import trigger_training

# 1. Définition des paramètres par défaut du pipeline
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# 2. Instanciation du DAG (Le Pipeline)
with DAG(
    'pytorch_classification_pipeline',
    default_args=default_args,
    description='Pipeline MLOps : Ingestion QNAP et Déclenchement Picsellia via config YAML',
    #schedule_interval='@weekly', # Tourne une fois par semaine
    schedule_interval=None,
    catchup=False,
) as dag:

    # 3. TÂCHE 1 : L'Ingestion
    # La variable AIRFLOW_CTX_DAG_RUN_ID (utilisée dans ton script) est générée
    # automatiquement par Airflow à chaque exécution de ce DAG.
    ingestion_task = PythonOperator(
        task_id='ingest_data_from_qnap',
        python_callable=ingest_classification_dataset,
    )

    # 4. TÂCHE 2 : Le Déclenchement de l'entraînement
    trigger_task = PythonOperator(
        task_id='trigger_myoboku_training',
        python_callable=trigger_training,
        # On utilise XCom pour récupérer l'ID retourné par la Tâche 1
        op_kwargs={
            'dataset_version_id': "{{ ti.xcom_pull(task_ids='ingest_data_from_qnap') }}",
            # NOUVEAU : Injection du fichier de configuration YAML à utiliser pour ce run
            'config_file_path': 'config/classification_resnext_config.yaml'
        }
    )

    # 5. Définition de l'ordre d'exécution (Tâche 1 PUIS Tâche 2)
    ingestion_task >> trigger_task