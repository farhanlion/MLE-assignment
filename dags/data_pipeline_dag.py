from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

with DAG(
    'data_pipeline_dag',
    default_args=default_args,
    description='Manual data pipeline for bronze, silver, and gold processing (with separate training task)',
    schedule_interval=None,   # disables automatic scheduling
    start_date=datetime(2025, 1, 1),  # safe placeholder
    catchup=False,             # ensures no backfill runs
) as dag:
    
    # pipeline start / orchestration chain
    pipeline_start = EmptyOperator(task_id="pipeline_start")

    # Bronze Processing
    bronze_layer = BashOperator(
        task_id='bronze_layer',
        bash_command="python /app/scripts/01_create_bronze_layer.py",
    )
   
    silver_layer = BashOperator(
        task_id='silver_layer',
        bash_command="python /app/scripts/02_create_silver_layer.py",
    )

    gold_layer = BashOperator(
        task_id='gold_layer',
        bash_command="python /app/scripts/03_create_gold_layer.py",
    )

    # Main pipeline chain
    pipeline_start >> bronze_layer >> silver_layer >> gold_layer

    # -------------------------
    # Independent training task
    # -------------------------
    # This task is intentionally NOT linked to the main pipeline above.
    # It runs the training script as a separate subtask and can be
    # triggered manually from the Airflow UI without affecting the
    # Bronze→Silver→Gold flow.
    train_model = BashOperator(
        task_id='train_model',
        bash_command="python /app/scripts/04_train_model.py",
    )

    # note: do NOT set dependencies between `train_model` and the main chain
    # e.g., don't do: bronze_layer >> train_model  (we intentionally avoid this)
