from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
from airflow.utils.task_group import TaskGroup

DUMMY_DATE = "2016-04-01"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id="scheduled_training_dag",
    description="One-shot training using today's date as a dummy cutoff.",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),  # any past date is fine
    schedule_interval="@once",        # run once (trigger when ready)
    catchup=False,
) as dag:

    start = EmptyOperator(task_id="start")

    # Bronze Processing
    inference = BashOperator(
        task_id='inference',
        bash_command="python /app/scripts/05_run_inference.py",
    )
   
    with TaskGroup(group_id="model_monitoring") as monitoring:
        model_performance = BashOperator(
            task_id="check_model_performance",
            bash_command=(
                f"python /app/scripts/06_model_performance_monitoring.py "
                f"--train_date {DUMMY_DATE}"
            )
        )
        data_drift = BashOperator(
            task_id="check_data_drift",
            bash_command=(
                f"python /app/scripts/07_data_drift_monitoring.py "
                f"--train_date {DUMMY_DATE}"
            )
        )

    end = EmptyOperator(task_id="end")

    start >> inference >> monitoring >> end