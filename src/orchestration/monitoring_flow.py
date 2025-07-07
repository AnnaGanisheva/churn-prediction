from prefect import flow, task

from src.monitoring.evidently_drift import run_data_drift_report


@task
def run_monitoring():
    run_data_drift_report()


@flow(name="monitoring-pipeline")
def monitoring_pipeline():
    run_monitoring()


if __name__ == "__main__":
    monitoring_pipeline()
