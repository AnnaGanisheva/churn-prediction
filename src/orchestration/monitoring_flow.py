from prefect import flow, task

from src.monitoring.evidently_drift import run_evidently_monitoring


@task
def run_monitoring():
    run_evidently_monitoring()


@flow(name="monitoring-pipeline")
def monitoring_pipeline():
    run_monitoring()


if __name__ == "__main__":
    monitoring_pipeline()
