import subprocess

from prefect import flow, task


@task(retries=3, retry_delay_seconds=2)
def ingest_data():
    subprocess.run(["python", "-m", "src.pipelines.data_split"], check=True)


@task
def split_data():
    subprocess.run(["python", "-m", "src.pipelines.data_split"], check=True)


@task(log_prints=True)
def train_model():
    subprocess.run(["python", "-m", "src.pipelines.train_optuna_rf"], check=True)


@task(log_prints=True)
def evaluate_model():
    subprocess.run(["python", "-m", "src.pipelines.evaluate_model"], check=True)


@flow(name="Churn Prediction Training Pipeline")
def training_pipeline():
    ingest_data()
    split_data()
    train_model()
    evaluate_model()


if __name__ == "__main__":
    training_pipeline()
