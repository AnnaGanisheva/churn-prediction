from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

from src.orchestration.train_flow import training_pipeline

Deployment.build_from_flow(
    flow=training_pipeline,
    name="daily-training",
    schedule=CronSchedule(cron="0 6 * * *"),
).apply()
