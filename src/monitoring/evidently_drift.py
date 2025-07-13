import json
import os
from pathlib import Path

import pandas as pd
import psycopg
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report

from src.utils.common import read_yaml
from src.utils.logger import logger

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS evidently_reports (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT now(),
    report_json JSONB
);
"""


def get_db_connection(autocommit=True):
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "db"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "test"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "example"),
        autocommit=autocommit,
    )


def prep_db():
    with get_db_connection() as conn:
        conn.execute(CREATE_TABLE_SQL)


def save_report_to_db(report: Report):
    report_json = report.as_dict()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO evidently_reports (report_json) VALUES (%s)",
                (json.dumps(report_json),),
            )


def run_data_drift_report():
    logger.info("Start Evidently Data Drift Report generation")
    config = read_yaml(Path("src/config/config.yaml"))

    # load data
    reference_data = pd.read_csv(Path(config.data_paths.test_data))
    current_data = pd.read_csv(Path(config.data_paths.val_data))

    # generate report
    report = Report(
        metrics=[
            ColumnDriftMetric(column_name='Churn'),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ]
    )
    report.run(reference_data=reference_data, current_data=current_data)

    # save report
    report_path_html = Path(config.monitoring.evidently_report_html)
    report_path_html.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(config.monitoring.evidently_report_html)
    report.save_json(config.monitoring.evidently_report_json)
    save_report_to_db(report)
    logger.info("Report saved at %s", report_path_html.absolute())


def run_evidently_monitoring():
    prep_db()
    run_data_drift_report()


if __name__ == "__main__":
    run_evidently_monitoring()
