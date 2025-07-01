import os
import pandas as pd
from src.utils.common import read_yaml
from pathlib import Path
from src.utils.logger import logger


def ingest_data(data_path: Path, output_path: Path) -> None:
    """
    Ingests data from CSV file and save the processed data.
    """

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Clean TotalCharges (some have ' ')
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    
    # Map target variable
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # Ensure output folder exists

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    logger.info("Starting data ingestion pipeline...")
    config = read_yaml(Path("src/config/config.yaml"))
    raw_data_path = Path(config.data_paths.raw_data)
    output_data_path = Path(config.data_paths.processed_data)
    ingest_data(raw_data_path, output_data_path)
