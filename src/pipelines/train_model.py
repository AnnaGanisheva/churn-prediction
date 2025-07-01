import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pathlib import Path

from src.pipelines.data_preprocessing import build_preprocessing_pipeline
from src.utils.common import read_yaml
from src.utils.logger import logger


def train():
    logger.info("Starting training pipeline...")

    # Load config
    config = read_yaml(Path("src/config/config.yaml"))
    train_path = Path(config.data_paths.train_data)
    model_path = Path(config.model_paths.model)
    preprocessor_path = Path(config.model_paths.preprocessor)

    # Load data
    df = pd.read_csv(train_path)
    logger.info(f"Loaded training data: {df.shape}")

    # Split X/y
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(df, target_column="Churn")

    # Build full model pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # Fit
    pipeline.fit(X, y)
    logger.info("Training complete.")

    # Save pipeline (model + preprocessing)
    # Ensure output folder exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info(f"Model pipeline saved to {model_path}")

    # Save just the preprocessor if needed
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")


if __name__ == "__main__":
    train()
