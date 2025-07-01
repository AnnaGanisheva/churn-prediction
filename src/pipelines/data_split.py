import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.utils.common import read_yaml
from src.utils.logger import logger


def split_and_save_data(
    input_path: Path,
    output_dir: Path,
    target_column: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Splits the dataset into train, validation, and test sets with stratification.
    Saves the resulting files to the output directory.
    """
    logger.info(f"Reading cleaned data from {input_path}")
    df = pd.read_csv(input_path)

    logger.info("Performing train/test split...")
    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_column],
        random_state=random_state,
    )

    val_relative_size = val_size / (1 - test_size)
    logger.info("Performing train/val split...")
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_relative_size,
        stratify=df_train_val[target_column],
        random_state=random_state,
    )

    os.makedirs(output_dir, exist_ok=True)
    df_train.to_csv(output_dir / "train.csv", index=False)
    df_val.to_csv(output_dir / "val.csv", index=False)
    df_test.to_csv(output_dir / "test.csv", index=False)

    logger.info(f"Train/Val/Test data saved in {output_dir}")


if __name__ == "__main__":
    logger.info("Starting data splitting pipeline...")
    config = read_yaml(Path("src/config/config.yaml"))
    input_path = Path(config.data_paths.processed_data)
    output_dir = Path(config.data_paths.split_data_dir)
    target_column = config.data_params.target

    split_and_save_data(
        input_path=input_path,
        output_dir=output_dir,
        target_column=target_column,
        test_size=config.data_params.test_size,
        val_size=config.data_params.val_size,
        random_state=config.data_params.random_state,
    )
