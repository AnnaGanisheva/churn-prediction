import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils.logger import logger


def build_preprocessing_pipeline(df: pd.DataFrame, target_column: str):
    """
    Builds preprocessing pipeline for numerical and categorical features.
    Returns sklearn ColumnTransformer.
    """

    logger.info("Building preprocessing pipeline...")

    # Drop target for feature analysis
    X = df.drop(columns=[target_column])

    # Split column types
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    logger.info(f"Numerical features: {num_features}")
    logger.info(f"Categorical features: {cat_features}")

    # Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    logger.info("Preprocessing pipeline built successfully.")
    return preprocessor
