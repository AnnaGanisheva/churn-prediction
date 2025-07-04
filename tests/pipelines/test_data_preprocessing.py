import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.pipelines.data_preprocessing import build_preprocessing_pipeline


def test_build_preprocessing_pipeline():
    df = pd.DataFrame(
        {
            "age": [25, np.nan, 35],
            "salary": [50000, 60000, np.nan],
            "gender": ["male", "female", np.nan],
            "Churn": [0, 1, 0],
        }
    )

    pipeline = build_preprocessing_pipeline(df, target_column="Churn")

    assert isinstance(pipeline, ColumnTransformer)

    transformed = pipeline.fit_transform(df.drop(columns=["Churn"]))

    assert transformed.shape[0] == 3
    assert transformed.shape[1] >= 3
