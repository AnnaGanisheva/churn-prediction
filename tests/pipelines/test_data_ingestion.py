import pandas as pd

from src.pipelines.data_ingestion import ingest_data


def test_ingest_data(tmp_path):
    raw_data = pd.DataFrame(
        {
            "customerID": pd.Series(["123-ABC", "456-DEF"], dtype=str),
            "TotalCharges": pd.Series(["29.85", " "], dtype=str),
            "Churn": pd.Series(["No", "Yes"], dtype=str),
        }
    )
    raw_path = tmp_path / "raw.csv"
    raw_data.to_csv(raw_path, index=False)

    output_path = tmp_path / "processed" / "processed.csv"

    ingest_data(raw_path, output_path)

    assert output_path.exists()

    df = pd.read_csv(output_path)

    assert len(df) == 1
    assert df.iloc[0]["customerID"] == "123-ABC"
    assert df.iloc[0]["TotalCharges"] == 29.85
    assert df.iloc[0]["Churn"] == 0
