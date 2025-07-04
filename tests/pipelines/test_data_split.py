import pandas as pd

from src.pipelines.data_split import split_and_save_data


def test_split_and_save_data(tmp_path):
    df = pd.DataFrame({"feature": list(range(10)), "Churn": [0, 1] * 5})

    input_csv = tmp_path / "processed.csv"
    df.to_csv(input_csv, index=False)

    output_dir = tmp_path / "splits"

    split_and_save_data(
        input_path=input_csv,
        output_dir=output_dir,
        target_column="Churn",
        test_size=0.2,
        val_size=0.2,
        random_state=42,
    )

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    total_rows = len(df_train) + len(df_val) + len(df_test)
    assert total_rows == len(df)

    original_ratio = df["Churn"].mean()
    train_ratio = df_train["Churn"].mean()
    val_ratio = df_val["Churn"].mean()
    test_ratio = df_test["Churn"].mean()

    assert abs(train_ratio - original_ratio) < 0.2
    assert abs(val_ratio - original_ratio) < 0.2
    assert abs(test_ratio - original_ratio) < 0.2
