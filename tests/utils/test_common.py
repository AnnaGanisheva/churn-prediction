import pandas as pd
import yaml

from src.utils.common import prepare_train_val_data, read_yaml


def test_read_yaml(tmp_path):
    config_data = {"param1": 123, "param2": "abc"}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config_data))

    config = read_yaml(config_file)

    assert config["param1"] == 123
    assert config["param2"] == "abc"


def test_prepare_train_val_data(tmp_path):
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    train_df = pd.DataFrame({"feature": [1, 2], "Churn": [0, 1]})
    val_df = pd.DataFrame({"feature": [3, 4], "Churn": [1, 0]})
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    X_train, y_train, X_val, y_val = prepare_train_val_data(
        train_path=str(train_path), val_path=str(val_path), target="Churn"
    )

    assert list(X_train.columns) == ["feature"]
    assert list(y_train) == [0, 1]
    assert list(X_val["feature"]) == [3, 4]
    assert list(y_val) == [1, 0]
