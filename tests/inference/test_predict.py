from unittest.mock import MagicMock, patch

from box import ConfigBox

from src.inference.predict import load_model, predict


@patch("src.inference.predict.read_yaml")
@patch("mlflow.sklearn.load_model")
def test_load_model(mock_load_model, mock_read_yaml):
    mock_read_yaml.return_value = ConfigBox(
        {"model_registry": {"name": "test_model", "stage": "Production"}}
    )
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    model = load_model()

    mock_load_model.assert_called_once_with("models:/test_model/Production")
    assert model == mock_model


@patch("src.inference.predict.load_model")
def test_predict(mock_load_model):
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]
    mock_load_model.return_value = mock_model

    input_data = {"feature1": 10, "feature2": "A"}
    prediction, probability = predict(input_data)

    assert prediction == 1
    assert probability == 0.8
    mock_model.predict.assert_called_once()
    mock_model.predict_proba.assert_called_once()
