# Customer Churn Prediction

This project aims to build a machine learning system to predict customer churn based on telecom user data.  
It follows modern MLOps best practices — from data preparation and versioning to model training, experiment tracking, and deployment via an interactive Streamlit app.

---

## Goal  
Identify customers with a high risk of churn so the business can proactively retain them.

---

## Project Structure

```
churn-prediction/
├── data/              # Raw and processed data
├── notebooks/         # EDA and modeling experiments
├── src/               # Source code for training, evaluation, and preprocessing
├── scripts/           # Helper scripts for CLI usage or orchestration
├── streamlit_app/     # Streamlit UI for interactive prediction
├── tests/             # Unit tests and integration tests
├── requirements.txt   # Project dependencies
├── README.md          # Project overview
```

---

## Data Processing

- Dataset: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (Kaggle)
- Removed rows with missing `TotalCharges`
- Converted `TotalCharges` to numeric
- Dropped `customerID` (not informative)
- Target column `Churn` mapped: Yes → 1, No → 0

**Class imbalance:**  
The target variable is imbalanced (`Yes`: 26%, `No`: 74%) and this is taken into account during train/validation/test split.

**Data Split:**
- 60% training
- 20% validation
- 20% test  
(Stratified splitting to maintain target distribution)

**Feature Engineering**

- Numerical features → `StandardScaler`
- Categorical features → `OneHotEncoder`

All preprocessing is done via a `ColumnTransformer` inside a `Pipeline`. The trained model and preprocessing steps are saved together as `model_pipeline.pkl`.


---

## Model Training

Model is trained using Scikit-learn with Logistic Regression as a baseline. Pipeline and training logic are stored in `src/pipelines/train_model.py`.

Hyperparameter tuning and experiment tracking are integrated via MLflow (hosted on DagsHub).


---
## Model and Results

This project uses **Logistic Regression** as the baseline model for predicting customer churn. After tuning hyperparameters with Optuna and adjusting for class imbalance using `class_weight="balanced"`, the model achieved the following validation metrics:

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.773   |
| Precision  | 0.555   |
| Recall     | 0.730   |
| F1 Score   | 0.630   |
| ROC AUC    | 0.759   |

## Potential Improvements

As the primary focus of this project is to practice **MLOps skills**, not to optimize model accuracy, the training process was intentionally kept simple. However, model performance could be improved further through:

- Trying different classifiers (e.g., XGBoost, RandomForest)
- Addressing class imbalance with techniques like oversampling or undersampling
- Tuning the classification threshold (instead of using the default 0.5)
- Feature engineering or adding domain-specific features
- Using cross-validation during hyperparameter tuning

## Model Training and Optimization

This project uses both **Logistic Regression** and **Random Forest** classifiers to model churn prediction. We use **Optuna** for hyperparameter optimization and **MLflow** for experiment tracking.

### Why Optuna?

[Optuna](https://optuna.org/) is a modern, flexible hyperparameter optimization library that allows efficient searching via techniques like **Tree-structured Parzen Estimator (TPE)**. We chose it because of:
- automatic support for early stopping and pruning,
- clean integration with Python code,
- excellent visualization and analysis tools,
- easy nesting of MLflow logging inside optimization trials.

---

## Results

### Logistic Regression (with `class_weight="balanced"`)

**Best hyperparameters:**

| Parameter   | Value         |
|-------------|---------------|
| C           | 1.2481        |
| max_iter    | 833           |
| solver      | liblinear     |

**Metrics (Validation set):**

| Metric     | Value          |
|------------|----------------|
| accuracy   | 0.7726         |
| precision  | 0.5549         |
| recall     | 0.7299         |
| f1         | 0.6305         |
| roc_auc    | 0.7590         |

---

### Random Forest Classifier (with `class_weight="balanced"`)

**Best hyperparameters:**

| Parameter         | Value |
|-------------------|-------|
| n_estimators      | 184   |
| max_depth         | 15    |
| min_samples_split | 6     |
| min_samples_leaf  | 3     |
| max_features      | sqrt  |
| random_state      | 42    |

**Metrics (Validation set):**

| Metric     | Value          |
|------------|----------------|
| accuracy   | 0.7278         |
| precision  | 0.4928         |
| recall     | 0.8262         |
| f1         | 0.6174         |
| roc_auc    | 0.7592         |

---

These results show that Logistic Regression achieves a better **precision**, while Random Forest reaches a higher **recall**, which might be important if you want to reduce false negatives in churn prediction.

> ⚠️ Note: The goal of this project is not to maximize model performance, but to demonstrate **end-to-end MLOps workflow** including model versioning, configuration management, and reproducible pipelines.


## MLflow Tracking with DagsHub

This project uses **MLflow** for experiment tracking, integrated with [DagsHub](https://dagshub.com) as the remote tracking server. All models, parameters, and metrics are logged remotely and can be visualized in the DagsHub UI.

### Environment Variables

To use MLflow with DagsHub, create a `.env` file in the root of the project with the following variables:

```
MLFLOW_TRACKING_URI=https://dagshub.com/YOUR_USERNAME/YOUR_REPO_NAME.mlflow
MLFLOW_TRACKING_USERNAME=your-dagshub-username
MLFLOW_TRACKING_PASSWORD=your-dagshub-token
```

Replace `YOUR_USERNAME`, `YOUR_REPO_NAME`, and the credentials with your actual DagsHub account information.

### Python Integration

Make sure to install `python-dotenv`:

```
pip install python-dotenv
```

In your training scripts, load the environment variables with:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Token Generation

You can generate a personal access token from your [DagsHub profile](https://dagshub.com/settings/token). Use this token as the value for `MLFLOW_TRACKING_PASSWORD`.

### Notes

- All MLflow runs will be visible under the corresponding experiment in your DagsHub UI.
- If you use nested runs (e.g., for hyperparameter tuning with Optuna), they will also be grouped and logged properly.
- For optimal reproducibility and collaboration, do **not** commit your `.env` file to version control. Add it to `.gitignore`.


## Tech Stack
- Python 3.10
- Pandas, scikit-learn
- MLflow for experiment tracking
- DVC for data versioning
- Docker for containerization
- Streamlit for UI
- GitHub Actions (planned) for CI/CD

6. How to Run
7. Results / Metrics

8. Future Work
