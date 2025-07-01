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
