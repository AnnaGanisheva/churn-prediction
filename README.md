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

## Tech Stack
- Python 3.10
- Pandas, scikit-learn
- MLflow for experiment tracking
- DVC for data versioning
- Docker for containerization
- Streamlit for UI
- GitHub Actions (planned) for CI/CD
