.PHONY: all ingest split train_logreg train_rf train_all

# Run the full pipeline: data ingestion -> split -> training both models
all: ingest split train_all

# Download or ingest raw data
ingest:
	PYTHONPATH=. python src/pipelines/data_ingestion.py

# Split data into train/validation/test sets
split:
	PYTHONPATH=. python src/pipelines/data_split.py

# Train Logistic Regression with Optuna and track with MLflow
train_logreg:
	PYTHONPATH=. python src/pipelines/train_optuna_lr.py

# Train Random Forest with Optuna and track with MLflow
train_rf:
	PYTHONPATH=. python src/pipelines/train_optuna_rf.py

# Run both training scripts
train_all: train_logreg train_rf

# Evaluate the selected model from MLflow on the test set
evaluate:
	PYTHONPATH=. python src/pipelines/evaluate_model.py
