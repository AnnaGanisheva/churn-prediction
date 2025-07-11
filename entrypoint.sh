#!/bin/bash
echo
sleep 10

echo "Train models and register..."
make all

echo "Orchestrate Prefect..."
make orchestration

echo "Starting Prefect Agent..."
prefect agent start --pool default-agent-pool --work-queue default &

echo "Run Streamlit..."
exec streamlit run src/inference/app.py
