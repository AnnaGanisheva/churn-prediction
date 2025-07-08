#!/bin/bash
echo
sleep 10

echo "train models and register..."
make all

echo "Occhestrate Prefect..."
make orchestration

echo "Run Streamlit..."
exec streamlit run src/inference/app.py
