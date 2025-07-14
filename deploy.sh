#!/bin/bash

# === CONFIGURATION ===
#  Replace these values with your own if needed
AWS_REGION="eu-central-1"
AWS_ACCOUNT_ID="123456789012"  # Replace with your actual AWS Account ID
REPO_NAME="churn-app"
IMAGE_NAME="churn-app"
TAG="latest"
TERRAFORM_DIR="terraform"  # where your .tf files are

# === AUTO-GENERATED VALUES ===
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:${TAG}"

echo "STEP 0: Ensure ECR repository exists"
aws ecr describe-repositories \
  --repository-names "${REPO_NAME}" \
  --region "${AWS_REGION}" > /dev/null 2>&1

if [ $? -ne 0 ]; then
  echo "Repository not found. Creating it..."
  aws ecr create-repository --repository-name "${REPO_NAME}" --region "${AWS_REGION}"
else
  echo "Repository already exists."
fi

echo "STEP 1: Build Docker image for platform linux/amd64"
docker build --platform=linux/amd64 -t ${IMAGE_NAME} .

echo "STEP 2: Login to AWS ECR"
aws ecr get-login-password --region ${AWS_REGION} | \
docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo "STEP 3: Tag Docker image for ECR"
docker tag ${IMAGE_NAME}:latest ${ECR_URI}

echo "STEP 4: Push Docker image to ECR"
docker push ${ECR_URI}

echo "STEP 5: Terraform destroy old infrastructure"
cd ${TERRAFORM_DIR}
terraform destroy -auto-approve -var="image_uri=${ECR_URI}"

echo "STEP 6: Terraform apply new infrastructure"
terraform apply -auto-approve -var="image_uri=${ECR_URI}"

echo "DONE — check ECS for running task and public IP"
