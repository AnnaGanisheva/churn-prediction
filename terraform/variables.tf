
variable "image_uri" {
  description = "Docker image URI (ECR або DockerHub)"
  type        = string
  default     = "546727414131.dkr.ecr.eu-central-1.amazonaws.com/churn-app:latest"
}
