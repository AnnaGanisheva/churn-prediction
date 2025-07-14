# MLFLOW on Fargate with S3

# S3 for Artifacts

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "mlflow-artifacts-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "mlflow" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  block_public_acls   = true
  block_public_policy = true
  ignore_public_acls  = true
  restrict_public_buckets = true
}


# IAM Role

data "aws_caller_identity" "current" {}

data "aws_iam_policy_document" "ecs_task_assume" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "mlflow_task_execution_role" {
  name               = "mlflowTaskExecutionRole"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
}

resource "aws_iam_role_policy_attachment" "mlflow_execution_policy" {
  role       = aws_iam_role.mlflow_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_policy" "mlflow_s3_policy" {
  name        = "mlflow-s3-access"
  description = "Allow MLflow to use S3 bucket"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = ["s3:*"],
        Effect = "Allow",
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_policy_attachment" "mlflow_attach_policy" {
  name       = "mlflow-s3-access-attach"
  policy_arn = aws_iam_policy.mlflow_s3_policy.arn
  roles      = [aws_iam_role.mlflow_task_execution_role.name]
}


# Task Definition

resource "aws_ecs_task_definition" "mlflow" {
  family                   = "mlflow-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.mlflow_task_execution_role.arn

  container_definitions = jsonencode([
    {
      name      = "mlflow"
      image     = "ghcr.io/mlflow/mlflow"
      portMappings = [
        {
          containerPort = 5050
          protocol      = "tcp"
        }
      ]
      command = [
        "mlflow",
        "server",
        "--host", "0.0.0.0",
        "--port", "5050",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "s3://${aws_s3_bucket.mlflow_artifacts.bucket}"
      ]
    }
  ])
}


# ECS Service
resource "aws_ecs_service" "mlflow" {
  name            = "mlflow-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.mlflow.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = [aws_subnet.main.id]
    security_groups  = [aws_security_group.ecs_sg.id]
    assign_public_ip = true
  }
}
