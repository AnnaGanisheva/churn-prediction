resource "aws_security_group" "prefect_sg" {
  name        = "prefect-sg"
  description = "Allow port 4200"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 4200
    to_port     = 4200
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_task_definition" "prefect" {
  family                   = "prefect-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn

  container_definitions = jsonencode([
    {
      name      = "prefect"
      image     = "prefecthq/prefect:2.10.13-python3.10"
      command   = ["prefect", "server", "start", "--host", "0.0.0.0"]
      portMappings = [{
        containerPort = 4200
        protocol      = "tcp"
      }]
    }
  ])
}

resource "aws_ecs_service" "prefect_service" {
  name            = "prefect-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.prefect.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = [aws_subnet.main.id]
    assign_public_ip = true
    security_groups  = [aws_security_group.prefect_sg.id]
  }
}
