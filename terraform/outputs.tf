output "service_name" {
  value = aws_ecs_service.streamlit_service.name
}
output "mlflow_ui_url" {
  value = "http://${aws_lb.mlflow_lb.dns_name}:5050"
}

output "streamlit_url" {
  value = "http://${aws_lb.streamlit_lb.dns_name}:8501"
}
