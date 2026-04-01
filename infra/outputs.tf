output "agent_ips" {
  description = "Private IPs of agent instances"
  value = var.use_spot ? [
    for req in aws_spot_instance_request.agent_spot : req.private_ip
  ] : [
    for inst in aws_instance.agent : inst.private_ip
  ]
}

output "agent_public_ips" {
  description = "Public IPs of agent instances (for SSH)"
  value = var.use_spot ? [
    for req in aws_spot_instance_request.agent_spot : req.public_ip
  ] : [
    for inst in aws_instance.agent : inst.public_ip
  ]
}

output "broker_ip" {
  description = "Private IP of broker instance"
  value = var.create_broker ? (
    var.use_spot ? (
      length(aws_spot_instance_request.broker_spot) > 0
      ? aws_spot_instance_request.broker_spot[0].private_ip
      : null
    ) : (
      length(aws_instance.broker) > 0
      ? aws_instance.broker[0].private_ip
      : null
    )
  ) : null
}

output "broker_public_ip" {
  description = "Public IP of broker instance (for SSH)"
  value = var.create_broker ? (
    var.use_spot ? (
      length(aws_spot_instance_request.broker_spot) > 0
      ? aws_spot_instance_request.broker_spot[0].public_ip
      : null
    ) : (
      length(aws_instance.broker) > 0
      ? aws_instance.broker[0].public_ip
      : null
    )
  ) : null
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.benchmark.id
}

output "s3_bucket" {
  description = "S3 bucket name for results"
  value       = aws_s3_bucket.results.id
}
