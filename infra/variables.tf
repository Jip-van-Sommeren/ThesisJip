variable "region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-1"
}

variable "instance_type" {
  description = "EC2 instance type for benchmark hosts"
  type        = string
  default     = "t3.medium"
}

variable "agent_host_count" {
  description = "Number of agent EC2 instances"
  type        = number
  default     = 2
}

variable "create_broker" {
  description = "Whether to create a dedicated broker instance"
  type        = bool
  default     = true
}

variable "key_name" {
  description = "Name of the AWS key pair for SSH access"
  type        = string
}

variable "ssh_allowed_cidr" {
  description = "CIDR block allowed to SSH (your IP, e.g. 1.2.3.4/32)"
  type        = string
}

variable "ami_id" {
  description = "AMI ID (Ubuntu 22.04). Leave empty to auto-detect."
  type        = string
  default     = ""
}

variable "use_spot" {
  description = "Use spot instances for cost savings"
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Maximum price for spot instances (empty = on-demand price)"
  type        = string
  default     = ""
}

variable "project_name" {
  description = "Project name for resource tagging"
  type        = string
  default     = "mas-benchmark"
}
