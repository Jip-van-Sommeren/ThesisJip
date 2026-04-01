terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# --- AMI lookup (Ubuntu 22.04) ---

data "aws_ami" "ubuntu" {
  count       = var.ami_id == "" ? 1 : 0
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  owners = ["099720109477"] # Canonical
}

locals {
  ami_id = var.ami_id != "" ? var.ami_id : data.aws_ami.ubuntu[0].id
}

# --- Default VPC ---

data "aws_vpc" "default" {
  default = true
}

# --- Security Group ---

resource "aws_security_group" "benchmark" {
  name        = "${var.project_name}-sg"
  description = "Security group for distributed benchmarks"
  vpc_id      = data.aws_vpc.default.id

  # SSH from allowed CIDR
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_allowed_cidr]
    description = "SSH access"
  }

  # REST (Flask agent servers)
  ingress {
    from_port   = 5000
    to_port     = 5100
    protocol    = "tcp"
    self        = true
    description = "REST communication"
  }

  # gRPC
  ingress {
    from_port   = 50051
    to_port     = 50151
    protocol    = "tcp"
    self        = true
    description = "gRPC communication"
  }

  # MQTT
  ingress {
    from_port   = 1883
    to_port     = 1883
    protocol    = "tcp"
    self        = true
    description = "MQTT broker"
  }

  # Kafka
  ingress {
    from_port   = 9092
    to_port     = 9092
    protocol    = "tcp"
    self        = true
    description = "Kafka broker"
  }

  # Zookeeper / Kafka controller
  ingress {
    from_port   = 9093
    to_port     = 9093
    protocol    = "tcp"
    self        = true
    description = "Kafka controller"
  }

  # Agent worker control port
  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = [var.ssh_allowed_cidr]
    description = "Agent worker control"
  }

  # Allow all within security group
  ingress {
    from_port = 8080
    to_port   = 8080
    protocol  = "tcp"
    self      = true
    description = "Agent worker internal"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound"
  }

  tags = {
    Name    = "${var.project_name}-sg"
    Project = var.project_name
  }
}

# --- Agent instances ---

resource "aws_instance" "agent" {
  count = var.use_spot ? 0 : var.agent_host_count

  ami                    = local.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.benchmark.id]
  user_data              = file("${path.module}/user_data.sh")

  tags = {
    Name    = "${var.project_name}-agent-${count.index}"
    Project = var.project_name
    Role    = "agent"
  }
}

resource "aws_spot_instance_request" "agent_spot" {
  count = var.use_spot ? var.agent_host_count : 0

  ami                    = local.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.benchmark.id]
  user_data              = file("${path.module}/user_data.sh")

  spot_price                  = var.spot_max_price != "" ? var.spot_max_price : null
  wait_for_fulfillment        = true
  instance_interruption_behavior = "terminate"

  tags = {
    Name    = "${var.project_name}-agent-spot-${count.index}"
    Project = var.project_name
    Role    = "agent"
  }
}

# --- Broker instance ---

resource "aws_instance" "broker" {
  count = var.create_broker && !var.use_spot ? 1 : 0

  ami                    = local.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.benchmark.id]
  user_data              = file("${path.module}/user_data.sh")

  tags = {
    Name    = "${var.project_name}-broker"
    Project = var.project_name
    Role    = "broker"
  }
}

resource "aws_spot_instance_request" "broker_spot" {
  count = var.create_broker && var.use_spot ? 1 : 0

  ami                    = local.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.benchmark.id]
  user_data              = file("${path.module}/user_data.sh")

  spot_price                  = var.spot_max_price != "" ? var.spot_max_price : null
  wait_for_fulfillment        = true
  instance_interruption_behavior = "terminate"

  tags = {
    Name    = "${var.project_name}-broker-spot"
    Project = var.project_name
    Role    = "broker"
  }
}

# --- S3 bucket for results ---

resource "aws_s3_bucket" "results" {
  bucket_prefix = "${var.project_name}-results-"
  force_destroy = true

  tags = {
    Name    = "${var.project_name}-results"
    Project = var.project_name
  }
}
