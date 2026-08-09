terraform {
  required_version = ">= 1.5"

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

# ---------- AMI lookup (latest AWS Deep Learning Base GPU AMI, Ubuntu 24.04) ----------
# NVIDIA driver + CUDA runtime preinstalled; nvidia-smi works at first boot.
# The repo's own uv environment provides torch, so no framework AMI is needed.

data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# ---------- SSH key pair ----------

resource "aws_key_pair" "cantollm" {
  key_name   = "cantollm-key"
  public_key = var.ssh_public_key
}

# ---------- Security group ----------

resource "aws_security_group" "cantollm" {
  name        = "cantollm-sg"
  description = "Allow SSH inbound, all outbound"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "cantollm-sg"
  }
}

# ---------- GPU instance ----------

resource "aws_instance" "cantollm" {
  ami                    = data.aws_ami.dlami.id
  instance_type          = var.instance_type
  availability_zone      = var.availability_zone
  key_name               = aws_key_pair.cantollm.key_name
  vpc_security_group_ids = [aws_security_group.cantollm.id]

  root_block_device {
    volume_size = var.root_volume_gb # the DLAMI itself is large; leave headroom for the repo + wheels
    volume_type = "gp3"
  }

  user_data                   = file("${path.module}/cloud-init.yaml")
  user_data_replace_on_change = true

  tags = {
    Name = "cantollm"
  }
}
