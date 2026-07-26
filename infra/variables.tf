variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "instance_type" {
  description = "GPU instance type (g6.xlarge = 1x L4 24GB, bf16-capable, ~$0.80/hr on-demand)"
  type        = string
  default     = "g6.xlarge"
}

variable "ssh_public_key" {
  description = "Public key for SSH access (set in terraform.tfvars)"
  type        = string
}

variable "ssh_cidr" {
  description = "CIDR allowed to SSH in. up.sh overrides this with your current public IP as a /32; the default is unroutable on purpose so nothing is ever open by accident."
  type        = string
  default     = "127.0.0.1/32"
}
