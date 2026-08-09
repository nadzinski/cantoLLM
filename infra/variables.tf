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

variable "root_volume_gb" {
  description = "Root EBS volume size in GB. The DLAMI eats most of the default 100; big-model sessions (weights land in the HF cache) should raise it, e.g. ROOT_VOLUME_GB=150 for the H100 profile."
  type        = number
  default     = 100
}

variable "availability_zone" {
  description = "Pin the instance to one AZ. Null lets AWS pick. Scarce types (p5.4xlarge) can hit InsufficientInstanceCapacity in a given AZ; pinning turns a capacity failure into 'try the next AZ'."
  type        = string
  default     = null
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
