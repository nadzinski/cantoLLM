output "instance_id" {
  value = aws_instance.cantollm.id
}

output "public_ip" {
  value = aws_instance.cantollm.public_ip
}

output "ami_id" {
  value = data.aws_ami.dlami.id
}

output "ami_name" {
  value = data.aws_ami.dlami.name
}
