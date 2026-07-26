# infra/ — on-demand GPU node for CUDA work

Terraform for a single GPU EC2 node in the personal AWS account, used when the
5090 isn't reachable (travel) or when a task wants a throwaway CUDA box: the
CUDA-graphs capture (`viz/capture_cudagraphs.py`) was its first job.

Defaults: **g6.xlarge** (1x L4 24GB, bf16-capable, ~$0.80/hr on-demand) in
**us-west-2**, running the AWS Deep Learning Base GPU AMI (Ubuntu 24.04,
NVIDIA driver preinstalled; `nvidia-smi` works at first boot). The repo's own
uv environment provides CUDA torch; the AMI brings only the driver.

## The loop

```
./up.sh      # price check, then apply; SSH is locked to your current public IP (/32)
./sync.sh    # rsync the working tree to ubuntu@node:cantoLLM/ (gitignore-aware)
./ssh.sh     # ssh in (args pass through: ./ssh.sh nvidia-smi)
             # on the node: cd cantoLLM && uv sync && ...   (use tmux for long runs)
             # copy results home with scp, or run sync in reverse by hand
./down.sh    # destroy the instance only; key pair + security group persist
```

`up.sh` is safe to re-run: if the node is already up and your IP changed
(new cafe), it just updates the SSH rule in place.

## Notes

- **State is local** (`terraform.tfstate`, gitignored), so up/down works only
  from the machine that applied. Same convention as the personal infra repo.
- `terraform.tfvars` (gitignored) holds `ssh_public_key`; see
  `terraform.tfvars.example`.
- **Cost control is `./down.sh`**: the instance is the only metered resource.
  There is no auto-shutdown timer; do not leave the node up overnight.
- If apply fails with `InsufficientInstanceCapacity`, retry, or set
  `INSTANCE_TYPE=g5.xlarge ./up.sh` (A10G) as the fallback type.
- The instance's public IP changes on every up; the scripts always read it
  from `terraform output`.
