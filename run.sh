python scripts/train.py --config configs/base.yaml \
    distributed.backend=xla \
    paths.scratch_dir=~/data

gcloud compute tpus tpu-vm ssh latent-reasoning --zone=us-central2-b

PJRT_DEVICE=TPU python scripts/evaluate.py --config configs/base.yaml --checkpoint checkpoints/step_4096/checkpoint.pt distributed.backend=xla