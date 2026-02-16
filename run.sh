python scripts/train.py --config configs/base.yaml \
    distributed.backend=xla \
    paths.scratch_dir=~/data

gcloud compute tpus tpu-vm ssh latent-reasoning --zone=us-central2-b