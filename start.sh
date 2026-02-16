git clone https://github.com/Ramora0/LatentReasoning.git
sudo apt-get update
sudo apt-get install -y python3.10-venv
cd LatentReasoning/
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[tpu]"
pip uninstall -y torch torch_xla
pip install torch==2.9.0 torch_xla==2.9.0 -f https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-2.9.0-cp310-cp310-linux_x86_64.whl
python scripts/download.py --scratch-dir ./data
wandb login
PJRT_DEVICE=TPU python scripts/train.py --config configs/tpu.yaml \
    distributed.backend=xla \
    paths.scratch_dir=./data \