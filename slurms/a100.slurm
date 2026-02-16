#!/bin/bash
#SBATCH --job-name=latent-reasoning
#SBATCH --partition=nextgen
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --account=PAS2836
#SBATCH --output=./slurms/slurm-%j.out
#SBATCH --error=./slurms/slurm-%j.err

set -euo pipefail

# CLI arguments
# Usage: sbatch slurms/a100.slurm [branch] [run_name] [config overrides...]
# Example: sbatch slurms/a100.slurm main my-experiment training.learning_rate=1e-4 latent.K=12
BRANCH="${1:-main}"
shift || true
RUN_NAME="${1:-}"
shift || true
EXTRA_ARGS="$@"

# Each job gets unique directory (SLURM_JOB_ID ensures no collisions)
CODE_LOCAL="$TMPDIR/LatentReasoning-$SLURM_JOB_ID"

echo "================================================"
echo "Cloning branch '$BRANCH' to local storage..."
echo "Job ID: $SLURM_JOB_ID"
echo "================================================"

# Get remote URL from local repo, then clone from remote
REMOTE_URL=$(git -C "$HOME/personal/LatentReasoning" remote get-url origin)

# Clone specific branch directly into isolated temp directory
git clone --branch "$BRANCH" --single-branch --depth 1 \
    "$REMOTE_URL" "$CODE_LOCAL"

echo "Clone complete. Branch: $(cd "$CODE_LOCAL" && git branch --show-current)"
echo "Commit: $(cd "$CODE_LOCAL" && git rev-parse --short HEAD)"

# Activate python env
source "$HOME/personal/LatentReasoning/.a100/bin/activate"

# Go to local code copy
cd "$CODE_LOCAL"

# Build config override args
OVERRIDE_ARGS=""
if [ -n "$RUN_NAME" ]; then
    OVERRIDE_ARGS="logging.wandb_run_name=$RUN_NAME"
fi
if [ -n "$EXTRA_ARGS" ]; then
    OVERRIDE_ARGS="$OVERRIDE_ARGS $EXTRA_ARGS"
fi

# Run training
python scripts/train.py --config configs/base.yaml $OVERRIDE_ARGS

echo "================================================"
echo "Training complete. Cleaning up..."
echo "================================================"
rm -rf "$CODE_LOCAL"
