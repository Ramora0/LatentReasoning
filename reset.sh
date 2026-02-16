#!/bin/bash
# Reset TPU state after a crashed/hung training run

# Kill any lingering Python processes (training workers)
pkill -9 -f "scripts/train.py" 2>/dev/null
pkill -9 -f "torch_xla" 2>/dev/null

# Kill any orphan processes holding TPU locks
sudo lsof /dev/accel* 2>/dev/null | awk 'NR>1 {print $2}' | sort -u | xargs -r sudo kill -9 2>/dev/null

# Reset TPU chips
sudo ls /dev/accel* >/dev/null 2>&1 && echo "TPU devices present" || echo "No TPU devices found"

# Clear any leftover shared memory
rm -rf /tmp/torch_xla_* 2>/dev/null
rm -rf /tmp/xla_* 2>/dev/null
rm -rf /dev/shm/torch_xla_* 2>/dev/null

# Clear Python cache that might hold stale compiled graphs
find . -path '*torch_xla*__pycache__*' -delete 2>/dev/null

# Brief pause to let resources release
sleep 2

echo "Reset complete. You can re-run training now."
