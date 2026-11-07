# Verify one ALFWorld update

Follow the [ALFWorld setup](../README.md) through the game download. In the same shell, run one update on the eight GPUs used by the ALFWorld training configuration.

```bash
mkdir -p "$PWD/runs"
export RUN_DIR="$(mktemp -d "$PWD/runs/alfworld-step1-XXXXXX")"
set -euo pipefail
bash scripts/train_alfworld.sh \
  trainer.total_training_steps=1 trainer.save_freq=1 trainer.test_freq=-1 \
  2>&1 | tee "$RUN_DIR/train.log"
python scripts/audit_step.py "$RUN_DIR"
torchrun --standalone --nproc-per-node=8 scripts/verify_checkpoint.py \
  "$RUN_DIR/global_step_1/actor" "$MODEL_PATH"
```

The rollout audit checks all 64 student trajectories, which groups received teacher attempts, and which verified recoveries entered the update. The checkpoint check reloads the actor, compares its first-layer normalization weights with the input SFT checkpoint, and verifies that AdamW advanced. The number of successful recoveries can vary between runs.
