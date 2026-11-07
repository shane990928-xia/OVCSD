# OVCSD

Training code for *From Scoring to Acting: Outcome-Verified Comparative Self-Distillation for LLM Agents*.

OVCSD trains a skill-free policy with GRPO in ALFWorld. When all eight rollouts for a task fail, the same policy uses retrieved skills to retry from a verified prefix. Successful recoveries provide supervision at the first aligned action difference and on the steps that follow. Training uses the [bundled verl implementation](verl/).

## Installation

Use Linux, Python 3.12 and the CUDA 12.8 toolkit.

```bash
git clone https://github.com/shane990928-xia/OVCSD.git
cd OVCSD
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel ninja packaging psutil
python -m pip install -r requirements-verl.txt
MAX_JOBS=4 python -m pip install --no-build-isolation flash-attn==2.8.3
```

## Train on ALFWorld

Initialize Qwen3-1.7B on the ALFWorld subset of the official [SkillRL SFT dataset](https://huggingface.co/datasets/Jianwen/SkillRL-SFT-Data). The script downloads the base model and training data.

```bash
bash scripts/rebuild_sft.sh
export MODEL_PATH="$PWD/runs/sft/global_step_228"
```

Download the games and start training with the ALFWorld configuration from the paper.

```bash
export ALFWORLD_DATA="$PWD/assets/alfworld"
mkdir -p "$ALFWORLD_DATA"
alfworld-download --data-dir "$ALFWORLD_DATA"
bash scripts/train_alfworld.sh
```

[Training configuration](recipe/skill_sd/config/skill_sd_trainer.yaml) · [Single-update verification](docs/reproduction.md) · [Third-party notices](Notice.txt)
