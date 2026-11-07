#!/usr/bin/env bash
# Full-parameter SFT for Qwen3-1.7B on the released SkillRL ALFWorld data.
set -euo pipefail

PY_BIN="$(dirname "$(command -v python)")"
export PATH="${PY_BIN}:${PATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT"
SPLIT_DIR="${SPLIT_DIR:-$REPO_ROOT/runs/sft-input}"
mkdir -p "$SPLIT_DIR"
if [[ -z "${RAW_TRAIN:-}" ]]; then
  RAW_TRAIN="$(hf download Jianwen/SkillRL-SFT-Data --repo-type dataset --revision bd3996c4e863ac59e6e4ab35549cf3741faf5e4f alfworld/train-00000-of-00001.parquet)"
fi
if [[ -z "${BASE_MODEL:-}" ]]; then
  BASE_MODEL="$(hf download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e)"
fi
TRAIN_FILE="$SPLIT_DIR/train.parquet"
VAL_FILE="$SPLIT_DIR/val.parquet"
NUM_GPUS="${NUM_GPUS:-8}"
EPOCHS="${EPOCHS:-3}"
TRAIN_BSZ="${TRAIN_BSZ:-64}"
MICRO_BSZ="${MICRO_BSZ:-4}"
MAX_LEN="${MAX_LEN:-1536}"
LR="${LR:-1e-5}"
VAL_FRAC="${VAL_FRAC:-0.02}"

EXP_NAME="${EXP_NAME:-alfworld-qwen3-sft}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/runs/sft}"

for executable in python torchrun; do
  if [[ ! -x "${PY_BIN}/${executable}" ]]; then
    echo "Missing executable: ${PY_BIN}/${executable}" >&2
    exit 2
  fi
done
for required in "${RAW_TRAIN}" "${BASE_MODEL}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${SAVE_DIR}" "${SPLIT_DIR}"

# The released data is train-only. Create the same deterministic 2% validation
# split while keeping each instruction and response together.
if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
  "${PY_BIN}/python" - "${RAW_TRAIN}" "${TRAIN_FILE}" "${VAL_FILE}" "${VAL_FRAC}" <<'PY'
import os
import sys

import numpy as np
import pandas as pd

raw_path, train_path, val_path, val_frac = sys.argv[1:]
df = pd.read_parquet(raw_path)
rng = np.random.RandomState(0)
indices = rng.permutation(len(df))
n_val = max(1, int(len(df) * float(val_frac)))
val = df.iloc[indices[:n_val]].reset_index(drop=True)
train = df.iloc[indices[n_val:]].reset_index(drop=True)
os.makedirs(os.path.dirname(train_path), exist_ok=True)
train.to_parquet(train_path, index=False)
val.to_parquet(val_path, index=False)
print(f"created exact-content split: train={len(train)} val={len(val)}")
PY
fi

# Refuse to train if the split is not an exact partition of the released data
# or if any instruction lost its skill block.
"${PY_BIN}/python" - "${RAW_TRAIN}" "${TRAIN_FILE}" "${VAL_FILE}" <<'PY'
from collections import Counter
import hashlib
import sys

import pandas as pd

raw_path, train_path, val_path = sys.argv[1:]
raw = pd.read_parquet(raw_path)
train = pd.read_parquet(train_path)
val = pd.read_parquet(val_path)
required = {"instruction", "output"}
for name, frame in (("raw", raw), ("train", train), ("val", val)):
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"{name} data is missing columns: {sorted(missing)}")
    if frame["instruction"].isna().any() or frame["output"].isna().any():
        raise SystemExit(f"{name} data contains null instruction/output values")
    skill_mask = frame["instruction"].astype(str).str.contains(
        "## Retrieved Relevant Experience", regex=False
    )
    if not bool(skill_mask.all()):
        raise SystemExit(
            f"{name} data lost skill blocks in {(~skill_mask).sum()} rows"
        )

def rows(frame):
    return Counter(zip(frame["instruction"].astype(str), frame["output"].astype(str)))

combined = pd.concat([train, val], ignore_index=True)
if rows(raw) != rows(combined):
    raise SystemExit("train/val is not an exact partition of released SkillRL data")

with open(raw_path, "rb") as handle:
    digest = hashlib.sha256(handle.read()).hexdigest()
print(
    f"SkillRL preflight passed: raw={len(raw)} train={len(train)} val={len(val)} "
    f"all_skill_blocks=true sha256={digest}"
)
PY

cd "${REPO_ROOT}"
export WANDB_NAME="${EXP_NAME}"
"${PY_BIN}/torchrun" --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.prompt_key=instruction \
  data.response_key=output \
  'data.prompt_dict_keys=[]' \
  'data.response_dict_keys=[]' \
  data.train_batch_size="${TRAIN_BSZ}" \
  data.micro_batch_size_per_gpu="${MICRO_BSZ}" \
  data.max_length="${MAX_LEN}" \
  data.truncation=error \
  data.multiturn.enable=False \
  model.partial_pretrain="${BASE_MODEL}" \
  model.lora_rank=0 \
  model.strategy=fsdp2 \
  optim.lr="${LR}" \
  optim.weight_decay=0.01 \
  optim.warmup_steps_ratio=0.05 \
  optim.lr_scheduler=cosine \
  use_remove_padding=True \
  ulysses_sequence_parallel_size=1 \
  trainer.default_local_dir="${SAVE_DIR}" \
  trainer.project_name=OVCSD-SFT \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.total_epochs="${EPOCHS}" \
  trainer.total_training_steps=228 \
  trainer.seed=0 \
  trainer.logger="[console]" \
  trainer.default_hdfs_dir=null \
  "$@"

echo "SFT complete: ${SAVE_DIR}"
