#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the ALFWorld 1.7B SFT step-228 checkpoint}"
: "${ALFWORLD_DATA:?Set ALFWORLD_DATA to the ALFWorld asset directory}"
export OVCSD_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$OVCSD_ROOT"
export MODEL_PATH="$(realpath "$MODEL_PATH")" ALFWORLD_DATA="$(realpath "$ALFWORLD_DATA")"
export RUN_DIR="$(realpath -m "${RUN_DIR:-$OVCSD_ROOT/runs/alfworld-$(date +%Y%m%d-%H%M%S)}")"
export OVCSD_RAY_DIR="$(mktemp -d /tmp/ovcsd-ray-XXXXXX)"
mkdir -p "$RUN_DIR/tmp" "$RUN_DIR/fast-downward"
ln -s "$RUN_DIR/tmp" "$OVCSD_RAY_DIR/scratch"
# multiprocessing uses AF_UNIX sockets under TMPDIR with a small path limit.
export TMPDIR="$OVCSD_RAY_DIR/scratch" CPSD_FAST_DOWNWARD_LIB_DIR="$RUN_DIR/fast-downward"
export CPSD_FAST_DOWNWARD_LOAD_LOCK="$RUN_DIR/fast-downward/load.lock"
export PYTHONPATH="$OVCSD_ROOT" OVCSD_ENV_PROCESS_BACKEND=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 TOKENIZERS_PARALLELISM=false
export RAY_num_server_call_thread=1 RAY_worker_num_grpc_internal_threads=1 RAY_num_grpc_internal_threads=1
export RAY_gcs_server_rpc_client_thread_num=1 RAY_gcs_server_rpc_server_thread_num=2 RAY_object_manager_rpc_threads_num=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
python scripts/prepare_alfworld.py "$ALFWORLD_DATA" "$RUN_DIR/data"
python -m recipe.skill_sd.main_skill_sd "$@"
