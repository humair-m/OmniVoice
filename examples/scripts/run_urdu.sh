#!/bin/bash
# Phase 2: Urdu pretraining — streams tokenised audio from HF Hub.
set -euo pipefail

# ===== Configure =====
GPU_IDS="0"
NUM_GPUS=1
TRAIN_CONFIG="examples/config/train_config_urdu.json"
DATA_CONFIG_TEMPLATE="examples/config/data_config_urdu.json"
RESOLVED_DATA_CONFIG="examples/config/data_config_urdu_resolved.json"
OUTPUT_DIR="exp/omnivoice_urdu"
HF_PROCESSED_REPO="Humair332/urdu-omnivoice-tokens"   # output of Phase 1
HF_MODEL_REPO="Humair332/omnivoice-urdu"              # checkpoints go here
# =====================

# Ensure we're in the repo root
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"

echo "[Phase2] Discovering shards from ${HF_PROCESSED_REPO}..."

# Auto-discover shards from HF and write resolved config
python3 - <<EOF
import json
import os
import sys
from omnivoice.utils.hf_shards import discover_shards

repo = "${HF_PROCESSED_REPO}"
token = os.environ.get("HF_TOKEN")
shards = discover_shards(repo, token)

if not shards:
    print(f"Error: No shards found in {repo}")
    sys.exit(1)

print(f"Found {len(shards)} shards.")

with open("${DATA_CONFIG_TEMPLATE}", "r") as f:
    config = json.load(f)

# Update the first training dataset with discovered shards
config["train"][0]["audio_shards"] = shards

with open("${RESOLVED_DATA_CONFIG}", "w") as f:
    json.dump(config, f, indent=2)

print(f"Resolved config written to ${RESOLVED_DATA_CONFIG}")
EOF

echo "[Phase2] Launching training with ${NUM_GPUS} GPUs..."

accelerate launch \
    --gpu_ids "${GPU_IDS}" \
    --num_processes ${NUM_GPUS} \
    -m omnivoice.cli.train \
    --train_config  "${TRAIN_CONFIG}" \
    --data_config   "${RESOLVED_DATA_CONFIG}" \
    --output_dir    "${OUTPUT_DIR}"
