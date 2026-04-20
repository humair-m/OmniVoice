# OmniVoice Urdu Pipeline — Claude Instruction File

> **Repo:** `k2-fsa/OmniVoice` (clone into your AntiGravity workspace)
> **Goal:** Replace the monolithic `run_emilia.sh` with a clean two-phase,
> resumable, HuggingFace-native pipeline for the Urdu dataset.

---

## 0. Context You Must Understand First

Read these files from the cloned repo before writing a single line:

| File | Why |
|---|---|
| `omnivoice/scripts/extract_audio_tokens.py` | Core tokenisation logic you will reuse |
| `omnivoice/cli/train.py` | Entry-point for training; resume logic lives here |
| `omnivoice/data/` | WebDataset / shard loading code |
| `examples/config/train_config_emilia.json` | Reference config (provided above) |
| `examples/config/data_config_emilia.json` | Data path schema expected by trainer |

The existing pipeline expects **WebDataset `.tar` shards** for audio tokens and
parallel **`.jsonl`** files for text. Your new pipeline must produce the same
format and upload them as a HuggingFace dataset so Phase 2 can stream directly
from HF without local copies.

---

## 1. What To Build — High-Level

```
Phase 1 — Data Ingestion & Tokenisation (run once, fully resumable)
  └─ for each chunk of N parquet files from Humair332/Urdu-munch-1:
        1. Download chunk
        2. Tokenise audio  →  .tar shards + .jsonl files
        3. Upload shards to HF dataset repo
        4. Delete local copies
        5. Write a progress ledger so the loop can resume after crash

Phase 2 — Pretraining (stream from HF, push checkpoints to HF Hub)
  └─ accelerate launch omnivoice/cli/train.py
        • Download tokenised shards from HF dataset (no local copy needed)
        • saves checkpoint every save_steps  →  push to HF Hub
        • resumes from last checkpoint with full optimizer + scheduler state
        • never finetunes: always init from raw LLM weights or a pretrain ckpt
```

---

## 2. Phase 1 — Detailed Specification

### 2.1 New Script: `examples/scripts/phase1_tokenise_urdu.py`

Create this file. It must do everything below.

#### CLI Arguments

```
--hf_source_repo      str   default="Humair332/Urdu-munch-1"
--hf_token            str   required  (or read from HF_TOKEN env var)
--hf_processed_repo   str   required  e.g. "YourOrg/urdu-omnivoice-tokens"
--tokenizer_path      str   default="eustlb/higgs-audio-v2-tokenizer"
--chunk_size          int   default=5   # parquet files per iteration
--shard_max_tokens    int   default=50000  # tokens per .tar shard
--output_dir          str   default="tmp_phase1"  # scratch space, wiped each chunk
--ledger_path         str   default="phase1_ledger.json"
--gpu_ids             str   default="0"
--nj_per_gpu          int   default=2
--audio_col           str   default="audio"
--text_col            str   default="text"
--split               str   default="train"  # HF dataset split to process
```

#### Ledger Schema (`phase1_ledger.json`)

```json
{
  "total_parquet_files": 42,
  "processed_parquet_files": [],
  "uploaded_shards": [],
  "last_updated": "2025-01-01T00:00:00"
}
```

- On **startup**, load ledger if it exists; skip already-processed files.
- On **each chunk completion**, atomically write updated ledger before deleting
  local files.  Use `json.dump` + `os.replace` (write-then-rename) so a crash
  cannot corrupt it.

#### Main Loop (pseudocode — implement exactly this logic)

```python
source_files = list_parquet_files(hf_source_repo, split)        # use datasets lib
remaining    = [f for f in source_files if f not in ledger["processed"]]

for chunk in batch(remaining, chunk_size):
    # 1. Download
    rows = load_dataset_chunk(hf_source_repo, chunk,
                              audio_col=audio_col, text_col=text_col)

    # 2. Write temp JSONL manifest  (path, duration, text)
    manifest_path = write_temp_manifest(rows, output_dir)

    # 3. Tokenise  (reuse existing extract_audio_tokens logic)
    tar_pattern  = f"{output_dir}/shards/shard-%06d.tar"
    jsonl_pattern= f"{output_dir}/shards/shard-%06d.jsonl"
    run_tokenisation(manifest_path, tar_pattern, jsonl_pattern,
                     tokenizer_path, gpu_ids, nj_per_gpu)

    # 4. Upload shards to HF dataset repo
    upload_shards_to_hf(output_dir, hf_processed_repo, hf_token)

    # 5. Update ledger  (atomic write)
    ledger["processed"].extend(chunk)
    write_ledger(ledger_path, ledger)

    # 6. Delete local scratch
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Phase1] Chunk done. {len(ledger['processed'])}/{len(source_files)} files processed.")

print("[Phase1] Complete.")
```

#### Audio Decoding

The `audio` column in `Humair332/Urdu-munch-1` contains HuggingFace Audio
feature dicts with keys `{"bytes": ..., "path": ...}` (or a decoded dict with
`array` + `sampling_rate`).  Handle **both** formats:

```python
def decode_audio(row, col):
    val = row[col]
    if isinstance(val, dict) and "array" in val:
        # already decoded
        return val["array"], val["sampling_rate"]
    elif isinstance(val, dict) and "bytes" in val:
        import soundfile as sf, io
        wav, sr = sf.read(io.BytesIO(val["bytes"]))
        return wav, sr
    else:
        raise ValueError(f"Unknown audio format: {type(val)}")
```

Resample to **24 kHz** (required by higgs-audio tokenizer) using `torchaudio`.

#### Manifest JSONL Format

Match what `extract_audio_tokens.py` already expects:

```jsonl
{"audio_path": "/abs/path/to/chunk_0001.wav", "text": "...", "duration": 3.14}
```

Write decoded WAVs to `{output_dir}/wavs/` and reference them absolutely.
Clean up wavs after tokenisation, before upload.

#### HF Upload

Use `huggingface_hub.HfApi`:

```python
api = HfApi(token=hf_token)
api.create_repo(hf_processed_repo, repo_type="dataset", exist_ok=True)

for shard_file in glob(f"{output_dir}/shards/*"):
    api.upload_file(
        path_or_fileobj=shard_file,
        path_in_repo=f"data/{os.path.basename(shard_file)}",
        repo_id=hf_processed_repo,
        repo_type="dataset",
    )
```

Upload `.tar` and `.jsonl` shards together.  Use `upload_folder` if the number
of files per chunk is large (> 20) — it batches internally.

---

## 3. Phase 2 — Detailed Specification

### 3.1 Config Changes

#### New file: `examples/config/train_config_urdu.json`

Copy `train_config_emilia.json` and change only these keys:

```json
{
  "llm_name_or_path": "Qwen/Qwen3-0.6B",
  "language_ratio": 0.0,
  "use_pinyin_ratio": 0.0,
  "instruct_ratio": 0.0,
  "only_instruct_ratio": 0.0,
  "resume_from_checkpoint": null,
  "init_from_checkpoint": null,
  "save_steps": 1000,
  "eval_steps": 500,
  "logging_steps": 50,
  "steps": 300000,
  "push_to_hub": true,
  "hub_model_id": "YourOrg/omnivoice-urdu",
  "hub_token": null
}
```

> **`push_to_hub`** and **`hub_model_id`** are new keys you will add to the
> trainer.  `hub_token` can be `null` to fall back to `HF_TOKEN` env var.

#### New file: `examples/config/data_config_urdu.json`

This must point to your HF-hosted processed repo using the WebDataset HTTP
streaming URL pattern:

```json
{
  "train": [
    {
      "name": "urdu_train",
      "audio_pattern": "pipe:curl -s -L https://huggingface.co/datasets/YourOrg/urdu-omnivoice-tokens/resolve/main/data/shard-{000000..XXXXXX}.tar",
      "text_pattern":  "pipe:curl -s -L https://huggingface.co/datasets/YourOrg/urdu-omnivoice-tokens/resolve/main/data/shard-{000000..XXXXXX}.jsonl",
      "weight": 1.0
    }
  ],
  "dev": []
}
```

> You will need to update the shard range `XXXXXX` after Phase 1 completes, or
> make the trainer discover shards dynamically (preferred — see §3.3).

### 3.2 Trainer Changes: `omnivoice/cli/train.py`

Make **minimal, surgical edits**.  Add the following behaviour:

#### A. Push checkpoint to HF Hub after every `save_steps`

After the existing `accelerator.save_state(checkpoint_dir)` call, add:

```python
if cfg.get("push_to_hub") and accelerator.is_main_process:
    from huggingface_hub import HfApi
    token = cfg.get("hub_token") or os.environ.get("HF_TOKEN")
    api   = HfApi(token=token)
    repo  = cfg["hub_model_id"]
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo,
        repo_type="model",
        path_in_repo=f"checkpoint-{global_step}",
        commit_message=f"checkpoint step {global_step}",
    )
    logger.info(f"Pushed checkpoint-{global_step} to {repo}")
```

#### B. Resume with full optimizer + scheduler state (pretraining, not finetuning)

The existing code likely uses `accelerator.load_state()`.  Make sure:

1. When `resume_from_checkpoint` is a **local path** — use as-is.
2. When `resume_from_checkpoint` is a **HF Hub repo string**
   (`"YourOrg/omnivoice-urdu:checkpoint-5000"`) — download it first:

```python
def resolve_checkpoint(ckpt_ref: str, local_dir: str) -> str:
    """Returns local path to checkpoint, downloading from HF if needed."""
    if ckpt_ref is None:
        return None
    if os.path.isdir(ckpt_ref):
        return ckpt_ref
    # HF Hub format:  "repo_id:subfolder"
    if ":" in ckpt_ref:
        repo_id, subfolder = ckpt_ref.split(":", 1)
    else:
        repo_id, subfolder = ckpt_ref, None
    from huggingface_hub import snapshot_download
    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{subfolder}/**"] if subfolder else None,
        local_dir=local_dir,
        token=os.environ.get("HF_TOKEN"),
    )
```

3. Do **NOT** reset the optimizer or scheduler when resuming — load them
   directly via `accelerator.load_state(checkpoint_dir)` which already handles
   this correctly.  The key rule: **never call `optimizer.zero_grad()` or
   reinitialise the scheduler after loading state**.

4. Resume the DataLoader from the correct position.  After
   `accelerator.load_state`, set the dataloader skip:

```python
# Inside training loop setup, after loading checkpoint:
if resume_step > 0:
    dataloader = accelerator.skip_first_batches(dataloader, resume_step % len(dataloader))
```

#### C. Dynamic shard discovery from HF dataset

Instead of hardcoding shard ranges in `data_config_urdu.json`, add a helper
that queries the HF API for all `.tar` files in the processed repo:

```python
def discover_shards(hf_repo: str, token: str = None) -> list[str]:
    from huggingface_hub import HfApi
    api   = HfApi(token=token)
    files = api.list_repo_files(hf_repo, repo_type="dataset")
    tars  = sorted([f for f in files if f.endswith(".tar")])
    base  = f"https://huggingface.co/datasets/{hf_repo}/resolve/main"
    return [f"pipe:curl -s -L {base}/{f}" for f in tars]
```

Call this at trainer startup and write the result to a temp
`data_config_resolved.json` that the dataset loader consumes.

### 3.3 New Launch Script: `examples/scripts/run_urdu.sh`

```bash
#!/bin/bash
# Phase 2: Urdu pretraining — streams tokenised audio from HF Hub.
set -euo pipefail

# ===== Configure =====
GPU_IDS="0,1,2,3,4,5,6,7"
NUM_GPUS=8
TRAIN_CONFIG="config/train_config_urdu.json"
DATA_CONFIG="config/data_config_urdu.json"
OUTPUT_DIR="exp/omnivoice_urdu"
HF_PROCESSED_REPO="YourOrg/urdu-omnivoice-tokens"   # output of Phase 1
HF_MODEL_REPO="YourOrg/omnivoice-urdu"              # checkpoints go here
# =====================

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"

# Auto-discover shards from HF (writes tmp data config)
python - <<'EOF'
import json, os, sys
sys.path.insert(0, os.environ["PYTHONPATH"].split(":")[0])
from omnivoice.utils.hf_shards import discover_shards
repo  = os.environ["HF_PROCESSED_REPO"]
token = os.environ["HF_TOKEN"]
shards = discover_shards(repo, token)
cfg = {"train": [{"name": "urdu_train", "audio_shards": shards, "weight": 1.0}], "dev": []}
with open("config/data_config_urdu_resolved.json", "w") as f:
    json.dump(cfg, f, indent=2)
print(f"Discovered {len(shards)} shards from {repo}")
EOF

accelerate launch \
    --gpu_ids "${GPU_IDS}" \
    --num_processes ${NUM_GPUS} \
    -m omnivoice.cli.train \
    --train_config  ${TRAIN_CONFIG} \
    --data_config   config/data_config_urdu_resolved.json \
    --output_dir    ${OUTPUT_DIR} \
    --hub_model_id  ${HF_MODEL_REPO}
```

---

## 4. New Utility Module: `omnivoice/utils/hf_shards.py`

Create this file with the `discover_shards` function from §3.2-C and also:

```python
def upload_folder_chunked(api, folder_path, repo_id, path_in_repo, token,
                           commit_message="upload"):
    """Upload all files in folder_path to HF dataset repo, in one commit."""
    ...

def write_ledger(path: str, data: dict):
    """Atomic JSON write via temp file + rename."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)

def load_ledger(path: str) -> dict:
    if not os.path.exists(path):
        return {"processed_parquet_files": [], "uploaded_shards": []}
    with open(path) as f:
        return json.load(f)
```

---

## 5. Dependencies to Add

Add to `requirements.txt` (or `pyproject.toml`):

```
huggingface_hub>=0.23.0
soundfile>=0.12.1
torchaudio>=2.3.0
datasets>=2.19.0
```

---

## 6. Environment Variables Reference

| Variable | Used by | Notes |
|---|---|---|
| `HF_TOKEN` | Both phases | Must have write access to processed + model repos |
| `PYTHONPATH` | Both phases | Set to repo root |
| `CUDA_VISIBLE_DEVICES` | Phase 1 tokeniser | Comma-separated GPU IDs |

---

## 7. File Tree of New / Changed Files

```
OmniVoice/
├── examples/
│   ├── scripts/
│   │   ├── phase1_tokenise_urdu.py       ← NEW
│   │   └── run_urdu.sh                   ← NEW
│   └── config/
│       ├── train_config_urdu.json        ← NEW
│       └── data_config_urdu.json         ← NEW (template; resolved at runtime)
└── omnivoice/
    ├── cli/
    │   └── train.py                      ← EDIT (§3.2 A, B, C)
    └── utils/
        └── hf_shards.py                  ← NEW
```

Do **not** modify `extract_audio_tokens.py` — call it as a subprocess or
import its core function if it exposes one.

---

## 8. Strict Rules for Claude

1. **Never delete ledger file** under any circumstance.
2. **Never reset optimizer/scheduler on resume** — this is pretraining, not
   fine-tuning.  Loading `accelerator.load_state()` is sufficient.
3. **Always write ledger before deleting local files** — crash safety.
4. **Keep changes to `train.py` minimal** — add blocks, do not rewrite.
5. **All HF uploads use `exist_ok=True`** on `create_repo` — idempotent.
6. **No hardcoded shard counts** in data configs — always discover dynamically.
7. **Resample audio to 24 kHz** before passing to tokenizer.
8. **Test Phase 1 on `chunk_size=1`** first before running bulk.
9. **Phase 2 must be launchable independently** — it must not depend on Phase 1
   being run in the same session or machine.
10. Use `accelerator.is_main_process` guards for all HF API calls and file
    writes to avoid duplicate uploads in multi-GPU runs.

---

## 9. Quick Test Commands

```bash
# Test Phase 1 with a single parquet file
python examples/scripts/phase1_tokenise_urdu.py \
    --hf_source_repo Humair332/Urdu-munch-1 \
    --hf_processed_repo YourOrg/urdu-omnivoice-tokens-test \
    --chunk_size 1 \
    --gpu_ids 0 \
    --nj_per_gpu 1

# Resume Phase 1 (ledger will skip already-processed files)
python examples/scripts/phase1_tokenise_urdu.py \
    --hf_source_repo Humair332/Urdu-munch-1 \
    --hf_processed_repo YourOrg/urdu-omnivoice-tokens \
    --chunk_size 5

# Test Phase 2 resume from HF Hub checkpoint
HF_TOKEN=xxx \
HF_PROCESSED_REPO=YourOrg/urdu-omnivoice-tokens \
bash examples/scripts/run_urdu.sh
# Then edit train_config_urdu.json: set resume_from_checkpoint to
# "YourOrg/omnivoice-urdu:checkpoint-1000" and rerun.
```

---

## 10. Common Pitfalls to Avoid

| Pitfall | Fix |
|---|---|
| `audio` column is raw bytes, not decoded | Use `decode_audio()` helper in §2.1 |
| Multiple GPUs all upload shards | Guard with `accelerator.is_main_process` |
| Crash between upload and ledger write | Always write ledger **after** upload succeeds |
| Shard numbering collides across chunks | Use global shard counter from ledger |
| HF rate limits on large uploads | Use `upload_folder` not per-file `upload_file` |
| DataLoader doesn't skip batches on resume | Use `accelerator.skip_first_batches` |
| Optimizer LR reset on resume | Do NOT recreate scheduler after `load_state` |
