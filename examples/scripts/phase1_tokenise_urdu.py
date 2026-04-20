#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Phase 1: Urdu Data Ingestion & Tokenisation.
Processes parquet/arrow files from a source HF repo, extracts audio tokens,
and uploads WebDataset shards to a destination HF repo.
Uses Subprocess Isolation for strict RAM management and GPU for resampling.
"""

import argparse
import datetime
import gc
import io
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
from glob import glob
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from tqdm.auto import tqdm

from omnivoice.utils.hf_shards import load_ledger, write_ledger

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def decode_audio(row, col):
    """Decode HF Audio feature into numpy array and sample rate."""
    val = row[col]
    if isinstance(val, dict):
        if "array" in val:
            return val["array"], val["sampling_rate"]
        elif "bytes" in val:
            wav, sr = sf.read(io.BytesIO(val["bytes"]))
            return wav, sr
        elif "path" in val and os.path.exists(val["path"]):
            wav, sr = sf.read(val["path"])
            return wav, sr
    raise ValueError(f"Unknown audio format for column {col}")


def process_single_file_isolated(pf, args, output_manifest_path, device_str):
    """
    Subprocess function: Processes a single parquet/arrow file.
    Runs on a specific device (CPU or GPU).
    """
    import gc
    device = torch.device(device_str)
    
    # Download file
    local_pf = hf_hub_download(
        repo_id=args.hf_source_repo,
        repo_type="dataset",
        filename=pf,
        token=args.hf_token
    )
    
    # Determine dataset type
    ext = pf.split(".")[-1].lower()
    ds_type = "parquet" if ext == "parquet" else "arrow"
    
    # Load dataset (memory-mapped)
    ds = load_dataset(ds_type, data_files=local_pf, split="train", keep_in_memory=False)
    
    wav_dir = Path(args.output_dir) / "wavs"
    manifest_entries = []
    
    # Pre-setup resampler on GPU if needed
    resamplers = {}

    for i, row in enumerate(tqdm(ds, desc=f"Samples in {pf}", leave=False)):
        try:
            wav_arr, sr = decode_audio(row, args.audio_col)
            wav_tensor = torch.from_numpy(wav_arr).float()
            if len(wav_tensor.shape) == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            
            # Resample to 24kHz using GPU if available
            if sr != 24000:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000).to(device)
                
                wav_tensor = wav_tensor.to(device)
                wav_tensor = resamplers[sr](wav_tensor)
                wav_tensor = wav_tensor.to("cpu")
            
            # Save to temp WAV
            orig_id = row.get("id", f"{pf.replace('/', '_')}_{i}")
            sample_id = str(orig_id)
            wav_path = wav_dir / f"{sample_id}.wav"
            torchaudio.save(wav_path, wav_tensor, 24000)
            
            duration = float(wav_tensor.shape[1] / 24000)
            manifest_entries.append({
                "id": sample_id,
                "audio_path": str(wav_path.absolute()),
                "text": str(row[args.text_col]),
                "audio_duration": duration,
                "language_id": "ur",
                "instruct": "None"
            })
        except Exception as e:
            pass # logging might be tricky in subprocess, keep it simple

    # Write partial manifest for this file
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Cleanup
    del ds
    del manifest_entries
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def process_chunk(chunk_files, args, ledger, api):
    """Process a chunk of files using subprocess isolation for RAM safety."""
    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wavs"
    shard_dir = output_dir / "shards"
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    wav_dir.mkdir(parents=True)
    shard_dir.mkdir(parents=True)

    gpu_id = args.gpu_ids.split(",")[0]
    device_str = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    
    ctx = mp.get_context("spawn")
    partial_manifests = []
    
    logger.info(f"Processing chunk of {len(chunk_files)} files with isolation...")
    for pf in tqdm(chunk_files, desc="Files in Chunk"):
        partial_manifest_path = output_dir / f"manifest_{pf.replace('/', '_')}.jsonl"
        
        # Start subprocess
        p = ctx.Process(
            target=process_single_file_isolated,
            args=(pf, args, partial_manifest_path, device_str)
        )
        p.start()
        p.join()
        
        if p.exitcode != 0:
            logger.error(f"Subprocess failed for {pf} with exit code {p.exitcode}")
        elif partial_manifest_path.exists():
            partial_manifests.append(partial_manifest_path)
            
        # Hard GC in parent too
        gc.collect()

    # Merge partial manifests
    merged_manifest_path = output_dir / "chunk_manifest.jsonl"
    with open(merged_manifest_path, "w", encoding="utf-8") as out_f:
        for p_man in partial_manifests:
            with open(p_man, "r") as in_f:
                shutil.copyfileobj(in_f, out_f)

    if not merged_manifest_path.exists() or merged_manifest_path.stat().st_size == 0:
        logger.warning("No valid entries in this chunk. Skipping.")
        return

    # Run tokenisation
    logger.info("Running audio tokenisation...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    env["PYTHONUNBUFFERED"] = "1"
    
    cmd = [
        "python", "-u", "-m", "omnivoice.scripts.extract_audio_tokens",
        "--input_jsonl", str(merged_manifest_path),
        "--tar_output_pattern", str(shard_dir / "shard-%06d.tar"),
        "--jsonl_output_pattern", str(shard_dir / "shard-%06d.jsonl"),
        "--tokenizer_path", args.tokenizer_path,
        "--nj_per_gpu", str(args.nj_per_gpu),
        "--loader_workers", "2",
        "--samples_per_shard", "5000",
        "--min_num_shards", "1",
        "--skip_errors"
    ]
    subprocess.run(cmd, env=env, check=True)

    # Rename and upload shards
    produced_shards = sorted(glob(str(shard_dir / "*.tar")))
    num_shards = len(produced_shards)
    start_id = ledger["next_shard_id"]
    
    for i, tar_file in enumerate(produced_shards):
        global_id = start_id + i
        new_tar = shard_dir / f"shard-{global_id:06d}.tar"
        new_jsonl = shard_dir / f"shard-{global_id:06d}.jsonl"
        os.rename(tar_file, new_tar)
        os.rename(tar_file.replace(".tar", ".jsonl"), new_jsonl)
        
        api.upload_file(path_or_fileobj=str(new_tar), path_in_repo=f"data/{new_tar.name}", 
                        repo_id=args.hf_processed_repo, repo_type="dataset", token=args.hf_token)
        api.upload_file(path_or_fileobj=str(new_jsonl), path_in_repo=f"data/{new_jsonl.name}", 
                        repo_id=args.hf_processed_repo, repo_type="dataset", token=args.hf_token)

    # Update ledger and remote progress.json
    ledger["processed_parquet_files"].extend(chunk_files)
    ledger["next_shard_id"] += num_shards
    ledger["last_updated"] = datetime.datetime.now().isoformat()
    write_ledger(args.ledger_path, ledger)
    
    api.upload_file(path_or_fileobj=args.ledger_path, path_in_repo="progress.json",
                    repo_id=args.hf_processed_repo, repo_type="dataset", token=args.hf_token)
    logger.info(f"Chunk completed. Next Shard ID: {ledger['next_shard_id']}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Urdu Tokenisation (Hardware Optimized)")
    parser.add_argument("--hf_source_repo", default="Humair332/Urdu-munch-1")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--hf_processed_repo", required=True)
    parser.add_argument("--tokenizer_path", default="eustlb/higgs-audio-v2-tokenizer")
    parser.add_argument("--chunk_size", type=int, default=5)
    parser.add_argument("--output_dir", default="tmp_phase1")
    parser.add_argument("--ledger_path", default="progress.json")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--nj_per_gpu", type=int, default=2)
    parser.add_argument("--audio_col", default="audio")
    parser.add_argument("--text_col", default="text")
    args = parser.parse_args()

    api = HfApi(token=args.hf_token)
    api.create_repo(args.hf_processed_repo, repo_type="dataset", exist_ok=True)

    # Sync remote progress
    try:
        if "progress.json" in api.list_repo_files(args.hf_processed_repo, repo_type="dataset"):
            hf_hub_download(repo_id=args.hf_processed_repo, repo_type="dataset", filename="progress.json",
                            local_dir=".", local_dir_use_symlinks=False, token=args.hf_token)
    except Exception: pass

    ledger = load_ledger(args.ledger_path)
    all_files = list_repo_files(args.hf_source_repo, repo_type="dataset", token=args.hf_token)
    valid_files = sorted([f for f in all_files if f.endswith(".parquet") or f.endswith(".arrow")])
    remaining = [f for f in valid_files if f not in ledger["processed_parquet_files"]]
    
    logger.info(f"Found {len(valid_files)} files. {len(remaining)} remaining.")
    for i in range(0, len(remaining), args.chunk_size):
        process_chunk(remaining[i : i + args.chunk_size], args, ledger, api)
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)

if __name__ == "__main__":
    main()
