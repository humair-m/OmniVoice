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
Processes parquet files from a source HF repo, extracts audio tokens,
and uploads WebDataset shards to a destination HF repo.
"""

import argparse
import io
import json
import logging
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


def process_chunk(
    chunk_files,
    args,
    ledger,
    api
):
    """Process a chunk of parquet files: Download -> Decode -> Tokenise -> Upload."""
    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wavs"
    shard_dir = output_dir / "shards"
    
    # Cleanup and recreate dirs
    if output_dir.exists():
        shutil.rmtree(output_dir)
    wav_dir.mkdir(parents=True)
    shard_dir.mkdir(parents=True)

    manifest_entries = []
    
    logger.info(f"Downloading and decoding {len(chunk_files)} parquet files...")
    for pf in tqdm(chunk_files, desc="Parquet Files"):
        # Download parquet file
        local_pf = hf_hub_download(
            repo_id=args.hf_source_repo,
            repo_type="dataset",
            filename=pf,
            token=args.hf_token
        )
        
        # Load dataset from the local parquet
        ds = load_dataset("parquet", data_files=local_pf, split="train")
        
        for i, row in enumerate(ds):
            try:
                wav_arr, sr = decode_audio(row, args.audio_col)
                wav_tensor = torch.from_numpy(wav_arr).float()
                if len(wav_tensor.shape) == 1:
                    wav_tensor = wav_tensor.unsqueeze(0)
                
                # Resample to 24kHz
                if sr != 24000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)
                    wav_tensor = resampler(wav_tensor)
                
                # Save to temp WAV
                sample_id = row.get("id", f"{pf.replace('/', '_')}_{i}")
                wav_path = wav_dir / f"{sample_id}.wav"
                torchaudio.save(wav_path, wav_tensor, 24000)
                
                duration = wav_tensor.shape[1] / 24000
                manifest_entries.append({
                    "id": sample_id,
                    "audio_path": str(wav_path.absolute()),
                    "text": row[args.text_col],
                    "audio_duration": duration,
                    "language_id": "ur",
                    "instruct": "None"
                })
            except Exception as e:
                logger.error(f"Failed to process row {i} in {pf}: {e}")

    if not manifest_entries:
        logger.warning("No valid entries in this chunk. Skipping.")
        return

    # Write temp JSONL manifest
    manifest_path = output_dir / "chunk_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Run tokenisation script as a subprocess
    logger.info("Running audio tokenisation...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    cmd = [
        "python", "-m", "omnivoice.scripts.extract_audio_tokens",
        "--input_jsonl", str(manifest_path),
        "--tar_output_pattern", str(shard_dir / "shard-%06d.tar"),
        "--jsonl_output_pattern", str(shard_dir / "shard-%06d.jsonl"),
        "--tokenizer_path", args.tokenizer_path,
        "--nj_per_gpu", str(args.nj_per_gpu),
        "--samples_per_shard", str(1000), # Fixed for now
        "--skip_errors"
    ]
    
    subprocess.run(cmd, env=env, check=True)

    # Rename shards using global counter
    produced_shards = sorted(glob(str(shard_dir / "*.tar")))
    num_shards = len(produced_shards)
    start_id = ledger["next_shard_id"]
    
    logger.info(f"Renaming and uploading {num_shards} shards starting from ID {start_id}...")
    for i, tar_file in enumerate(produced_shards):
        global_id = start_id + i
        new_tar = shard_dir / f"shard-{global_id:06d}.tar"
        new_jsonl = shard_dir / f"shard-{global_id:06d}.jsonl"
        
        os.rename(tar_file, new_tar)
        os.rename(tar_file.replace(".tar", ".jsonl"), new_jsonl)
        
        # Upload
        api.upload_file(
            path_or_fileobj=str(new_tar),
            path_in_repo=f"data/{new_tar.name}",
            repo_id=args.hf_processed_repo,
            repo_type="dataset",
            token=args.hf_token
        )
        api.upload_file(
            path_or_fileobj=str(new_jsonl),
            path_in_repo=f"data/{new_jsonl.name}",
            repo_id=args.hf_processed_repo,
            repo_type="dataset",
            token=args.hf_token
        )

    # Update ledger
    ledger["processed_parquet_files"].extend(chunk_files)
    ledger["next_shard_id"] += num_shards
    ledger["last_updated"] = torch.datetime.datetime.now().isoformat() if hasattr(torch, "datetime") else "" # Fallback
    import datetime
    ledger["last_updated"] = datetime.datetime.now().isoformat()
    
    write_ledger(args.ledger_path, ledger)
    logger.info(f"Chunk completed. Next Shard ID: {ledger['next_shard_id']}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Urdu Tokenisation")
    parser.add_argument("--hf_source_repo", default="Humair332/Urdu-munch-1")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--hf_processed_repo", required=True)
    parser.add_argument("--tokenizer_path", default="eustlb/higgs-audio-v2-tokenizer")
    parser.add_argument("--chunk_size", type=int, default=5)
    parser.add_argument("--output_dir", default="tmp_phase1")
    parser.add_argument("--ledger_path", default="phase1_ledger.json")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--nj_per_gpu", type=int, default=2)
    parser.add_argument("--audio_col", default="audio")
    parser.add_argument("--text_col", default="text")
    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError("HF_TOKEN must be provided via --hf_token or HF_TOKEN env var.")

    api = HfApi(token=args.hf_token)
    api.create_repo(args.hf_processed_repo, repo_type="dataset", exist_ok=True)

    ledger = load_ledger(args.ledger_path)
    
    # List source files
    all_files = list_repo_files(args.hf_source_repo, repo_type="dataset", token=args.hf_token)
    parquet_files = sorted([f for f in all_files if f.endswith(".parquet")])
    
    remaining = [f for f in parquet_files if f not in ledger["processed_parquet_files"]]
    logger.info(f"Found {len(parquet_files)} parquet files. {len(remaining)} remaining.")

    for i in range(0, len(remaining), args.chunk_size):
        chunk = remaining[i : i + args.chunk_size]
        logger.info(f"Processing chunk {i//args.chunk_size + 1}: {chunk}")
        process_chunk(chunk, args, ledger, api)
        
        # Explicit cleanup after each chunk
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)

    logger.info("Phase 1 Complete.")


if __name__ == "__main__":
    main()
