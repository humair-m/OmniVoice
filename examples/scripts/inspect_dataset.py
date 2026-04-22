#!/usr/bin/env python3
"""
Dataset inspection script for Urdu OmniVoice training data.

Usage (in Colab):
    python examples/scripts/inspect_dataset.py \
        --data_config examples/config/data_config_urdu_resolved.json \
        --train_config examples/config/train_config_urdu.json \
        --num_samples 5
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from transformers import AutoTokenizer

from omnivoice.data.dataset import WebDatasetReader, prepare_data_manifests_from_json
from omnivoice.data.processor import OmniVoiceSampleProcessor
from omnivoice.data.batching import PackingIterableDataset
from omnivoice.data.collator import PackingDataCollator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", required=True, help="Path to resolved data config JSON")
    parser.add_argument("--train_config", required=True, help="Path to training config JSON")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of raw samples to inspect")
    parser.add_argument("--num_batches", type=int, default=2, help="Number of packed batches to inspect")
    args = parser.parse_args()

    with open(args.train_config) as f:
        tc = json.load(f)

    # ── 1. Tokenizer ────────────────────────────────────────────────────────
    tokenizer_path = tc.get("llm_name_or_path") or tc.get("resume_from_checkpoint") or tc.get("init_from_checkpoint")
    if not tokenizer_path:
        print("ERROR: set llm_name_or_path or resume_from_checkpoint in train config to load tokenizer.")
        sys.exit(1)

    # Handle "repo:subfolder" syntax
    if ":" in tokenizer_path and not tokenizer_path.startswith("/"):
        from huggingface_hub import snapshot_download
        repo_id, subfolder = tokenizer_path.split(":", 1)
        local = snapshot_download(repo_id=repo_id, repo_type="model",
                                  allow_patterns=[f"{subfolder}/*"],
                                  token=os.environ.get("HF_TOKEN"))
        tokenizer_path = os.path.join(local, subfolder)

    print(f"\n{'='*60}")
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer vocab size : {len(tokenizer)}")
    print(f"{'='*60}\n")

    # ── 2. Raw shard samples ─────────────────────────────────────────────────
    train_manifests, _ = prepare_data_manifests_from_json(args.data_config)
    print(f"Found {len(train_manifests)} shards in data config.\n")

    raw_ds = WebDatasetReader(manifests=train_manifests, evaluation=False)

    print(f"{'='*60}")
    print(f"{'  RAW SAMPLE INSPECTION  ':^60}")
    print(f"{'='*60}")

    for i, sample in enumerate(raw_ds):
        if i >= args.num_samples:
            break

        text = sample.get("text", "N/A")
        lang = sample.get("language_id", "N/A")
        num_tokens = sample.get("num_tokens", "N/A")
        duration = sample.get("audio_duration", "N/A")
        tokens = sample.get("tokens")  # np array if pre-tokenized

        print(f"\n[Sample {i+1}]")
        print(f"  language_id  : {lang}")
        print(f"  text         : {str(text)[:200]}")
        print(f"  audio_dur    : {duration:.2f}s" if isinstance(duration, float) else f"  audio_dur    : {duration}")
        print(f"  num_tokens   : {num_tokens}")
        if tokens is not None:
            t = torch.as_tensor(tokens)
            print(f"  tokens shape : {tuple(t.shape)}  (codebooks × frames)")
        print(f"  sample keys  : {list(sample.keys())}")

    # ── 3. Processed batch inspection ───────────────────────────────────────
    processor = OmniVoiceSampleProcessor(
        text_tokenizer=tokenizer,
        num_channels=tc.get("num_audio_codebook", 8),
        audio_mask_id=tc.get("audio_mask_id", 1024),
        prompt_ratio_range=tc.get("prompt_ratio_range", [0.0, 0.3]),
        mask_ratio_range=tc.get("mask_ratio_range", [0.0, 1.0]),
        drop_cond_ratio=tc.get("drop_cond_ratio", 0.1),
        language_ratio=tc.get("language_ratio", 0.0),
        use_pinyin_ratio=tc.get("use_pinyin_ratio", 0.0),
        instruct_ratio=tc.get("instruct_ratio", 0.0),
        only_instruct_ratio=tc.get("only_instruct_ratio", 0.0),
    )

    batch_tokens = tc.get("batch_tokens", 8192)
    packed_ds = PackingIterableDataset(raw_ds, processor, batch_tokens)
    collator = PackingDataCollator(processor, batch_tokens)

    from torch.utils.data import DataLoader
    loader = DataLoader(packed_ds, batch_size=None, collate_fn=collator, num_workers=0)

    print(f"\n{'='*60}")
    print(f"{'  PACKED BATCH INSPECTION  ':^60}")
    print(f"{'='*60}")

    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break

        input_ids  = batch["input_ids"]   # [1, C, L]
        audio_mask = batch["audio_mask"]  # [1, L]
        labels     = batch["labels"]      # [1, C, L]
        doc_ids    = batch["document_ids"]

        seq_len    = input_ids.shape[-1]
        num_codebooks = input_ids.shape[-2]
        num_docs   = doc_ids[0].max().item() + 1
        text_frames = (~audio_mask[0].bool()).sum().item()
        audio_frames = audio_mask[0].bool().sum().item()

        # Decode only text frames (codebook 0, text positions)
        text_ids_only = input_ids[0, 0, :][~audio_mask[0].bool()]
        decoded = tokenizer.decode(text_ids_only.tolist(), skip_special_tokens=True)

        print(f"\n[Batch {i+1}]")
        print(f"  Shape                : input_ids {tuple(input_ids.shape)}  [batch, codebooks, frames]")
        print(f"  Frames total         : {seq_len}  (batch_tokens = {batch_tokens})")
        print(f"  Text frames          : {text_frames}  ({100*text_frames/seq_len:.1f}%)")
        print(f"  Audio frames         : {audio_frames}  ({100*audio_frames/seq_len:.1f}%)")
        print(f"  Packed samples       : {int(num_docs)} samples in this batch")
        print(f"  Approx audio seconds : {audio_frames / 25:.1f}s  (at 25 frames/sec)")
        print(f"\n  ── Decoded text (text positions only) ──")
        print(f"  {decoded[:800]}")
        print()


if __name__ == "__main__":
    main()
