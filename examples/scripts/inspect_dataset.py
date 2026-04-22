#!/usr/bin/env python3
"""
Dataset inspection script for Urdu OmniVoice training data.
Discovers shards directly from HuggingFace Hub — no pre-generated config needed.

Usage (in Colab):
    python examples/scripts/inspect_dataset.py \
        --hf_token_repo Humair332/urdu-omnivoice-tokens \
        --train_config examples/config/train_config_urdu.json \
        --num_samples 5 \
        --num_batches 2
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from transformers import AutoTokenizer

from omnivoice.data.dataset import WebDatasetReader
from omnivoice.data.processor import OmniVoiceSampleProcessor
from omnivoice.data.batching import PackingIterableDataset
from omnivoice.data.collator import PackingDataCollator
from omnivoice.utils.hf_shards import discover_shards


def resolve_tokenizer_path(tc: dict) -> str:
    """Resolve tokenizer path from train config, handling Hub repo:subfolder syntax."""
    path = (
        tc.get("llm_name_or_path")
        or tc.get("resume_from_checkpoint")
        or tc.get("init_from_checkpoint")
    )
    if not path:
        print("ERROR: set llm_name_or_path or resume_from_checkpoint in train config.")
        sys.exit(1)

    if ":" in path and not path.startswith("/"):
        from huggingface_hub import snapshot_download
        repo_id, subfolder = path.split(":", 1)
        local = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[f"{subfolder}/*"],
            token=os.environ.get("HF_TOKEN"),
        )
        return os.path.join(local, subfolder)

    expanded = os.path.expanduser(path)
    if os.path.exists(expanded):
        return expanded

    from huggingface_hub import snapshot_download
    return snapshot_download(path, token=os.environ.get("HF_TOKEN"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_token_repo",
        default="Humair332/urdu-omnivoice-tokens",
        help="HuggingFace dataset repo containing the .tar shards",
    )
    parser.add_argument("--train_config", required=True, help="Path to training config JSON")
    parser.add_argument("--num_samples", type=int, default=5,  help="Raw samples to inspect")
    parser.add_argument("--num_batches", type=int, default=2,  help="Packed batches to inspect")
    args = parser.parse_args()

    with open(args.train_config) as f:
        tc = json.load(f)

    # ── 1. Tokenizer ────────────────────────────────────────────────────────
    tokenizer_path = resolve_tokenizer_path(tc)
    print(f"\n{'='*60}")
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer vocab size : {len(tokenizer)}")
    print(f"{'='*60}\n")

    # ── 2. Discover shards from HF Hub ──────────────────────────────────────
    print(f"Discovering shards from: {args.hf_token_repo} ...")
    shard_urls = discover_shards(args.hf_token_repo, token=os.environ.get("HF_TOKEN"))
    print(f"Found {len(shard_urls)} shards.\n")

    if not shard_urls:
        print("ERROR: No shards found. Check the HF repo name and HF_TOKEN.")
        sys.exit(1)

    # Build manifests in the format WebDatasetReader expects:
    # (tar_path, jsonl_path, num_items, num_seconds)
    manifests = [(url, "", 1000, 3600.0) for url in shard_urls]

    # ── 3. Raw sample inspection ─────────────────────────────────────────────
    raw_ds = WebDatasetReader(manifests=manifests, evaluation=False)

    print(f"{'='*60}")
    print(f"{'  RAW SAMPLE INSPECTION  ':^60}")
    print(f"{'='*60}")

    for i, sample in enumerate(raw_ds):
        if i >= args.num_samples:
            break

        text      = sample.get("text", "N/A")
        lang      = sample.get("language_id", "N/A")
        num_tokens = sample.get("num_tokens", "N/A")
        duration  = sample.get("audio_duration", "N/A")
        tokens    = sample.get("tokens")

        print(f"\n[Sample {i + 1}]")
        print(f"  language_id  : {lang}")
        print(f"  text         : {str(text)[:300]}")
        if isinstance(duration, float):
            print(f"  audio_dur    : {duration:.2f}s")
        else:
            print(f"  audio_dur    : {duration}")
        print(f"  num_tokens   : {num_tokens}  (frames)")
        if tokens is not None:
            t = torch.as_tensor(tokens)
            print(f"  tokens shape : {tuple(t.shape)}  → (codebooks={t.shape[0]}, frames={t.shape[1]})")
            print(f"  actual IDs   : {t.shape[0] * t.shape[1]}  total IDs stored")
        print(f"  sample keys  : {list(sample.keys())}")

    # ── 4. Packed batch inspection ───────────────────────────────────────────
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

        input_ids  = batch["input_ids"]    # [1, C, L]
        audio_mask = batch["audio_mask"]   # [1, L]
        doc_ids    = batch["document_ids"] # [1, L]

        seq_len       = input_ids.shape[-1]
        text_frames   = (~audio_mask[0].bool()).sum().item()
        audio_frames  = audio_mask[0].bool().sum().item()
        num_docs      = int(doc_ids[0].max().item()) + 1

        # Decode ONLY text positions using audio_mask — no garbage
        text_ids = input_ids[0, 0, :][~audio_mask[0].bool()]
        decoded  = tokenizer.decode(text_ids.tolist(), skip_special_tokens=True)

        print(f"\n[Batch {i + 1}]")
        print(f"  Shape                : {tuple(input_ids.shape)}  [batch, codebooks, frames]")
        print(f"  Total frames         : {seq_len}  (batch_tokens = {batch_tokens})")
        print(f"  Text frames          : {text_frames}  ({100 * text_frames / seq_len:.1f}%)")
        print(f"  Audio frames         : {audio_frames}  ({100 * audio_frames / seq_len:.1f}%)")
        print(f"  Packed samples       : {num_docs} clips")
        print(f"  Approx audio         : {audio_frames / 25:.1f}s  (@25 frames/sec)")
        print(f"\n  ── Clean decoded text (text positions only) ──")
        print(f"  {decoded[:600]}")
        print()


if __name__ == "__main__":
    main()
