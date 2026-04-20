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

import json
import logging
import os
from typing import List, Optional

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


def discover_shards(hf_repo: str, token: Optional[str] = None) -> List[str]:
    """
    Find all .tar shards in a HuggingFace dataset repo.
    Returns a list of 'pipe:curl ...' URLs for WebDataset.
    """
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    try:
        files = api.list_repo_files(hf_repo, repo_type="dataset")
    except Exception as e:
        logger.error(f"Failed to list files from HF repo {hf_repo}: {e}")
        return []

    tars = sorted([f for f in files if f.endswith(".tar")])
    base = f"https://huggingface.co/datasets/{hf_repo}/resolve/main"

    return [f"pipe:curl -s -L {base}/{f}" for f in tars]


def write_ledger(path: str, data: dict):
    """
    Atomic JSON write to prevent corruption on crash.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_ledger(path: str) -> dict:
    """
    Load ledger with basic default structure if it doesn't exist.
    """
    if not os.path.exists(path):
        return {
            "processed_parquet_files": [],
            "uploaded_shards": [],
            "next_shard_id": 0,
            "last_updated": None,
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def upload_folder_chunked(
    api: HfApi,
    folder_path: str,
    repo_id: str,
    path_in_repo: str = "data",
    repo_type: str = "dataset",
    commit_message: str = "upload folder",
):
    """
    Upload a folder to HF Hub via upload_folder (handles large numbers of files).
    """
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
