#!/usr/bin/env python3
"""
Download HuggingFace datasets to local storage for offline usage.
This script downloads all datasets used in the nanochat project.
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset

# Dataset configurations
DATASETS = [
    {
        "name": "smol-smoltalk",
        "repo": "HuggingFaceTB/smol-smoltalk",
        "splits": ["train", "test"],
        "local_name": "smoltalk"
    },
    {
        "name": "mmlu",
        "repo": "cais/mmlu",
        "configs": ["all", "auxiliary_train"],
        "splits": ["train", "validation", "dev", "test"],
        "local_name": "mmlu"
    },
    {
        "name": "humaneval",
        "repo": "openai/openai_humaneval",
        "splits": ["test"],
        "local_name": "humaneval"
    },
    {
        "name": "gsm8k",
        "repo": "openai/gsm8k",
        "configs": ["main", "socratic"],
        "splits": ["train", "test"],
        "local_name": "gsm8k"
    },
    {
        "name": "ai2_arc",
        "repo": "allenai/ai2_arc",
        "configs": ["ARC-Easy", "ARC-Challenge"],
        "splits": ["train", "validation", "test"],
        "local_name": "arc"
    }
]

def download_dataset(dataset_config, data_dir, force=False):
    """Download a single dataset to local storage."""
    name = dataset_config["name"]
    repo = dataset_config["repo"]
    local_name = dataset_config["local_name"]

    print(f"Downloading {name}...")

    # Create dataset directory
    dataset_dir = Path(data_dir) / local_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    configs = dataset_config.get("configs", [None])
    splits = dataset_config["splits"]

    for config in configs:
        for split in splits:
            try:
                # Determine local file path
                if config:
                    file_name = f"{config}_{split}.parquet"
                    local_path = dataset_dir / file_name
                    if local_path.exists() and not force:
                        print(f"  Skipping {config}/{split} - already exists")
                        continue

                    print(f"  Downloading {config}/{split}...")
                    ds = load_dataset(repo, name=config, split=split)
                else:
                    file_name = f"{split}.parquet"
                    local_path = dataset_dir / file_name
                    if local_path.exists() and not force:
                        print(f"  Skipping {split} - already exists")
                        continue

                    print(f"  Downloading {split}...")
                    ds = load_dataset(repo, split=split)

                # Save to parquet
                ds.to_parquet(str(local_path))
                print(f"  Saved to {local_path}")

            except Exception as e:
                print(f"  Error downloading {config}/{split}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets for offline usage")
    parser.add_argument("--data-dir", default="./data", help="Directory to store datasets")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    parser.add_argument("--dataset", help="Download only specific dataset (by local name)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_download = DATASETS
    if args.dataset:
        datasets_to_download = [d for d in DATASETS if d["local_name"] == args.dataset]
        if not datasets_to_download:
            print(f"Dataset '{args.dataset}' not found. Available: {[d['local_name'] for d in DATASETS]}")
            return

    print(f"Downloading datasets to {data_dir.absolute()}")

    for dataset_config in datasets_to_download:
        download_dataset(dataset_config, data_dir, args.force)

    print("Download complete!")

if __name__ == "__main__":
    main()
