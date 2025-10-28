#!/usr/bin/env python3
"""
统一的数据准备脚本
下载并验证所有训练需要的数据集和文件

目录结构:
data/
├── base_data/           # FineWeb-Edu 基础训练数据
├── eval_bundle/         # 评估数据包
├── identity_conversations.jsonl  # 身份对话数据
├── smoltalk/           # SmolTalk 数据集
├── mmlu/               # MMLU 数据集
├── humaneval/          # HumanEval 数据集
├── gsm8k/              # GSM8K 数据集
└── arc/                # ARC 数据集
"""

import os
import sys
import time
import argparse
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset

# 数据集配置
HUGGINGFACE_DATASETS = [
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

# S3 数据配置
S3_FILES = [
    {
        "name": "eval_bundle",
        "url": "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip",
        "type": "zip",
        "extract_to": "eval_bundle"
    },
    {
        "name": "identity_conversations.jsonl",
        "url": "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl",
        "type": "file"
    }
]

# FineWeb-Edu 配置
BASE_DATA_CONFIG = {
    "url_template": "https://www.modelscope.cn/datasets/Thackeray/karpathy-fineweb-edu-100b-shuffle-240shard/resolve/master/shard_{:05d}.parquet",
    "max_shard": 1822,
    "local_dir": "base_data"
}


def download_file_with_retry(url: str, filepath: str, max_attempts: int = 5) -> bool:
    """带重试机制的文件下载"""
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"  尝试 {attempt}/{max_attempts}: 下载 {os.path.basename(filepath)}...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            # 写入临时文件
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)

            # 移动到最终位置
            os.rename(temp_path, filepath)
            print(f"  ✓ 下载成功: {os.path.basename(filepath)}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"  ✗ 下载失败: {e}")
            # 清理部分文件
            for path in [temp_path, filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass

            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"  ✗ 下载失败，已达最大重试次数")
                return False

    return False


def download_s3_files(data_dir: Path) -> bool:
    """下载 S3 文件"""
    print("\n" + "="*60)
    print("步骤 1: 下载 S3 文件")
    print("="*60)

    all_success = True

    for file_config in S3_FILES:
        name = file_config["name"]
        url = file_config["url"]
        file_type = file_config["type"]

        if file_type == "zip":
            # ZIP 文件需要解压
            extract_to = data_dir / file_config["extract_to"]
            if extract_to.exists() and any(extract_to.iterdir()):
                print(f"✓ {name} 已存在，跳过")
                continue

            print(f"\n下载并解压 {name}...")
            zip_path = data_dir / f"{name}.zip"

            if download_file_with_retry(url, str(zip_path)):
                try:
                    extract_to.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(data_dir)
                    os.remove(zip_path)
                    print(f"  ✓ 解压成功: {extract_to}")
                except Exception as e:
                    print(f"  ✗ 解压失败: {e}")
                    all_success = False
            else:
                all_success = False

        else:
            # 普通文件
            filepath = data_dir / name
            if filepath.exists():
                print(f"✓ {name} 已存在，跳过")
                continue

            print(f"\n下载 {name}...")
            if not download_file_with_retry(url, str(filepath)):
                all_success = False

    return all_success


def download_huggingface_dataset(dataset_config: dict, data_dir: Path, force: bool = False) -> bool:
    """下载单个 HuggingFace 数据集"""
    name = dataset_config["name"]
    repo = dataset_config["repo"]
    local_name = dataset_config["local_name"]

    print(f"\n下载 {name}...")

    dataset_dir = data_dir / local_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    configs = dataset_config.get("configs", [None])
    splits = dataset_config["splits"]

    all_success = True

    for config in configs:
        for split in splits:
            try:
                # 确定本地文件路径
                if config:
                    file_name = f"{config}_{split}.parquet"
                    local_path = dataset_dir / file_name
                    if local_path.exists() and not force:
                        print(f"  ✓ {config}/{split} 已存在，跳过")
                        continue

                    print(f"  下载 {config}/{split}...")
                    ds = load_dataset(repo, name=config, split=split)
                else:
                    file_name = f"{split}.parquet"
                    local_path = dataset_dir / file_name
                    if local_path.exists() and not force:
                        print(f"  ✓ {split} 已存在，跳过")
                        continue

                    print(f"  下载 {split}...")
                    ds = load_dataset(repo, split=split)

                # 保存为 parquet
                ds.to_parquet(str(local_path))
                print(f"  ✓ 保存到 {local_path}")

            except Exception as e:
                print(f"  ✗ 下载失败 {config}/{split}: {e}")
                all_success = False

    return all_success


def download_huggingface_datasets(data_dir: Path, force: bool = False) -> bool:
    """下载所有 HuggingFace 数据集"""
    print("\n" + "="*60)
    print("步骤 2: 下载 HuggingFace 数据集")
    print("="*60)

    all_success = True

    for dataset_config in HUGGINGFACE_DATASETS:
        if not download_huggingface_dataset(dataset_config, data_dir, force):
            all_success = False

    return all_success


def download_base_data(data_dir: Path, num_shards: int = -1, num_workers: int = 1) -> bool:
    """下载基础训练数据 (FineWeb-Edu)"""
    print("\n" + "="*60)
    print("步骤 3: 下载基础训练数据 (FineWeb-Edu)")
    print("="*60)

    base_data_dir = data_dir / BASE_DATA_CONFIG["local_dir"]
    base_data_dir.mkdir(parents=True, exist_ok=True)

    max_shard = BASE_DATA_CONFIG["max_shard"]
    num = max_shard + 1 if num_shards == -1 else min(num_shards, max_shard + 1)

    print(f"准备下载 {num} 个 shards (0-{num-1})")
    print(f"目标目录: {base_data_dir}")

    # 检查已存在的文件
    existing_files = list(base_data_dir.glob("shard_*.parquet"))
    print(f"已存在 {len(existing_files)} 个 shard 文件")

    if num_workers > 1:
        # 多进程下载
        from multiprocessing import Pool

        def download_shard(index: int) -> bool:
            filename = f"shard_{index:05d}.parquet"
            filepath = base_data_dir / filename

            if filepath.exists():
                return True

            url = BASE_DATA_CONFIG["url_template"].format(index)
            return download_file_with_retry(url, str(filepath))

        print(f"使用 {num_workers} 个进程并行下载...")
        with Pool(processes=num_workers) as pool:
            results = pool.map(download_shard, range(num))

        success_count = sum(1 for r in results if r)
        print(f"\n下载完成: {success_count}/{num} 个 shards")
        return success_count == num
    else:
        # 单进程下载
        success_count = 0
        for i in range(num):
            filename = f"shard_{i:05d}.parquet"
            filepath = base_data_dir / filename

            if filepath.exists():
                success_count += 1
                continue

            url = BASE_DATA_CONFIG["url_template"].format(i)
            if download_file_with_retry(url, str(filepath)):
                success_count += 1

        print(f"\n下载完成: {success_count}/{num} 个 shards")
        return success_count == num


def check_data_integrity(data_dir: Path) -> Tuple[bool, List[str]]:
    """检查数据完整性"""
    print("\n" + "="*60)
    print("数据完整性检查")
    print("="*60)

    missing_items = []

    # 检查 S3 文件
    print("\n检查 S3 文件:")
    for file_config in S3_FILES:
        name = file_config["name"]
        if file_config["type"] == "zip":
            path = data_dir / file_config["extract_to"]
            if not path.exists() or not any(path.iterdir()):
                print(f"  ✗ 缺失: {name}")
                missing_items.append(f"S3: {name}")
            else:
                print(f"  ✓ {name}")
        else:
            path = data_dir / name
            if not path.exists():
                print(f"  ✗ 缺失: {name}")
                missing_items.append(f"S3: {name}")
            else:
                print(f"  ✓ {name}")

    # 检查 HuggingFace 数据集
    print("\n检查 HuggingFace 数据集:")
    for dataset_config in HUGGINGFACE_DATASETS:
        local_name = dataset_config["local_name"]
        dataset_dir = data_dir / local_name

        if not dataset_dir.exists():
            print(f"  ✗ 缺失整个数据集: {local_name}")
            missing_items.append(f"HF Dataset: {local_name}")
            continue

        configs = dataset_config.get("configs", [None])
        splits = dataset_config["splits"]
        missing_files = []

        for config in configs:
            for split in splits:
                if config:
                    file_name = f"{config}_{split}.parquet"
                else:
                    file_name = f"{split}.parquet"

                if not (dataset_dir / file_name).exists():
                    missing_files.append(file_name)

        if missing_files:
            print(f"  ✗ {local_name}: 缺失 {len(missing_files)} 个文件")
            for f in missing_files[:3]:  # 只显示前3个
                print(f"      - {f}")
            if len(missing_files) > 3:
                print(f"      ... 还有 {len(missing_files) - 3} 个文件")
            missing_items.append(f"HF Dataset files: {local_name}")
        else:
            print(f"  ✓ {local_name}")

    # 检查基础训练数据
    print("\n检查基础训练数据:")
    base_data_dir = data_dir / BASE_DATA_CONFIG["local_dir"]
    if not base_data_dir.exists():
        print(f"  ✗ 缺失基础训练数据目录")
        missing_items.append("Base training data")
    else:
        parquet_files = list(base_data_dir.glob("shard_*.parquet"))
        if len(parquet_files) == 0:
            print(f"  ✗ 没有找到任何 shard 文件")
            missing_items.append("Base training data shards")
        else:
            print(f"  ✓ 找到 {len(parquet_files)} 个 shard 文件")

    # 总结
    print("\n" + "="*60)
    if missing_items:
        print("数据完整性检查: 失败")
        print(f"缺失 {len(missing_items)} 个项目")
        return False, missing_items
    else:
        print("数据完整性检查: 通过 ✓")
        return True, []


def main():
    parser = argparse.ArgumentParser(description="准备训练所需的所有数据")
    parser.add_argument("--data-dir", default="./data", help="数据存储目录 (默认: ./data)")
    parser.add_argument("--check-only", action="store_true", help="仅检查数据完整性，不下载")
    parser.add_argument("--force", action="store_true", help="强制重新下载已存在的文件")
    parser.add_argument("--skip-s3", action="store_true", help="跳过 S3 文件下载")
    parser.add_argument("--skip-hf", action="store_true", help="跳过 HuggingFace 数据集下载")
    parser.add_argument("--skip-base", action="store_true", help="跳过基础训练数据下载")
    parser.add_argument("--num-base-shards", type=int, default=-1, help="基础训练数据 shard 数量 (-1 = 全部)")
    parser.add_argument("--num-workers", type=int, default=4, help="并行下载进程数 (默认: 4)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("数据准备脚本")
    print("="*60)
    print(f"数据目录: {data_dir.absolute()}")
    print("")

    if args.check_only:
        # 仅检查数据完整性
        is_complete, missing = check_data_integrity(data_dir)
        if is_complete:
            print("\n✓ 所有数据已准备完毕！")
            sys.exit(0)
        else:
            print("\n✗ 数据不完整，请运行数据下载:")
            print(f"  python {sys.argv[0]} --data-dir {args.data_dir}")
            sys.exit(1)

    # 下载数据
    all_success = True

    if not args.skip_s3:
        if not download_s3_files(data_dir):
            all_success = False
            print("\n⚠ S3 文件下载未完全成功")

    if not args.skip_hf:
        if not download_huggingface_datasets(data_dir, args.force):
            all_success = False
            print("\n⚠ HuggingFace 数据集下载未完全成功")

    if not args.skip_base:
        if not download_base_data(data_dir, args.num_base_shards, args.num_workers):
            all_success = False
            print("\n⚠ 基础训练数据下载未完全成功")

    # 最终检查
    is_complete, missing = check_data_integrity(data_dir)

    if is_complete and all_success:
        print("\n" + "="*60)
        print("✓ 数据准备完成！所有数据已就绪")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("⚠ 数据准备未完全完成")
        if missing:
            print("\n缺失的数据项:")
            for item in missing:
                print(f"  - {item}")
        print("\n请重新运行此脚本或手动下载缺失的数据")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
