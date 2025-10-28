"""
数据完整性检查模块
在训练开始前验证所有必需的数据是否已准备完毕
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from nanochat.common import get_base_dir


def get_data_dir() -> Path:
    """获取数据目录路径"""
    base_dir = get_base_dir()
    # 优先使用 data 目录，如果不存在则使用 base_dir
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir = Path(base_dir)
    return data_dir


def check_s3_files(data_dir: Path) -> Tuple[bool, List[str]]:
    """检查 S3 文件是否存在"""
    missing = []

    # 检查 eval_bundle
    eval_bundle_path = data_dir / "eval_bundle"
    if not eval_bundle_path.exists() or not any(eval_bundle_path.iterdir()):
        missing.append("eval_bundle (evaluation data)")

    # 检查 identity_conversations.jsonl
    identity_path = data_dir / "identity_conversations.jsonl"
    if not identity_path.exists():
        missing.append("identity_conversations.jsonl")

    return len(missing) == 0, missing


def check_huggingface_datasets(data_dir: Path, required_datasets: List[str] = None) -> Tuple[bool, List[str]]:
    """
    检查 HuggingFace 数据集是否存在

    Args:
        data_dir: 数据目录
        required_datasets: 需要检查的数据集列表，如果为 None 则检查所有
    """
    # 所有可用的数据集配置
    all_datasets = {
        "smoltalk": {
            "splits": ["train", "test"],
        },
        "mmlu": {
            "configs": ["all", "auxiliary_train"],
            "splits": ["train", "validation", "dev", "test"],
        },
        "humaneval": {
            "splits": ["test"],
        },
        "gsm8k": {
            "configs": ["main", "socratic"],
            "splits": ["train", "test"],
        },
        "arc": {
            "configs": ["ARC-Easy", "ARC-Challenge"],
            "splits": ["train", "validation", "test"],
        },
    }

    # 如果没有指定，检查所有数据集
    if required_datasets is None:
        required_datasets = list(all_datasets.keys())

    missing = []

    for dataset_name in required_datasets:
        if dataset_name not in all_datasets:
            missing.append(f"Unknown dataset: {dataset_name}")
            continue

        dataset_config = all_datasets[dataset_name]
        dataset_dir = data_dir / dataset_name

        if not dataset_dir.exists():
            missing.append(f"{dataset_name} (entire dataset)")
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
            missing.append(f"{dataset_name}: {', '.join(missing_files)}")

    return len(missing) == 0, missing


def check_base_data(data_dir: Path, min_shards: int = 1) -> Tuple[bool, List[str]]:
    """
    检查基础训练数据是否存在

    Args:
        data_dir: 数据目录
        min_shards: 最少需要的 shard 数量
    """
    missing = []

    base_data_dir = data_dir / "base_data"
    if not base_data_dir.exists():
        missing.append(f"base_data directory")
        return False, missing

    parquet_files = list(base_data_dir.glob("shard_*.parquet"))
    if len(parquet_files) < min_shards:
        missing.append(f"base_data shards (found {len(parquet_files)}, need at least {min_shards})")

    return len(missing) == 0, missing


def check_tokenizer_data(min_shards: int = 1) -> Tuple[bool, List[str]]:
    """检查 tokenizer 训练所需的数据"""
    data_dir = get_data_dir()
    return check_base_data(data_dir, min_shards)


def check_base_training_data(min_shards: int = 10) -> Tuple[bool, List[str]]:
    """检查基础模型训练所需的数据"""
    all_missing = []

    data_dir = get_data_dir()

    # 检查基础训练数据
    ok, missing = check_base_data(data_dir, min_shards)
    if not ok:
        all_missing.extend(missing)

    return len(all_missing) == 0, all_missing


def check_mid_training_data() -> Tuple[bool, List[str]]:
    """检查中期训练所需的数据"""
    all_missing = []

    data_dir = get_data_dir()

    # 检查 HuggingFace 数据集
    required_datasets = ["smoltalk", "mmlu", "gsm8k"]
    ok, missing = check_huggingface_datasets(data_dir, required_datasets)
    if not ok:
        all_missing.extend(missing)

    # 检查 identity_conversations.jsonl
    ok, missing = check_s3_files(data_dir)
    if not ok:
        all_missing.extend(missing)

    return len(all_missing) == 0, all_missing


def check_sft_training_data() -> Tuple[bool, List[str]]:
    """检查 SFT 训练所需的数据"""
    all_missing = []

    data_dir = get_data_dir()

    # 检查 HuggingFace 数据集
    required_datasets = ["arc", "gsm8k", "smoltalk"]
    ok, missing = check_huggingface_datasets(data_dir, required_datasets)
    if not ok:
        all_missing.extend(missing)

    # 检查 identity_conversations.jsonl
    ok, missing = check_s3_files(data_dir)
    if not ok:
        all_missing.extend(missing)

    return len(all_missing) == 0, all_missing


def check_evaluation_data() -> Tuple[bool, List[str]]:
    """检查评估所需的数据"""
    all_missing = []

    data_dir = get_data_dir()

    # 检查 eval_bundle
    ok, missing = check_s3_files(data_dir)
    if not ok:
        all_missing.extend(missing)

    # 检查 HuggingFace 数据集（评估可能需要）
    required_datasets = ["mmlu", "arc", "gsm8k", "humaneval", "smoltalk"]
    ok, missing = check_huggingface_datasets(data_dir, required_datasets)
    if not ok:
        all_missing.extend(missing)

    return len(all_missing) == 0, all_missing


def check_all_data() -> Tuple[bool, Dict[str, List[str]]]:
    """检查所有数据"""
    results = {}

    # 检查各类数据
    ok, missing = check_base_training_data()
    if not ok:
        results["base_training"] = missing

    ok, missing = check_mid_training_data()
    if not ok:
        results["mid_training"] = missing

    ok, missing = check_sft_training_data()
    if not ok:
        results["sft_training"] = missing

    ok, missing = check_evaluation_data()
    if not ok:
        results["evaluation"] = missing

    return len(results) == 0, results


def require_data(check_func, error_message: str = None):
    """
    装饰器：在函数执行前检查数据完整性

    Usage:
        @require_data(check_base_training_data, "Base training data is required")
        def train_base_model():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            ok, missing = check_func()
            if not ok:
                print("\n" + "="*60)
                print("✗ 数据完整性检查失败")
                print("="*60)
                if error_message:
                    print(f"\n{error_message}")
                print("\n缺失的数据:")
                for item in missing:
                    print(f"  - {item}")
                print("\n请运行以下命令下载数据:")
                print("  python -m scripts.prepare_data --data-dir ./data")
                print("="*60 + "\n")
                sys.exit(1)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def print_data_status():
    """打印数据状态摘要"""
    print("\n" + "="*60)
    print("数据状态检查")
    print("="*60)

    checks = [
        ("Tokenizer 训练数据", check_tokenizer_data),
        ("基础模型训练数据", check_base_training_data),
        ("中期训练数据", check_mid_training_data),
        ("SFT 训练数据", check_sft_training_data),
        ("评估数据", check_evaluation_data),
    ]

    all_ok = True
    for name, check_func in checks:
        ok, missing = check_func()
        if ok:
            print(f"✓ {name}")
        else:
            print(f"✗ {name}")
            for item in missing:
                print(f"    - {item}")
            all_ok = False

    print("="*60)

    if all_ok:
        print("✓ 所有数据已准备完毕！\n")
    else:
        print("⚠ 部分数据缺失，请运行:")
        print("  python -m scripts.prepare_data --data-dir ./data\n")

    return all_ok


if __name__ == "__main__":
    # 运行数据状态检查
    ok = print_data_status()
    sys.exit(0 if ok else 1)
