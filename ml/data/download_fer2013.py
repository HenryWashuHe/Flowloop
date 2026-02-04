#!/usr/bin/env python3
"""
FER2013 Dataset Download and Preparation Script.

Downloads the FER2013 dataset from Kaggle and organizes it into
train/val/test folders with emotion class subfolders.

Usage:
    python download_fer2013.py

Requirements:
    - Kaggle API key configured at ~/.kaggle/kaggle.json
    - pip install kaggle pillow pandas tqdm

Output structure:
    data/processed/fer2013/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── sad/
    │   ├── surprise/
    │   └── neutral/
    ├── val/
    │   └── ... (same structure)
    └── test/
        └── ... (same structure)
"""

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Emotion class mapping (FER2013 uses integers 0-6)
EMOTION_CLASSES = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

# Usage split mapping
USAGE_MAPPING = {
    "Training": "train",
    "PublicTest": "val",
    "PrivateTest": "test",
}


def download_fer2013(data_dir: Path) -> Path:
    """
    Download FER2013 dataset from Kaggle.

    Returns:
        Path to the downloaded CSV file.
    """
    import kaggle

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "fer2013.csv"

    if csv_path.exists():
        print(f"✓ FER2013 already downloaded at {csv_path}")
        return csv_path

    print("Downloading FER2013 from Kaggle...")
    print("(This requires Kaggle API credentials at ~/.kaggle/kaggle.json)")

    try:
        # Download the dataset
        kaggle.api.dataset_download_files(
            "msambare/fer2013",
            path=str(raw_dir),
            unzip=True,
        )
        print(f"✓ Downloaded to {raw_dir}")
    except Exception as e:
        print(f"✗ Kaggle download failed: {e}")
        print("\nTrying alternative dataset (deadskull7/fer2013)...")
        try:
            kaggle.api.dataset_download_files(
                "deadskull7/fer2013",
                path=str(raw_dir),
                unzip=True,
            )
            print(f"✓ Downloaded to {raw_dir}")
        except Exception as e2:
            print(f"✗ Alternative download also failed: {e2}")
            print("\nPlease download manually:")
            print("1. Go to https://www.kaggle.com/datasets/msambare/fer2013")
            print("2. Download and extract to:", raw_dir)
            sys.exit(1)

    # Find the CSV file
    csv_files = list(raw_dir.glob("**/*.csv"))
    if csv_files:
        # Find fer2013.csv specifically or use the first CSV
        for cf in csv_files:
            if "fer2013" in cf.name.lower():
                return cf
        return csv_files[0]

    # If no CSV, check for image folders (newer format)
    if (raw_dir / "train").exists():
        print("✓ Dataset is in image folder format (already organized)")
        return None  # Signal that we have folder format

    raise FileNotFoundError("Could not find FER2013 data after download")


def convert_csv_to_images(csv_path: Path, output_dir: Path) -> None:
    """
    Convert FER2013 CSV format to image folders.

    The CSV has columns: emotion, pixels, Usage
    - emotion: 0-6 integer
    - pixels: space-separated 48x48 grayscale values
    - Usage: Training, PublicTest, or PrivateTest
    """
    print(f"Converting CSV to images: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    # Count per emotion and usage
    print("\nDataset distribution:")
    for usage in df["Usage"].unique():
        subset = df[df["Usage"] == usage]
        print(f"  {usage}: {len(subset)} samples")
        for emo_idx, emo_name in EMOTION_CLASSES.items():
            count = len(subset[subset["emotion"] == emo_idx])
            print(f"    {emo_name}: {count}")

    # Create output directories
    for usage_name in USAGE_MAPPING.values():
        for emotion_name in EMOTION_CLASSES.values():
            (output_dir / usage_name / emotion_name).mkdir(parents=True, exist_ok=True)

    # Convert each row to an image
    counters = {split: {emo: 0 for emo in EMOTION_CLASSES.values()} for split in USAGE_MAPPING.values()}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        emotion_idx = row["emotion"]
        pixels = row["pixels"]
        usage = row["Usage"]

        emotion_name = EMOTION_CLASSES[emotion_idx]
        split_name = USAGE_MAPPING[usage]

        # Convert pixels string to numpy array
        pixel_values = np.array([int(p) for p in pixels.split()], dtype=np.uint8)
        img_array = pixel_values.reshape(48, 48)

        # Create PIL image
        img = Image.fromarray(img_array, mode="L")  # L = grayscale

        # Save image
        img_idx = counters[split_name][emotion_name]
        img_path = output_dir / split_name / emotion_name / f"{emotion_name}_{img_idx:05d}.png"
        img.save(img_path)

        counters[split_name][emotion_name] += 1

    print(f"\n✓ Converted {len(df)} images to {output_dir}")


def organize_image_folders(source_dir: Path, output_dir: Path) -> None:
    """
    Organize already-downloaded image folders into our structure.

    Some Kaggle downloads come pre-organized as:
    train/angry/*.jpg, train/happy/*.jpg, etc.
    """
    print(f"Organizing image folders from {source_dir}")

    for split in ["train", "test"]:
        src_split = source_dir / split
        if not src_split.exists():
            continue

        # Map test -> test in output, but we also create a val split
        out_split = "val" if split == "test" else split

        for emotion_dir in src_split.iterdir():
            if not emotion_dir.is_dir():
                continue

            emotion_name = emotion_dir.name.lower()
            if emotion_name not in EMOTION_CLASSES.values():
                continue

            out_emotion_dir = output_dir / out_split / emotion_name
            out_emotion_dir.mkdir(parents=True, exist_ok=True)

            for img_path in emotion_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    # Copy (or move) the image
                    import shutil
                    dest = out_emotion_dir / img_path.name
                    if not dest.exists():
                        shutil.copy2(img_path, dest)

    print(f"✓ Organized images to {output_dir}")


def compute_class_weights(data_dir: Path) -> dict:
    """Compute class weights for imbalanced dataset."""
    train_dir = data_dir / "train"
    class_counts = {}

    for emotion_name in EMOTION_CLASSES.values():
        emotion_dir = train_dir / emotion_name
        if emotion_dir.exists():
            count = len(list(emotion_dir.glob("*")))
            class_counts[emotion_name] = count

    if not class_counts:
        return {}

    total = sum(class_counts.values())
    num_classes = len(class_counts)

    # Inverse frequency weighting
    weights = {}
    for emotion, count in class_counts.items():
        weights[emotion] = total / (num_classes * count) if count > 0 else 1.0

    return weights


def main():
    """Main entry point."""
    # Get script directory (ml/data/)
    script_dir = Path(__file__).parent
    data_dir = script_dir
    processed_dir = data_dir / "processed" / "fer2013"

    print("=" * 60)
    print("FER2013 Dataset Download and Preparation")
    print("=" * 60)

    # Check if already processed
    if (processed_dir / "train" / "angry").exists():
        train_count = sum(1 for _ in (processed_dir / "train").rglob("*.png"))
        if train_count > 0:
            print(f"✓ Dataset already prepared at {processed_dir}")
            print(f"  Train images: {train_count}")

            # Compute class weights
            weights = compute_class_weights(processed_dir)
            if weights:
                print("\nClass weights (for handling imbalance):")
                for emotion, weight in sorted(weights.items()):
                    print(f"  {emotion}: {weight:.3f}")
            return

    # Download dataset
    csv_path = download_fer2013(data_dir)

    if csv_path is None:
        # Dataset is in image folder format
        raw_dir = data_dir / "raw"
        organize_image_folders(raw_dir, processed_dir)
    elif csv_path.suffix == ".csv":
        # Convert CSV to images
        convert_csv_to_images(csv_path, processed_dir)
    else:
        print(f"Unknown format: {csv_path}")
        sys.exit(1)

    # Compute class weights
    weights = compute_class_weights(processed_dir)
    if weights:
        print("\nClass weights (for handling imbalance):")
        for emotion, weight in sorted(weights.items()):
            print(f"  {emotion}: {weight:.3f}")

        # Save weights to file
        import json
        weights_path = processed_dir / "class_weights.json"
        with open(weights_path, "w") as f:
            json.dump(weights, f, indent=2)
        print(f"\n✓ Saved class weights to {weights_path}")

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"\nYou can now train with:")
    print(f"  python training/train_emotion.py --data_dir {processed_dir}")


if __name__ == "__main__":
    main()
