"""
FER2013 Dataset Preparation Script.

Downloads and prepares the FER2013 dataset for emotion classification training.

Dataset structure:
- 35,887 grayscale 48x48 images
- 7 emotion classes: angry, disgust, fear, happy, sad, surprise, neutral
- Split: Training (28,709), PublicTest (3,589), PrivateTest (3,589)

Usage:
    python prepare_fer2013.py --output_dir ./data/processed/fer2013

Requirements:
    - Kaggle API credentials configured (~/.kaggle/kaggle.json)
    - pip install kaggle
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def download_fer2013(output_dir: Path) -> Path:
    """
    Download FER2013 dataset from Kaggle.

    Args:
        output_dir: Directory to save downloaded files

    Returns:
        Path to the fer2013.csv file
    """
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "fer2013.csv"

    if csv_path.exists():
        print(f"Dataset already exists at {csv_path}")
        return csv_path

    print("Downloading FER2013 from Kaggle...")
    print("Note: You need Kaggle API credentials configured.")
    print("See: https://github.com/Kaggle/kaggle-api#api-credentials")

    try:
        import kaggle

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "deadskull7/fer2013",
            path=str(raw_dir),
            unzip=True,
        )
        print(f"Downloaded to {raw_dir}")
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nManual download instructions:")
        print("1. Go to https://www.kaggle.com/datasets/deadskull7/fer2013")
        print("2. Download fer2013.csv")
        print(f"3. Place it in {raw_dir}")
        raise

    return csv_path


def parse_pixels(pixel_string: str) -> np.ndarray:
    """Convert space-separated pixel string to numpy array."""
    pixels = np.array([int(p) for p in pixel_string.split()], dtype=np.uint8)
    return pixels.reshape(48, 48)


def prepare_dataset(
    csv_path: Path,
    output_dir: Path,
    resize_to: int = 224,
    create_rgb: bool = True,
) -> dict[str, int]:
    """
    Process FER2013 CSV and save as image files.

    Args:
        csv_path: Path to fer2013.csv
        output_dir: Output directory for processed images
        resize_to: Target size for images (default: 224 for EfficientNet)
        create_rgb: Convert grayscale to RGB (default: True)

    Returns:
        Dictionary with split statistics
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Map Usage to split names
    split_map = {
        "Training": "train",
        "PublicTest": "val",
        "PrivateTest": "test",
    }

    stats: dict[str, int] = {}

    for usage, split_name in split_map.items():
        split_df = df[df["Usage"] == usage]
        split_dir = output_dir / split_name

        print(f"\nProcessing {split_name} split ({len(split_df)} images)...")

        # Create directories for each emotion
        for emotion in EMOTION_LABELS:
            (split_dir / emotion).mkdir(parents=True, exist_ok=True)

        # Process each image
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df)):
            emotion_idx = row["emotion"]
            emotion_name = EMOTION_LABELS[emotion_idx]
            pixels = parse_pixels(row["pixels"])

            # Convert to PIL Image
            img = Image.fromarray(pixels, mode="L")

            # Resize
            if resize_to != 48:
                img = img.resize((resize_to, resize_to), Image.Resampling.LANCZOS)

            # Convert to RGB if requested
            if create_rgb:
                img = img.convert("RGB")

            # Save
            img_path = split_dir / emotion_name / f"{idx:06d}.jpg"
            img.save(img_path, quality=95)

        stats[split_name] = len(split_df)

    return stats


def create_class_weights(output_dir: Path) -> np.ndarray:
    """
    Calculate class weights for imbalanced dataset handling.

    FER2013 is imbalanced - 'disgust' has far fewer samples than 'happy'.

    Returns:
        Array of class weights inversely proportional to class frequency
    """
    train_dir = output_dir / "train"
    counts = []

    for emotion in EMOTION_LABELS:
        emotion_dir = train_dir / emotion
        count = len(list(emotion_dir.glob("*.jpg")))
        counts.append(count)
        print(f"  {emotion}: {count}")

    counts = np.array(counts)
    total = counts.sum()

    # Inverse frequency weighting
    weights = total / (len(EMOTION_LABELS) * counts)

    # Normalize to sum to num_classes
    weights = weights / weights.sum() * len(EMOTION_LABELS)

    print(f"\nClass weights: {dict(zip(EMOTION_LABELS, weights.round(3)))}")

    # Save weights
    np.save(output_dir / "class_weights.npy", weights)

    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FER2013 dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed/fer2013",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=224,
        help="Target image size (default: 224)",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Keep images as grayscale (default: convert to RGB)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset
    csv_path = download_fer2013(output_dir)

    # Process and save images
    stats = prepare_dataset(
        csv_path=csv_path,
        output_dir=output_dir,
        resize_to=args.resize,
        create_rgb=not args.grayscale,
    )

    print("\n" + "=" * 50)
    print("Dataset preparation complete!")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    for split, count in stats.items():
        print(f"  {split}: {count} images")

    # Calculate and save class weights
    print("\nCalculating class weights...")
    create_class_weights(output_dir)

    print("\nNext steps:")
    print("1. Run training: python training/train_emotion.py --data_dir", output_dir)


if __name__ == "__main__":
    main()
