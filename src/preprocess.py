"""
preprocess.py — MediVision data preprocessing
Parses IU X-Ray XML reports, links them to PNG images, filters incomplete
records, and builds a clean dataset index. Also provides ViT-ready image
preprocessing transforms.

Usage:
    python src/preprocess.py
"""

import os
import glob
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths (relative to repo root — run from the repo root directory)
# ---------------------------------------------------------------------------
IMAGES_DIR = os.path.join("data", "images")
REPORTS_DIR = os.path.join("data", "reports")
OUTPUT_CSV = os.path.join("data", "dataset_index.csv")

# ViT expects 224x224 RGB, normalized with ImageNet stats
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# XML report parsing
# ---------------------------------------------------------------------------
def parse_report(xml_path: str) -> dict | None:
    """
    Parse a single IU X-Ray XML report.

    Returns a dict with report id, text sections, and linked image ids,
    or None if the file can't be parsed.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"  [skip] unparseable XML: {xml_path}")
        return None

    sections = {"COMPARISON": "", "INDICATION": "", "FINDINGS": "", "IMPRESSION": ""}
    for abstract_text in root.iter("AbstractText"):
        label = abstract_text.get("Label", "").upper()
        if label in sections:
            sections[label] = (abstract_text.text or "").strip()

    # parentImage tags link this report to one or more PNG files
    image_ids = [img.get("id") for img in root.iter("parentImage") if img.get("id")]

    report_id = os.path.splitext(os.path.basename(xml_path))[0]

    return {
        "report_id": report_id,
        "comparison": sections["COMPARISON"],
        "indication": sections["INDICATION"],
        "findings": sections["FINDINGS"],
        "impression": sections["IMPRESSION"],
        "image_ids": image_ids,
    }


def clean_text(text: str) -> str:
    """Normalize whitespace and strip de-identification placeholders."""
    if not text:
        return ""
    # IU X-Ray uses XXXX as a de-identification placeholder
    text = text.replace("XXXX", "").replace("xxxx", "")
    # Collapse repeated whitespace
    text = " ".join(text.split())
    return text.strip()


# ---------------------------------------------------------------------------
# Dataset index construction
# ---------------------------------------------------------------------------
def build_dataset_index() -> pd.DataFrame:
    """
    Walk all XML reports, link each to its PNG images, filter incomplete
    records, and return one row per (image, report) pair.
    """
    xml_files = glob.glob(os.path.join(REPORTS_DIR, "**", "*.xml"), recursive=True)
    print(f"Found {len(xml_files)} XML report files")

    rows = []
    skipped_no_text = 0
    skipped_no_image = 0

    for xml_path in xml_files:
        report = parse_report(xml_path)
        if report is None:
            continue

        findings = clean_text(report["findings"])
        impression = clean_text(report["impression"])

        # Need at least one of findings/impression to be useful for training
        if not findings and not impression:
            skipped_no_text += 1
            continue

        if not report["image_ids"]:
            skipped_no_image += 1
            continue

        for image_id in report["image_ids"]:
            image_path = os.path.join(IMAGES_DIR, f"{image_id}.png")
            if not os.path.exists(image_path):
                continue
            rows.append(
                {
                    "report_id": report["report_id"],
                    "image_id": image_id,
                    "image_path": image_path,
                    "indication": clean_text(report["indication"]),
                    "findings": findings,
                    "impression": impression,
                }
            )

    df = pd.DataFrame(rows)
    print(f"Skipped {skipped_no_text} reports with no findings/impression")
    print(f"Skipped {skipped_no_image} reports with no linked images")
    print(f"Final dataset: {len(df)} image-report pairs "
          f"from {df['report_id'].nunique() if len(df) else 0} unique reports")
    return df


# ---------------------------------------------------------------------------
# Image preprocessing (used later by extractor.py)
# ---------------------------------------------------------------------------
def get_transform() -> transforms.Compose:
    """Standard ViT preprocessing pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image(image_path: str) -> torch.Tensor:
    """
    Load a chest X-ray PNG and return a preprocessed tensor of shape
    (3, 224, 224) ready for ViT.
    """
    image = Image.open(image_path).convert("RGB")  # X-rays are grayscale; ViT wants 3 channels
    return get_transform()(image)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.isdir(IMAGES_DIR) or not os.path.isdir(REPORTS_DIR):
        raise SystemExit(
            "data/images or data/reports not found. "
            "Run this script from the repo root after extracting the dataset."
        )

    df = build_dataset_index()
    if df.empty:
        raise SystemExit("No valid image-report pairs found — check extraction paths.")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved dataset index to {OUTPUT_CSV}")

    # Sanity check: load one image through the ViT pipeline
    sample_tensor = load_image(df.iloc[0]["image_path"])
    print(f"Sample image tensor shape: {tuple(sample_tensor.shape)}")  # (3, 224, 224)
    print("\nSample record:")
    print(df.iloc[0][["image_id", "indication", "impression"]].to_string())


if __name__ == "__main__":
    main()