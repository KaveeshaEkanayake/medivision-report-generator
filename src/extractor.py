"""
extractor.py — MediVision visual feature extraction
Loads a pretrained Vision Transformer (ViT) from HuggingFace and extracts
a 768-dim CLS token feature vector from each chest X-ray image.
Features are saved to data/features.pt for use by the report generator.

Usage:
    python src/extractor.py
"""

import os
import torch
import pandas as pd
from PIL import Image
from transformers import ViTModel, ViTImageProcessor
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_CSV = os.path.join("data", "dataset_index.csv")
FEATURES_OUT = os.path.join("data", "features.pt")

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME = "google/vit-base-patch16-224-in21k"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_vit():
    print(f"Loading ViT model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(DEVICE)

    print("Model loaded successfully")
    return model, image_processor


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features_batch(image_paths: list, model, image_processor) -> torch.Tensor:
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"  [warn] could not load {path}: {e}")
            images.append(Image.new("RGB", (224, 224), color=0))

    inputs = image_processor(images=images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        cls_features = outputs.last_hidden_state[:, 0, :]

    return cls_features.cpu()


def extract_all_features(df: pd.DataFrame, model, image_processor) -> dict:
    image_ids = df["image_id"].tolist()
    image_paths = df["image_path"].tolist()

    features = {}
    total_batches = (len(image_ids) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nExtracting features from {len(image_ids)} images "
          f"in {total_batches} batches of {BATCH_SIZE}...")

    for i in tqdm(range(0, len(image_ids), BATCH_SIZE), total=total_batches):
        batch_ids = image_ids[i : i + BATCH_SIZE]
        batch_paths = image_paths[i : i + BATCH_SIZE]

        batch_features = extract_features_batch(batch_paths, model, image_processor)

        for image_id, feat in zip(batch_ids, batch_features):
            features[image_id] = feat

    return features


# ---------------------------------------------------------------------------
# Single image extraction (used by app.py at inference time)
# ---------------------------------------------------------------------------
def extract_single_image(image_path: str, model=None, image_processor=None) -> torch.Tensor:
    if model is None or image_processor is None:
        model, image_processor = load_vit()

    features = extract_features_batch([image_path], model, image_processor)
    return features[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(DATASET_CSV):
        raise SystemExit(f"{DATASET_CSV} not found. Run preprocess.py first.")

    df = pd.read_csv(DATASET_CSV)
    print(f"Loaded dataset index: {len(df)} image-report pairs")

    model, image_processor = load_vit()
    features = extract_all_features(df, model, image_processor)

    torch.save(features, FEATURES_OUT)
    print(f"\nSaved {len(features)} feature vectors to {FEATURES_OUT}")

    sample_id = list(features.keys())[0]
    sample_feat = features[sample_id]
    print(f"Sample — image_id: {sample_id}, feature shape: {tuple(sample_feat.shape)}")
    print(f"Feature vector norm: {sample_feat.norm().item():.4f}")


if __name__ == "__main__":
    main()