"""
generator.py — MediVision report generation
Combines ViT visual features + RAG context + Gemini vision
to generate a structured radiology report from a chest X-ray.

Usage:
    python src/generator.py
"""

import os
import base64
from urllib import response
import torch
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
import io

from extractor import extract_single_image, load_vit
from retriever import load_index, retrieve_similar_reports, format_rag_context

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DATASET_CSV = os.path.join("data", "dataset_index.csv")
MODEL_NAME  = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------
def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 string for Gemini vision input."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_prompt(rag_context: str, indication: str = "") -> str:
    """
    Build the prompt sent to Gemini combining RAG context
    and instructions for structured report generation.
    """
    indication_line = f"Clinical Indication: {indication}\n" if indication else ""

    return f"""You are an experienced radiologist analyzing a chest X-ray image.

{indication_line}
Below are similar cases retrieved from a medical database for reference:

{rag_context}

Based on the chest X-ray image provided and the reference cases above,
generate a structured radiology report with the following sections:

FINDINGS:
[Describe what you observe in the X-ray — heart size, lung fields, 
bones, soft tissues, any abnormalities]

IMPRESSION:
[Your overall conclusion — normal or abnormal, key diagnosis]

RECOMMENDATIONS:
[Any follow-up actions, additional imaging, or clinical correlation needed]

Be concise, professional, and use standard radiology terminology.
If the image appears normal, say so clearly.
"""


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    image_path: str,
    indication: str = "",
    model=None,
    image_processor=None,
) -> str:
    """
    Full pipeline: extract features → retrieve similar reports →
    build prompt → call Gemini → return structured report.
    """
    # Step 1 — Extract ViT features from the uploaded image
    print("Extracting visual features...")
    if model is None or image_processor is None:
        model, image_processor = load_vit()
    query_vector = extract_single_image(image_path, model, image_processor)

    # Step 2 — Retrieve similar reports from FAISS
    print("Retrieving similar reports...")
    df = pd.read_csv(DATASET_CSV)
    index, image_ids = load_index()
    similar_reports = retrieve_similar_reports(query_vector, index, image_ids, df)
    rag_context = format_rag_context(similar_reports)

    # Step 3 — Build prompt and call Gemini
    print("Generating report with Gemini...")
    prompt = build_prompt(rag_context, indication)

    # Encode image for Gemini vision
    image_data = encode_image_base64(image_path)
    image_part = {
        "inline_data": {
            "mime_type": "image/png",
            "data": image_data,
        }
    }

    response = client.models.generate_content(
    model=MODEL_NAME,
    contents=[prompt, types.Part.from_bytes(
        data=base64.b64decode(image_data),
        mime_type="image/png"
    )]
)
    return response.text


# ---------------------------------------------------------------------------
# Main — test with a sample image
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(DATASET_CSV)
    sample_row = df.iloc[0]
    image_path = sample_row["image_path"]
    indication = sample_row["indication"]

    print(f"Test image: {sample_row['image_id']}")
    print(f"Indication: {indication}")
    print("-" * 60)

    report = generate_report(image_path, indication)

    print("\nGENERATED REPORT:")
    print("=" * 60)
    print(report)
    print("=" * 60)

    # Compare with ground truth
    print("\nGROUND TRUTH IMPRESSION:")
    print(sample_row["impression"])


if __name__ == "__main__":
    main()