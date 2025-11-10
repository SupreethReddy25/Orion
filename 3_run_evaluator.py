# --- FINAL EVALUATION SCRIPT ("DAY 9") ---
# This script runs the ORIGINAL baseline test against
# your NEWLY trained ORION model.

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from PIL import Image, ImageDraw
import random
from tqdm.notebook import tqdm
import sys
import os
import requests
from io import BytesIO
import json

print("--- 🚀 Starting ORION Evaluation Script ---")

# --- 1. CONFIGURATION ---
NUM_SAMPLES = 1500 # We must use the SAME number as our baseline
MODEL_ID = "openai/clip-vit-base-patch32" # The original model (for the processor)
DATASET_ID = "yerevann/coco-karpathy"
OCCLUSION_PERCENT = 0.5 # The "dumb 50% box"
MODEL_SAVE_PATH = "/content/drive/MyDrive/ML_Projects/Orion/models/orion_clip_final"
METADATA_FILE = "/content/coco_test_metadata.json"
IMAGE_DIR = "/content/coco_test_images"

# --- 2. HELPER FUNCTIONS ---
def add_occlusion(image, occlusion_percent=0.5):
    """The 'dumb 50% box' baseline test."""
    try:
        image = image.convert("RGB")
    except Exception: return None
    width, height = image.size
    occ_area = width * height * occlusion_percent
    occ_width = int(occ_area ** 0.5)
    occ_height = occ_width
    if occ_width >= width or occ_height >= height: return image
    try:
        x1 = random.randint(0, width - occ_width)
        y1 = random.randint(0, height - occ_height)
    except ValueError: x1, y1 = 0, 0
    occluded_image = image.copy()
    draw = ImageDraw.Draw(occluded_image)
    draw.rectangle([(x1, y1), (x1 + occ_width, y1 + occ_height)], fill="black")
    return occluded_image

def get_retrieval_accuracy(image_features, text_features):
    """Calculates the Top-1 retrieval accuracy."""
    image_features_norm = F.normalize(image_features, p=2, dim=-1)
    text_features_norm = F.normalize(text_features, p=2, dim=-1)
    similarity = image_features_norm @ text_features_norm.T
    best_text_indices = similarity.argmax(dim=1)
    correct_indices = torch.arange(len(image_features)).to(image_features.device)
    correct_matches = (best_text_indices == correct_indices).sum().item()
    return correct_matches / len(image_features)

# --- 3. PRE-DOWNLOAD TEST DATA (To avoid network lag) ---
def download_test_data():
    print(f"Loading test dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split='test')
    if NUM_SAMPLES < len(dataset):
         dataset = dataset.select(range(NUM_SAMPLES))
    print(f"Loaded {len(dataset)} test samples. Pre-downloading images...")

    os.makedirs(IMAGE_DIR, exist_ok=True)
    metadata = []
    failed_downloads = 0

    for i in tqdm(range(len(dataset)), desc="Downloading Test Images"):
        item = dataset[i]
        url = item['url']
        caption = item['sentences'][0]
        local_path = os.path.join(IMAGE_DIR, f"test_image_{i:04d}.jpg")
        
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            with Image.open(BytesIO(response.content)) as img:
                img = img.convert("RGB")
                img.save(local_path, "JPEG")
            metadata.append({"local_path": local_path, "caption": caption})
        except Exception:
            failed_downloads += 1
            continue
            
    print(f"Test download complete. Failed: {failed_downloads}")
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)
    
    return metadata

# --- 4. THE EVALUATION SCRIPT ---
def main():
    print("--- Starting Evaluation ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading *YOUR* fine-tuned ORION model from: {MODEL_SAVE_PATH}")
    try:
        model = CLIPModel.from_pretrained(MODEL_SAVE_PATH).to(device)
    except Exception as e:
        print(f"--- ‼️ FATAL ERROR ‼️ ---")
        print(f"Could not load your saved model. Did the training run save correctly?")
        print(f"Make sure this path is correct: {MODEL_SAVE_PATH}")
        print(f"Error: {e}")
        return

    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()

    # --- 5. Prepare Data (from local files) ---
    print("Loading test metadata...")
    if not os.path.exists(METADATA_FILE):
        print("Test metadata not found. Running downloader first...")
        metadata = download_test_data()
    else:
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
            
    print(f"Calculating text features for {len(metadata)} captions...")
    captions = [item['caption'] for item in metadata]
    text_inputs = processor(
        text=captions, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)

    print(f"Calculating image features for {len(metadata)} images...")
    clean_image_features = []
    occluded_image_features = []

    for item in tqdm(metadata, desc="Processing images"):
        try:
            image = Image.open(item['local_path'])
        except Exception:
            continue

        occluded_image = add_occlusion(image, OCCLUSION_PERCENT)
        if occluded_image is None: continue

        inputs = processor(images=[image, occluded_image], return_tensors="pt").to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        clean_image_features.append(features[0])
        occluded_image_features.append(features[1])

    clean_image_features = torch.stack(clean_image_features)
    occluded_image_features = torch.stack(occluded_image_features)

    # --- 6. Calculate & Report Final Results ---
    print("\n--- Calculating Final Accuracy ---")
    clean_accuracy = get_retrieval_accuracy(clean_image_features, text_features)
    occluded_accuracy = get_retrieval_accuracy(occluded_image_features, text_features)

    print("\n" + "="*40)
    print("--- 🏆 FINAL PROJECT RESULTS 🏆 ---")
    print("="*40)
    print(f"Baseline CLIP (Clean):     48.47%")
    print(f"Baseline CLIP (Occluded):  11.73%  <-- (The 76% Drop)")
    print("\n--- ORION MODEL (OURS) ---")
    print(f"ORION Model (Clean):     {clean_accuracy * 100:.2f}%")
    print(f"ORION Model (Occluded):  {occluded_accuracy * 100:.2f}%  <-- (The 'Mind-Blown' Number)")
    print("="*40)

if __name__ == "__main__":
    main()