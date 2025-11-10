# --- STAGE 1: THE PRE-DOWNLOADER ---
# This script downloads all 25,000 images to Colab's fast local disk.

import os
import requests
from io import BytesIO
from PIL import Image
from datasets import load_dataset
from tqdm.notebook import tqdm
import json
import sys

# --- Configuration ---
TRAIN_SAMPLES = 25000
DATASET_ID = "yerevann/coco-karpathy"
IMAGE_DIR = "/content/coco_train_images" # Use the fast local disk
METADATA_FILE = "/content/train_metadata.json"

print(f"--- Stage 1: Pre-downloading {TRAIN_SAMPLES} images ---")
print(f"This will take 1-2 hours. Please be patient. The progress bar WILL move.")

# --- Load dataset metadata ---
print("Loading dataset metadata...")
try:
    full_train_dataset = load_dataset(DATASET_ID, split='train')
    if TRAIN_SAMPLES > len(full_train_dataset):
        train_subset = full_train_dataset
    else:
        train_subset = full_train_dataset.select(range(TRAIN_SAMPLES))
    print(f"Loaded metadata for {len(train_subset)} train samples.")
except Exception as e:
    print(f"Failed to load dataset. Error: {e}")
    sys.exit(1) # Can't continue if this fails

# --- Create directory ---
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Download loop ---
metadata = [] # We will save captions and filepaths here
failed_downloads = 0

for i in tqdm(range(len(train_subset)), desc="Downloading Images"):
    item = train_subset[i]
    url = item['url']
    caption = item['sentences'][0]
    
    local_path = os.path.join(IMAGE_DIR, f"image_{i:05d}.jpg") # e.g., image_00001.jpg
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        with Image.open(BytesIO(response.content)) as img:
            img = img.convert("RGB")
            img.save(local_path, "JPEG")
            
        metadata.append({
            "local_path": local_path,
            "caption": caption
        })
    except Exception as e:
        failed_downloads += 1
        continue

# --- Save metadata ---
with open(METADATA_FILE, 'w') as f:
    json.dump(metadata, f)

print("\n--- ✅ STAGE 1 (DOWNLOAD) COMPLETE! ---")
print(f"Successfully downloaded: {len(metadata)} images")
print(f"Failed to download:    {failed_downloads} images")
print(f"Image directory:       {IMAGE_DIR}")
print(f"Metadata file:         {METADATA_FILE}")
print("You are now ready for Stage 2: Training.")