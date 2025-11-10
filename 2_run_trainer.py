# --- STAGE 2: THE FAST TRAINER ---
# This script reads from the local files and will not freeze.

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import CLIPProcessor, CLIPModel, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm.notebook import tqdm
import os
import sys
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import random
import math
import json

# ---
# --- PART 1: LOCAL HELPER FUNCTIONS ---
# ---
print("Loading helper functions (Local Version)...")

def add_random_box(image):
    img_copy = image.convert("RGB").copy()
    width, height = img_copy.size
    total_area = width * height
    box_area = random.uniform(0.1, 0.5) * total_area
    aspect_ratio = random.uniform(0.5, 2.0)
    box_height = int((box_area / aspect_ratio) ** 0.5)
    box_width = int(box_area / box_height)
    box_width = min(box_width, width)
    box_height = min(box_height, height)
    try:
        x = random.randint(0, width - box_width)
        y = random.randint(0, height - box_height)
    except ValueError:
        x, y = 0, 0
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle([x, y, x + box_width, y + box_height], fill="black")
    return img_copy

def add_cutout(image):
    img_copy = image.convert("RGB").copy()
    width, height = img_copy.size
    total_area = width * height
    draw = ImageDraw.Draw(img_copy)
    for _ in range(10):
        patch_area = random.uniform(0.01, 0.1) * total_area
        aspect_ratio = random.uniform(0.5, 2.0)
        patch_height = int((patch_area / aspect_ratio) ** 0.5)
        patch_width = int(patch_area / patch_height)
        patch_width = min(patch_width, width)
        patch_height = min(patch_height, height)
        try:
            x = random.randint(0, width - patch_width)
            y = random.randint(0, height - patch_height)
        except ValueError:
            x, y = 0, 0
        draw.rectangle([x, y, x + patch_width, y + patch_height], fill="black")
    return img_copy

def add_contextual_occlusion_LOCAL(image, all_local_image_paths):
    img_copy = image.convert("RGB").copy()
    try:
        width, height = img_copy.size
        total_area = width * height
        random_path = random.choice(all_local_image_paths)
        random_image = Image.open(random_path).convert("RGB")
        patch_area = random.uniform(0.2, 0.4) * total_area
        aspect_ratio = random.uniform(0.5, 2.0)
        patch_height = int((patch_area / aspect_ratio) ** 0.5)
        patch_width = int(patch_area / patch_height)
        patch_width = min(patch_width, width, random_image.width)
        patch_height = min(patch_height, height, random_image.height)
        src_x = random.randint(0, max(0, random_image.width - patch_width))
        src_y = random.randint(0, max(0, random_image.height - patch_height))
        patch = random_image.crop((src_x, src_y, src_x + patch_width, src_y + patch_height))
        dst_x = random.randint(0, max(0, width - patch_width))
        dst_y = random.randint(0, max(0, height - patch_height))
        img_copy.paste(patch, (dst_x, dst_y))
        return img_copy
    except Exception:
        return add_cutout(img_copy)

# ---
# --- PART 2: THE NEW, 100% LOCAL DATASET ---
# ---
class OrionDataset_Local(Dataset):
    def __init__(self, metadata, all_local_image_paths):
        self.metadata = metadata
        self.all_local_paths = all_local_image_paths
        self.augmentation_functions = [
            add_random_box,
            add_cutout,
            lambda img: add_contextual_occlusion_LOCAL(img, self.all_local_paths)
        ]
        print(f"OrionDataset_Local initialized with {len(self.metadata)} samples.")
        print("Using Advanced Occlusion Pipeline (RandomBox, Cutout, Contextual).")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        try:
            item = self.metadata[idx]
            image_path = item['local_path']
            caption = item['caption']
            image = Image.open(image_path).convert("RGB")
            augmentation_fn = random.choice(self.augmentation_functions)
            augmented_image = augmentation_fn(image)
            return {
                'original_image': image,
                'occluded_image': augmented_image,
                'caption': caption
            }
        except Exception as e:
            return None

# ---
# --- PART 3: THE LOSS FUNCTION (UNCHANGED) ---
# ---
print("Loading custom loss function...")
def orion_loss(original_img_emb, occluded_img_emb, text_emb, logit_scale, lambda_const=0.5):
    occluded_img_emb_norm = F.normalize(occluded_img_emb, p=2, dim=-1)
    text_emb_norm = F.normalize(text_emb, p=2, dim=-1)
    logits_per_image = logit_scale * occluded_img_emb_norm @ text_emb_norm.t()
    logits_per_text = logits_per_image.t()
    labels = torch.arange(len(logits_per_image), dtype=torch.long).to(logits_per_image.device)
    clip_loss_i = F.cross_entropy(logits_per_image, labels)
    clip_loss_t = F.cross_entropy(logits_per_text, labels)
    loss_clip = (clip_loss_i + clip_loss_t) / 2
    target = torch.ones(len(original_img_emb), dtype=torch.long).to(original_img_emb.device)
    loss_const = F.cosine_embedding_loss(original_img_emb, occluded_img_emb, target)
    total_loss = loss_clip + (lambda_const * loss_const)
    return total_loss, loss_clip, loss_const

# ---
# --- PART 4: THE MAIN TRAINING SCRIPT ---
# ---
print("Loading training script components...")

# --- Configuration (All in one place) ---
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 1e-6
LAMBDA_CONST = 0.5
WARMUP_STEPS = 100
MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_SAVE_PATH = "/content/drive/MyDrive/ML_Projects/Orion/models/orion_clip_final"
METADATA_FILE = "/content/train_metadata.json"
IMAGE_DIR = "/content/coco_train_images"

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return {} 
    return {
        'original_image': [item['original_image'] for item in batch],
        'occluded_image': [item['occluded_image'] for item in batch],
        'caption': [item['caption'] for item in batch]
    }

def main():
    print("--- 🚀 Starting ORION Fine-Tuning Script (Local Data) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading base model: {MODEL_ID}")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    print(f"Loading pre-downloaded metadata from {METADATA_FILE}...")
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    all_local_image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    print(f"Found {len(metadata)} training samples and {len(all_local_image_paths)} images for context.")

    print("Initializing OrionDataset_Local...")
    train_dataset = OrionDataset_Local(
        metadata=metadata,
        all_local_image_paths=all_local_image_paths
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0, # "no-freeze" fix
        pin_memory=True
    )
    
    print("Setting up optimizer...")
    optimizer = AdamW(model.vision_model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    print("--- ⭐️ Starting Training ⭐️ ---")
    
    model.vision_model.train()
    model.text_model.eval()
    logit_scale = model.logit_scale.exp().detach()
    
    global_step = 0
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1} / {EPOCHS} ---")
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        total_loss_epoch = 0.0
        total_loss_clip_epoch = 0.0
        total_loss_const_epoch = 0.0
        
        for batch in progress_bar:
            if not batch: continue
            
            original_images = batch['original_image']
            occluded_images = batch['occluded_image']
            captions = batch['caption']
            
            with torch.no_grad():
                text_inputs = processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)
            original_image_inputs = processor(images=original_images, return_tensors="pt").to(device)
            occluded_image_inputs = processor(images=occluded_images, return_tensors="pt").to(device)

            with torch.no_grad():
                original_img_emb = model.get_image_features(**original_image_inputs)
                text_emb = model.get_text_features(**text_inputs)
            
            occluded_img_emb = model.get_image_features(**occluded_image_inputs)

            loss, loss_c, loss_k = orion_loss(
                original_img_emb,
                occluded_img_emb,
                text_emb,
                logit_scale,
                LAMBDA_CONST
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            total_loss_epoch += loss.item()
            total_loss_clip_epoch += loss_c.item()
            total_loss_const_epoch += loss_k.item()
            
            progress_bar.set_postfix({
                "Total Loss": f"{loss.item():.4f}",
                "CLIP Loss": f"{loss_c.item():.4f}",
                "Const Loss": f"{loss_k.item():.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.1e}"
            })

        avg_loss = total_loss_epoch / len(train_dataloader)
        avg_loss_clip = total_loss_clip_epoch / len(train_dataloader)
        avg_loss_const = total_loss_const_epoch / len(train_dataloader)
        print(f"End of Epoch {epoch + 1}")
        print(f"Average Total Loss: {avg_loss:.4f}")
        print(f"  > Avg CLIP Loss: {avg_loss_clip:.4f}")
        print(f"  > Avg Consistency Loss: {avg_loss_const:.4f}")

    print("\n--- ✅ Training Complete ---")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    
    model.save_pretrained(MODEL_SAVE_PATH)
    processor.save_pretrained(MODEL_SAVE_PATH)
    
    print("--- 🎉 Model Saved Successfully! ---")

if __name__ == "__main__":
    main()