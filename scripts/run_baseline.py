import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from PIL import Image, ImageDraw
import random
from tqdm import tqdm
import sys
import os

# --- 1. Configuration ---

NUM_SAMPLES = 5000
OCCLUSION_PERCENT = 0.5 # 50% occlusion
MODEL_ID = "openai/clip-vit-base-patch32"
DATASET_ID = "merve/coco_karpathy_test"
RESULTS_FILE = "baseline_results.txt" # We will save our results to a file

# --- 2. Setup Helper Function ---

def add_occlusion(image, occlusion_percent=0.5):
    """Adds a random black square occlusion to a PIL image."""
    try:
        image = image.convert("RGB")
    except Exception as e:
        print(f"Warning: Could not convert image. Skipping. Error: {e}", file=sys.stderr)
        return None
        
    width, height = image.size
    
    # Calculate occlusion dimensions
    occ_area = width * height * occlusion_percent
    occ_width = int(occ_area ** 0.5)
    occ_height = occ_width
    
    # Ensure occlusion is not larger than the image
    if occ_width >= width or occ_height >= height:
        print(f"Warning: Occlusion size ({occ_width}x{occ_height}) larger than image ({width}x{height}). Skipping.", file=sys.stderr)
        return image # Return original image if occlusion is too big

    # Get a random top-left corner
    try:
        x1 = random.randint(0, width - occ_width)
        y1 = random.randint(0, height - occ_height)
    except ValueError:
        # This can happen if width == occ_width
        x1, y1 = 0, 0
    
    # Create a copy and draw the black rectangle
    occluded_image = image.copy()
    draw = ImageDraw.Draw(occluded_image)
    draw.rectangle([(x1, y1), (x1 + occ_width, y1 + occ_height)], fill="black")
    
    return occluded_image

def get_retrieval_accuracy(image_features, text_features):
    """
    Calculates the Top-1 retrieval accuracy for image-to-text retrieval.
    Assumes image_features[i] corresponds to text_features[i].
    """
    # Normalize features for cosine similarity
    image_features_norm = F.normalize(image_features, p=2, dim=-1)
    text_features_norm = F.normalize(text_features, p=2, dim=-1)
    
    # Calculate cosine similarity
    similarity = image_features_norm @ text_features_norm.T
    
    # Find the best matching text for each image
    # similarity is [Num_Images, Num_Texts]
    best_text_indices = similarity.argmax(dim=1)
    
    # The correct text index for image `i` is also `i`
    correct_indices = torch.arange(len(image_features)).to(image_features.device)
    
    # Count how many images found their correct caption
    correct_matches = (best_text_indices == correct_indices).sum().item()
    
    return correct_matches / len(image_features)

# --- 3. Main Script Execution ---

def run_baseline():
    print(f"--- Starting Baseline Test ---")
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Occlusion: {OCCLUSION_PERCENT * 100}%\n")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: No GPU detected. This will be very slow.")

    # Load model and processor
    print("Loading CLIP model and processor...")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval() # Set model to evaluation mode (no gradients)

    # Load dataset
    print("Loading dataset...")
    try:
        # Use the Karpathy Test split
        dataset = load_dataset(DATASET_ID, split='test')
        if NUM_SAMPLES < len(dataset):
             dataset = dataset.select(range(NUM_SAMPLES))
        print(f"Loaded {len(dataset)} samples.")
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return

    # --- 4. Prepare Data ---
    
    # Pre-calculate text features
    print(f"\nCalculating text features for {len(dataset)} captions...")
    # Get the *first* caption for each image
    captions = [sample['caption'][0] for sample in dataset] 
    
    text_inputs = processor(
        text=captions,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    
    # Pre-calculate image features (clean and occluded)
    print(f"Calculating image features for {len(dataset)} images...")
    clean_image_features = []
    occluded_image_features = []
    
    for item in tqdm(dataset, desc="Processing images"):
        image = item['image']
        occluded_image = add_occlusion(image, OCCLUSION_PERCENT)
        
        if occluded_image is None:
            continue # Skip if image was problematic

        # Process both images in one batch
        inputs = processor(
            images=[image, occluded_image],
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            
        clean_image_features.append(features[0])
        occluded_image_features.append(features[1])

    # Stack into final tensors
    clean_image_features = torch.stack(clean_image_features)
    occluded_image_features = torch.stack(occluded_image_features)

    # --- 5. Calculate & Report Results ---
    print("\n--- Calculating Accuracy ---")
    
    # Calculate accuracy for clean images
    clean_accuracy = get_retrieval_accuracy(clean_image_features, text_features)
    
    # Calculate accuracy for occluded images
    occluded_accuracy = get_retrieval_accuracy(occluded_image_features, text_features)
    
    # Format results
    clean_acc_str = f"{clean_accuracy * 100:.2f}%"
    occ_acc_str = f"{occluded_accuracy * 100:.2f}%"
    
    # Print to console
    print("\n" + "="*30)
    print("--- BASELINE RESULTS (Top-1 Retrieval) ---")
    print(f"Clean Image Accuracy:   {clean_acc_str}")
    print(f"Occluded Image Accuracy: {occ_acc_str}")
    print("="*30)
    
    # Save results to a file
    with open(RESULTS_FILE, "w") as f:
        f.write("--- BASELINE RESULTS ---\n")
        f.write(f"Clean Image Accuracy:   {clean_acc_str}\n")
        f.write(f"Occluded Image Accuracy: {occ_acc_str}\n")
    print(f"Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_baseline()