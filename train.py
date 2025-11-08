import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_dataset
from tqdm.notebook import tqdm
import os
import sys

# --- 1. Add our 'src' folder to the Python path ---
# This lets us import our custom files
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import our own code!
try:
    from src.data_utils import OrionDataset
    from src.loss import orion_loss
    print("Successfully imported OrionDataset and orion_loss.")
except ImportError as e:
    print(f"ERROR: Could not import modules: {e}")
    sys.exit(1)

# --- 2. Configuration (All in one place) ---
# --- Training Hyperparameters ---
EPOCHS = 2                # How many times to loop over the data
BATCH_SIZE = 32           # How many samples per step
LEARNING_RATE = 1e-6      # Very low, as we are fine-tuning
LAMBDA_CONST = 0.5        # The weight for our 'consistency_loss' (0.5 means 50%)
WARMUP_STEPS = 100        # How many steps to "warm up" the learning rate

# --- Data & Model Configuration ---
MODEL_ID = "openai/clip-vit-base-patch32"
DATASET_ID = "yerevann/coco-karpathy"
TRAIN_SAMPLES = 25000     # How many samples to train on (25k is a solid number)

# --- Save Configuration ---
MODEL_SAVE_PATH = "./models/orion_clip_final" # Where to save the final model

# --- 3. Helper Functions ---

def collate_fn(batch):
    """
    This custom function tells the DataLoader how to handle
    our `OrionDataset` which returns `None` for bad images.
    It simply filters them out.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # This is the default way to combine a list of dicts
    return torch.utils.data.dataloader.default_collate(batch)

# --- 4. The Main Training Function ---
def main():
    print("--- 🚀 Starting ORION Fine-Tuning Script ---")
    
    # --- Setup Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model & Processor ---
    print(f"Loading base model: {MODEL_ID}")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    # --- Load Datasets ---
    # We need the full training set for the 'contextual_occlusion'
    print(f"Loading full training set metadata: {DATASET_ID}")
    full_train_dataset_hf = load_dataset(DATASET_ID, split='train')
    
    # Select our subset for actual training
    if TRAIN_SAMPLES > len(full_train_dataset_hf):
        train_subset_hf = full_train_dataset_hf
    else:
        train_subset_hf = full_train_dataset_hf.select(range(TRAIN_SAMPLES))
    print(f"Total training samples: {len(train_subset_hf)}")

    # --- Initialize Our Custom Dataset & DataLoader ---
    print("Initializing OrionDataset with advanced augmentations...")
    train_dataset = OrionDataset(
        hf_dataset=train_subset_hf,
        full_train_dataset_for_occluders=full_train_dataset_hf # Pass the *full* set
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn, # Use our custom collate_fn
        num_workers=2,         # Use 2 parallel workers to load data
        pin_memory=True
    )
    
    # --- Setup Optimizer & Scheduler ---
    print("Setting up optimizer...")
    
    # IMPORTANT: We ONLY train the Vision Encoder.
    # We "freeze" the text encoder by not passing its parameters to the optimizer.
    optimizer = AdamW(model.vision_model.parameters(), lr=LEARNING_RATE)
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * EPOCHS
    
    # Setup the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    # --- The Training Loop ---
    print("--- ⭐️ Starting Training ⭐️ ---")
    
    # Set model parts to the correct mode
    model.vision_model.train() # We are training this part
    model.text_model.eval()    # We are *not* training this part
    
    # Get the learnable 'logit_scale' (temperature) from the model
    # We don't train this, but our loss function needs it
    logit_scale = model.logit_scale.exp()
    
    global_step = 0
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1} / {EPOCHS} ---")
        
        # Use tqdm for a professional progress bar
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        # Initialize trackers for our loss
        total_loss_epoch = 0.0
        total_loss_clip_epoch = 0.0
        total_loss_const_epoch = 0.0
        
        for batch in progress_bar:
            # Skip empty batches (if all images failed)
            if batch is None:
                continue
            
            # --- 1. Move data to GPU ---
            original_images = batch['original_image']
            occluded_images = batch['occluded_image']
            captions = batch['caption']
            
            # --- 2. Process Data ---
            # Process text (with no gradients, as it's "frozen")
            with torch.no_grad():
                text_inputs = processor(
                    text=captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)
            
            # Process images
            original_image_inputs = processor(
                images=original_images,
                return_tensors="pt"
            ).to(device)
            
            occluded_image_inputs = processor(
                images=occluded_images,
                return_tensors="pt"
            ).to(device)

            # --- 3. Forward Pass ---
            # Get embeddings for all three inputs
            with torch.no_grad():
                # Original image embeddings (no grad, this is just for comparison)
                original_img_emb = model.get_image_features(**original_image_inputs)
                # Text embeddings (no grad, text model is frozen)
                text_emb = model.get_text_features(**text_inputs)
            
            # Occluded image embeddings (THIS is the one we train on)
            # We must *not* have 'no_grad' here
            occluded_img_emb = model.get_image_features(**occluded_image_inputs)

            # --- 4. Calculate Our Custom Loss ---
            loss, loss_c, loss_k = orion_loss(
                original_img_emb,
                occluded_img_emb,
                text_emb,
                logit_scale,
                LAMBDA_CONST
            )
            
            # --- 5. Backward Pass & Optimization ---
            # Clear old gradients
            optimizer.zero_grad()
            # Calculate new gradients
            loss.backward()
            # Update the model's weights
            optimizer.step()
            # Update the learning rate
            scheduler.step()
            
            global_step += 1
            
            # --- 6. Log Our Progress ---
            total_loss_epoch += loss.item()
            total_loss_clip_epoch += loss_c.item()
            total_loss_const_epoch += loss_k.item()
            
            # Update the progress bar description
            progress_bar.set_postfix({
                "Total Loss": f"{loss.item():.4f}",
                "CLIP Loss": f"{loss_c.item():.4f}",
                "Const Loss": f"{loss_k.item():.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.1e}"
            })

        # --- End of Epoch ---
        avg_loss = total_loss_epoch / len(train_dataloader)
        avg_loss_clip = total_loss_clip_epoch / len(train_dataloader)
        avg_loss_const = total_loss_const_epoch / len(train_dataloader)
        print(f"End of Epoch {epoch + 1}")
        print(f"Average Total Loss: {avg_loss:.4f}")
        print(f"  > Avg CLIP Loss: {avg_loss_clip:.4f}")
        print(f"  > Avg Consistency Loss: {avg_loss_const:.4f}")

    # --- 5. Save The Final Model ---
    print("\n--- ✅ Training Complete ---")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    
    # Save the *entire* model (Vision + Text)
    # This makes it easy to load and use later
    model.save_pretrained(MODEL_SAVE_PATH)
    processor.save_pretrained(MODEL_SAVE_PATH)
    
    print("--- 🎉 Model Saved Successfully! ---")

# --- This makes the script runnable from the command line ---
if __name__ == "__main__":
    main()