import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

# ==========================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==========================================
TRAIN_IMG_DIR = "train-new/Fluo-N2DL-HeLa/01"
TRAIN_MASK_DIR = "train-new/Fluo-N2DL-HeLa/01_ST/SEG"
VAL_IMG_DIR = "train-new/Fluo-N2DL-HeLa/02"
VAL_MASK_DIR = "train-new/Fluo-N2DL-HeLa/02_ST/SEG"

BATCH_SIZE = 8
NUM_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DATASET CLASS
# ==========================================
class CellDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = [img for img in os.listdir(images_dir) if img.endswith('.tif')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = "man_seg" + img_name[1:]
        mask_path = os.path.join(self.masks_dir, mask_name) 
        
        # Read 16-bit images
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # Resize to 512x512 for VRAM optimization
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # Preprocessing (Normalization)
        image = image.astype(np.float32) / image.max() 
        mask = (mask > 0).astype(np.float32)
        
        # Apply Albumentations transform IF provided
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        else:
            # Convert to PyTorch format manually
            image = torch.from_numpy(np.expand_dims(image, axis=0))
            mask = torch.from_numpy(np.expand_dims(mask, axis=0))
            
        return image, mask

# ==========================================
# 3. CUSTOM LOSS FUNCTION (BCE + Dice)
# ==========================================
class ComboLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(ComboLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss() 
        self.dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.w_bce = weight_bce
        self.w_dice = weight_dice

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return (self.w_bce * bce_loss) + (self.w_dice * dice_loss)


# 4. MAIN TRAINING SCRIPT

if __name__ == "__main__":
    print(f"Training on device: {DEVICE}")

    # Initialize Datasets & DataLoaders
    train_dataset = CellDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=None)
    val_dataset = CellDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize Model, Loss, Optimizer & Scheduler
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        decoder_attention_type="scse"
    ).to(DEVICE)

    weights_path = "best_unet_model.pt"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print("Loaded previous weights for fine-tuning!")

    criterion = ComboLoss()

    # Differential Learning Rates
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5},
        {'params': model.decoder.parameters(), 'lr': 1e-3},
        {'params': model.segmentation_head.parameters(), 'lr': 1e-3},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )

    # Mixed Precision Scaler (Updated to modern API)
    scaler = torch.amp.GradScaler('cuda')
    best_val_loss = float('inf')

    # The Training Loop
    for epoch in range(NUM_EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_unet_model.pt")
            print("--> Best model saved!")
