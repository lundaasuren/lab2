import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
from glob import glob
import warnings
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
warnings.filterwarnings('ignore')

# ===========================================
#  Setup & Reproducibility
# ===========================================
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===========================================
#  FIXED: Consistent Label Mapping
# ===========================================
class LabelMapper:
    """Fixed label mapping for OASIS dataset"""
    def __init__(self):
        # Standard OASIS label mapping
        self.mapping = {
            0: 0,    # Background
            85: 1,   # CSF
            170: 2,  # Gray Matter  
            255: 3   # White Matter
        }
        self.class_names = ['Background', 'CSF', 'Gray Matter', 'White Matter']
        self.num_classes = len(self.mapping)
    
    def map_labels(self, mask):
        """Apply consistent label mapping"""
        mapped_mask = np.zeros_like(mask, dtype=np.uint8)
        for original, new in self.mapping.items():
            mapped_mask[mask == original] = new
        
        # Handle any unmapped values
        unique_vals = np.unique(mask)
        unmapped = [val for val in unique_vals if val not in self.mapping]
        if unmapped:
            print(f"Warning: Found unmapped values {unmapped}, assigning to background")
            for val in unmapped:
                mapped_mask[mask == val] = 0  # Assign to background
        
        return mapped_mask

# ===========================================
#  IMPROVED: Dataset with Augmentation
# ===========================================
class OASISDataset(Dataset):
    def __init__(self, data_dir, folder_name, img_size=(256, 256), is_training=True, augment_prob=0.8):
        self.data_dir = data_dir
        self.img_size = img_size
        self.is_training = is_training
        self.label_mapper = LabelMapper()
        
        img_folder = os.path.join(data_dir, folder_name)
        seg_folder = os.path.join(data_dir, folder_name.replace('keras_png_slices_', 'keras_png_slices_seg_'))
        
        if not os.path.isdir(seg_folder):
            raise FileNotFoundError(f"Segmentation folder not found: {seg_folder}")
        
        self.img_paths = sorted(glob(os.path.join(img_folder, '*.png')))
        self.seg_paths = sorted(glob(os.path.join(seg_folder, '*.png')))
        
        print(f"Found {len(self.img_paths)} images and {len(self.seg_paths)} masks in {folder_name}")
        assert len(self.img_paths) == len(self.seg_paths), "Number of images and masks must match"
        
        # FIXED: Augmentation pipeline
        if is_training:
            self.transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, 
                                 border_mode=cv2.BORDER_CONSTANT, value=0, p=0.6),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, 
                                 border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.GaussNoise(var_limit=(0.0, 0.005), p=0.4),
                A.Normalize(mean=0.0, std=1.0),
            ], additional_targets={'mask': 'mask'}, p=augment_prob)
        else:
            self.transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.Normalize(mean=0.0, std=1.0),
            ], additional_targets={'mask': 'mask'})
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        
        # Load mask
        mask = cv2.imread(self.seg_paths[idx], cv2.IMREAD_UNCHANGED)
        
        # FIXED: Apply consistent label mapping BEFORE resizing
        mask = self.label_mapper.map_labels(mask)
        
        # Apply transforms
        transformed = self.transform(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']
        
        # Convert to tensors
        img = torch.from_numpy(img).unsqueeze(0).float()  # Add channel dim
        mask = torch.from_numpy(mask).long()
        
        return img, mask

# ===========================================
#  Model Components (Improved)
# ===========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),  # Handle small channels
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4, base=64):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, base, 0.1)
        self.down1 = DownBlock(base, base*2, 0.1)
        self.down2 = DownBlock(base*2, base*4, 0.2)
        self.down3 = DownBlock(base*4, base*8, 0.2)
        self.down4 = DownBlock(base*8, base*16, 0.3)
        self.up1 = UpBlock(base*16, base*8, 0.2)
        self.up2 = UpBlock(base*8, base*4, 0.2)
        self.up3 = UpBlock(base*4, base*2, 0.1)
        self.up4 = UpBlock(base*2, base, 0.1)
        self.outc = nn.Conv2d(base, n_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# ===========================================
#  IMPROVED: Loss Functions
# ===========================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, predictions, targets, num_classes=4):
        predictions = F.softmax(predictions, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        dice_loss = 0.0
        for c in range(num_classes):
            p = predictions[:, c]
            t = targets_one_hot[:, c]
            intersection = torch.sum(p * t, dim=(1, 2))
            union = torch.sum(p, dim=(1, 2)) + torch.sum(t, dim=(1, 2))
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice.mean())
        return dice_loss / num_classes

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, use_focal=True, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        if use_focal:
            self.ce_loss = FocalLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()
    def forward(self, predictions, targets):
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.alpha * ce + (1 - self.alpha) * dice

# ===========================================
#  FIXED: Class Weight Calculation
# ===========================================
def calculate_class_weights(dataset, sample_size=1000, method='balanced'):
    """Sample-based class weight calculation to avoid OOM"""
    print(f"Calculating class weights from {sample_size} samples...")
    
    # Sample subset to avoid loading entire dataset
    indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
    all_labels = []
    
    for idx in tqdm(indices, desc="Sampling for class weights"):
        _, mask = dataset[idx]
        all_labels.extend(mask.numpy().flatten())
    
    all_labels = np.array(all_labels)
    class_counts = np.bincount(all_labels, minlength=4)
    
    if method == 'balanced':
        # Standard sklearn-style balanced weighting
        n_samples = len(all_labels)
        n_classes = len(class_counts)
        weights = n_samples / (n_classes * (class_counts + 1e-6))
    elif method == 'inverse_freq':
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * len(class_counts)
    else:
        weights = np.ones(len(class_counts))
    
    weights_tensor = torch.FloatTensor(weights).to(device)
    
    # Print distribution
    mapper = LabelMapper()
    print("Class distribution and weights:")
    for i, (count, weight) in enumerate(zip(class_counts, weights)):
        print(f"{mapper.class_names[i]:<15}: {count:>8} samples ({count/len(all_labels)*100:>5.1f}%), weight: {weight:.4f}")
    
    return weights_tensor

# ===========================================
#  ADDED: Dice Coefficient Calculation
# ===========================================
def calculate_dice_coefficient(pred, target, num_classes=4, smooth=1e-6):
    """Calculate per-class Dice coefficient"""
    dice_scores = []
    pred = torch.argmax(pred, dim=1)
    
    for class_idx in range(num_classes):
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        
        intersection = torch.sum(pred_class * target_class)
        union = torch.sum(pred_class) + torch.sum(target_class)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
    
    return dice_scores

# ===========================================
#  IMPROVED: Training with Better Metrics
# ===========================================
from torch.cuda.amp import autocast, GradScaler

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, 
                class_weights=None, use_focal=True):
    criterion = CombinedLoss(alpha=0.5, use_focal=use_focal, class_weights=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    scaler = GradScaler()
    
    best_val_dice = 0.0
    patience, patience_counter = 25, 0
    mapper = LabelMapper()
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_dice_total = 0.0, [0.0] * 4
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Calculate dice scores
            with torch.no_grad():
                dice_scores = calculate_dice_coefficient(outputs, masks)
                for i, score in enumerate(dice_scores):
                    train_dice_total[i] += score
            
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_loss, val_dice_total = 0.0, [0.0] * 4
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, masks = images.to(device), masks.to(device)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                
                dice_scores = calculate_dice_coefficient(outputs, masks)
                for i, score in enumerate(dice_scores):
                    val_dice_total[i] += score
        
        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_dice_avg = [score / len(train_loader) for score in train_dice_total]
        val_dice_avg = [score / len(val_loader) for score in val_dice_total]
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(np.mean(train_dice_avg))
        history['val_dice'].append(np.mean(val_dice_avg))
        
        scheduler.step(val_loss)
        current_val_dice = np.mean(val_dice_avg)
        
        # Print detailed results
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Dice: {np.mean(train_dice_avg):.4f}, Val Dice: {current_val_dice:.4f}")
        print("Per-class Dice scores:")
        for i, (train_d, val_d) in enumerate(zip(train_dice_avg, val_dice_avg)):
            print(f"  {mapper.class_names[i]:<15}: Train {train_d:.4f}, Val {val_d:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Model saving based on dice score
        if current_val_dice > best_val_dice:
            best_val_dice = current_val_dice
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_dice': best_val_dice,
                'history': history
            }, 'best_unet_checkpoint.pth')
            print(f"✓ New best model saved! Val Dice: {best_val_dice:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
        
        print("-" * 60)
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    # Load best model
    checkpoint = torch.load('best_unet_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state'])
    print(f"Best model restored with Val Dice: {checkpoint['best_dice']:.4f}")
    return model, checkpoint['history']

# ===========================================
#  ADDED: Comprehensive Evaluation
# ===========================================
def evaluate_model(model, test_loader):
    """Comprehensive evaluation with per-class metrics"""
    model.eval()
    mapper = LabelMapper()
    all_dice_scores = [[] for _ in range(4)]
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            dice_scores = calculate_dice_coefficient(outputs, masks, num_classes=4)
            for i, score in enumerate(dice_scores):
                all_dice_scores[i].append(score)
    
    # Calculate statistics
    mean_dice_scores = [np.mean(scores) for scores in all_dice_scores]
    std_dice_scores = [np.std(scores) for scores in all_dice_scores]
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    for i, (mean_score, std_score) in enumerate(zip(mean_dice_scores, std_dice_scores)):
        status = "✓" if mean_score > 0.9 else "✗"
        print(f"{status} {mapper.class_names[i]:<15}: {mean_score:.4f} ± {std_score:.4f}")
    
    overall_mean = np.mean(mean_dice_scores)
    target_achieved = all(score > 0.9 for score in mean_dice_scores)
    
    print(f"\nOverall Mean Dice: {overall_mean:.4f}")
    print(f"Target DSC > 0.9 achieved: {'✓ YES' if target_achieved else '✗ NO'}")
    print("="*60)
    
    return mean_dice_scores, overall_mean, target_achieved

# ===========================================
#  ADDED: Visualization Function
# ===========================================
def visualize_results(model, test_dataset, num_samples=5):
    """Visualize predictions"""
    model.eval()
    mapper = LabelMapper()
    
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = test_dataset[idx]
            image_input = image.unsqueeze(0).to(device)
            
            output = model(image_input)
            prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
            
            # Display
            axes[i, 0].imshow(image.squeeze(), cmap='gray')
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask.numpy(), cmap='viridis', vmin=0, vmax=3)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(prediction, cmap='viridis', vmin=0, vmax=3)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
            
            # Calculate sample dice
            sample_dice = calculate_dice_coefficient(output, mask.unsqueeze(0).to(device))
            dice_text = f'Dice: {np.mean(sample_dice[1:]):.3f}'  # Exclude background
            
            # Overlay
            overlay = np.stack([image.squeeze().numpy()] * 3, axis=-1)
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title(f'Overlay\n{dice_text}')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

# ===========================================
#  MAIN: Fixed Pipeline
# ===========================================
def main():
    data_dir = '/home/groups/comp3710/OASIS/'
    batch_size = 8
    num_epochs = 50
    
    print("Loading datasets...")
    train_dataset = OASISDataset(data_dir, 'keras_png_slices_train', is_training=True)
    val_dataset = OASISDataset(data_dir, 'keras_png_slices_validate', is_training=False)
    test_dataset = OASISDataset(data_dir, 'keras_png_slices_test', is_training=False)
    
    # FIXED: Sample-based class weight calculation
    class_weights = calculate_class_weights(train_dataset, sample_size=1000, method='balanced')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Initialize and train model
    model = UNet(n_channels=1, n_classes=4, base=64).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\nStarting training...")
    model, history = train_model(model, train_loader, val_loader, num_epochs=num_epochs, 
                                class_weights=class_weights, use_focal=True)
    
    print("\nEvaluating on test set...")
    dice_scores, mean_dice, target_achieved = evaluate_model(model, test_loader)
    
    print("\nVisualizing results...")
    visualize_results(model, test_dataset, num_samples=5)
    
    return model, history, dice_scores, target_achieved

if __name__ == "__main__":
    model, history, evaluation_results = main()