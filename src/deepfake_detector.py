import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    # ===== YOUR DATASET PATHS - UPDATE THESE =====
    USE_PREPROCESSED = True  # You ran preprocess.py
    PREPROCESSED_PATH = r"D:\Deepfake_Project\processed_dataset"
    SCENE_REAL_PATH = r"D:\Deepfake_Project\dataset\real_scene"
    SCENE_FAKE_PATH = r"D:\Deepfake_Project\dataset\fake_scene"
    
    # ===== TRAINING SETTINGS =====
    EPOCHS = 30  # Start with 2 for testing, then change to 30 for full training
    BATCH_SIZE = 32  # Start with 4 for testing, then change to 32 for full training
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    
    # ===== MODEL SETTINGS (Optimized for RTX 4060 8GB) =====
    IMAGE_SIZE = 224
    PATCH_SIZE = 224
    NUM_FACE_PATCHES = 3
    NUM_SCENE_PATCHES = 6
    
    # ===== SYSTEM SETTINGS =====
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_DIR = "checkpoints"
    MIXED_PRECISION = True  # FP16 for faster training
    
    # ===== EARLY STOPPING =====
    EARLY_STOP_PATIENCE = 5  # Stop if no improvement for 5 epochs

# ==================== DATA LOADING ====================
class UniversalDeepfakeDataset(Dataset):
    def __init__(self, samples, labels, split='train', augment=True):
        self.samples = samples
        self.labels = labels
        self.split = split
        self.augment = augment
        
        # Augmentation pipeline
        if augment and split == 'train':
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback to next sample if image fails to load
            return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Check if this is a preprocessed face
        is_preprocessed_face = 'processed_dataset' in file_path and 'face' in file_path.lower()
        
        if is_preprocessed_face:
            # Preprocessed face - use as single patch
            resized = cv2.resize(image, (Config.PATCH_SIZE, Config.PATCH_SIZE))
            transformed = self.transform(image=resized)
            patch = transformed['image']
            
            # Create tensor with single patch, pad to expected size
            patches_tensor = patch.unsqueeze(0)
            while patches_tensor.shape[0] < Config.NUM_SCENE_PATCHES:
                patches_tensor = torch.cat([patches_tensor, torch.zeros(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)])
        else:
            # Scene image - extract multiple patches
            patches = self._extract_grid_patches(image)
            transformed_patches = []
            
            for patch in patches[:Config.NUM_SCENE_PATCHES]:
                transformed = self.transform(image=patch)
                transformed_patches.append(transformed['image'])
            
            # Pad if necessary
            while len(transformed_patches) < Config.NUM_SCENE_PATCHES:
                transformed_patches.append(torch.zeros(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            
            patches_tensor = torch.stack(transformed_patches)
        
        return patches_tensor, torch.tensor(label, dtype=torch.long)
    
    def _extract_grid_patches(self, image):
        """Extract overlapping grid patches from scene images"""
        h, w = image.shape[:2]
        patches = []
        stride = Config.PATCH_SIZE // 2
        
        for y in range(0, h - Config.PATCH_SIZE + 1, stride):
            for x in range(0, w - Config.PATCH_SIZE + 1, stride):
                patch = image[y:y+Config.PATCH_SIZE, x:x+Config.PATCH_SIZE]
                patches.append(patch)
                
                if len(patches) >= Config.NUM_SCENE_PATCHES:
                    return patches
        
        # If image too small, resize entire image
        if len(patches) == 0:
            resized = cv2.resize(image, (Config.PATCH_SIZE, Config.PATCH_SIZE))
            patches.append(resized)
        
        return patches


def create_dataloaders(train_ratio=0.8):
    """Create train and validation dataloaders"""
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    
    all_samples = []
    all_labels = []
    
    # Load preprocessed faces
    if Config.USE_PREPROCESSED:
        preprocessed_path = Path(Config.PREPROCESSED_PATH)
        
        # Real faces
        real_face_path = preprocessed_path / "real_face"
        if real_face_path.exists():
            real_faces = list(real_face_path.rglob("*.jpg"))
            all_samples.extend([str(f) for f in real_faces])
            all_labels.extend([0] * len(real_faces))
            print(f"✓ Loaded {len(real_faces):,} real face images")
        
        # Fake faces
        fake_face_path = preprocessed_path / "fake_face"
        if fake_face_path.exists():
            fake_faces = list(fake_face_path.rglob("*.jpg"))
            all_samples.extend([str(f) for f in fake_faces])
            all_labels.extend([1] * len(fake_faces))
            print(f"✓ Loaded {len(fake_faces):,} fake face images")
    
    # Load scene images
    scene_real_path = Path(Config.SCENE_REAL_PATH)
    if scene_real_path.exists():
        real_scenes = (list(scene_real_path.rglob("*.jpg")) + 
                      list(scene_real_path.rglob("*.png")) + 
                      list(scene_real_path.rglob("*.jpeg")) +
                      list(scene_real_path.rglob("*.JPG")) +
                      list(scene_real_path.rglob("*.PNG")) +
                      list(scene_real_path.rglob("*.JPEG")))
        all_samples.extend([str(f) for f in real_scenes])
        all_labels.extend([0] * len(real_scenes))
        print(f"✓ Loaded {len(real_scenes):,} real scene images")
    
    scene_fake_path = Path(Config.SCENE_FAKE_PATH)
    if scene_fake_path.exists():
        fake_scenes = (list(scene_fake_path.rglob("*.jpg")) + 
                      list(scene_fake_path.rglob("*.png")) + 
                      list(scene_fake_path.rglob("*.jpeg")) +
                      list(scene_fake_path.rglob("*.JPG")) +
                      list(scene_fake_path.rglob("*.PNG")) +
                      list(scene_fake_path.rglob("*.JPEG")))
        
        # Balance fake scenes with real scenes if needed
        if len(fake_scenes) > len(real_scenes) * 1.5:
            import random
            fake_scenes = random.sample(fake_scenes, int(len(real_scenes) * 1.3))
            print(f"✓ Balanced fake scenes to {len(fake_scenes):,} images")
        
        all_samples.extend([str(f) for f in fake_scenes])
        all_labels.extend([1] * len(fake_scenes))
        print(f"✓ Loaded {len(fake_scenes):,} fake scene images")
    
    # Split train/val
    print(f"\nTotal samples: {len(all_samples):,}")
    real_count = all_labels.count(0)
    fake_count = all_labels.count(1)
    print(f"  Real: {real_count:,} | Fake: {fake_count:,} | Ratio: {fake_count/real_count:.2f}:1")
    
    indices = np.random.permutation(len(all_samples))
    split_idx = int(len(all_samples) * train_ratio)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_samples = [all_samples[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    # Create datasets
    train_dataset = UniversalDeepfakeDataset(train_samples, train_labels, split='train', augment=True)
    val_dataset = UniversalDeepfakeDataset(val_samples, val_labels, split='val', augment=False)
    
    print(f"\nTrain set: {len(train_dataset):,} samples")
    print(f"Val set: {len(val_dataset):,} samples")
    print("=" * 70 + "\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ==================== MODEL ARCHITECTURE ====================

class FrequencyBranch(nn.Module):
    """Frequency domain feature extraction using FFT"""
    def __init__(self):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(256, 256)
    
    def compute_fft(self, x):
        """Compute FFT magnitude spectrum"""
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        fft = torch.fft.fft2(gray)
        fft_shifted = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shifted)
        magnitude = torch.log(magnitude + 1e-8)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        return magnitude.unsqueeze(1)
    
    def forward(self, x):
        fft_input = self.compute_fft(x)
        features = self.conv_blocks(fft_input)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features


class SpatialBranch(nn.Module):
    """Spatial feature extraction using EfficientNet"""
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
    
    def forward(self, x):
        return self.backbone(x)


class PatchFusion(nn.Module):
    """Fuse spatial and frequency features"""
    def __init__(self, spatial_dim=1280, freq_dim=256, hidden_dim=512, output_dim=128):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, spatial_feat, freq_feat):
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.fusion(combined)


class AttentionAggregation(nn.Module):
    """Aggregate patch features using self-attention"""
    def __init__(self, feature_dim=128, num_heads=4):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x):
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        aggregated = x.mean(dim=1)
        attn_weights = attn_weights.mean(dim=1)
        return aggregated, attn_weights


class UniversalDeepfakeDetector(nn.Module):
    """Complete deepfake detection model"""
    def __init__(self):
        super().__init__()
        
        self.spatial_branch = SpatialBranch()
        self.frequency_branch = FrequencyBranch()
        self.patch_fusion = PatchFusion(spatial_dim=1280, freq_dim=256, hidden_dim=512, output_dim=128)
        self.attention_aggregation = AttentionAggregation(feature_dim=128, num_heads=4)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def forward(self, patches):
        batch_size, num_patches = patches.shape[:2]
        
        patch_features = []
        for i in range(num_patches):
            patch = patches[:, i]
            spatial_feat = self.spatial_branch(patch)
            freq_feat = self.frequency_branch(patch)
            fused_feat = self.patch_fusion(spatial_feat, freq_feat)
            patch_features.append(fused_feat)
        
        patch_features = torch.stack(patch_features, dim=1)
        aggregated_feat, attn_weights = self.attention_aggregation(patch_features)
        logits = self.classifier(aggregated_feat)
        
        return logits, attn_weights


# ==================== TRAINING ====================

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    from tqdm import tqdm
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for patches, labels in pbar:
        patches = patches.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, _ = model(patches)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(patches)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    from tqdm import tqdm
    
    with torch.no_grad():
        for patches, labels in tqdm(dataloader, desc="Validating", leave=False):
            patches = patches.to(device)
            labels = labels.to(device)
            
            logits, _ = model(patches)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    """Main training function"""
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create save directory
    Path(Config.SAVE_DIR).mkdir(exist_ok=True)
    
    # Print configuration
    print("\n" + "=" * 70)
    print("DEEPFAKE DETECTION TRAINING")
    print("=" * 70)
    
    if Config.DEVICE == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print(f"📊 Epochs: {Config.EPOCHS}")
    print(f"📦 Batch size: {Config.BATCH_SIZE}")
    print(f"🖼️  Image size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    print(f"⚡ Mixed precision: {Config.MIXED_PRECISION}")
    print("=" * 70 + "\n")
    
    # Load data
    train_loader, val_loader = create_dataloaders()
    
    # Initialize model
    print("Initializing model...")
    model = UniversalDeepfakeDetector().to(Config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} ({trainable_params:,} trainable)\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if Config.MIXED_PRECISION and Config.DEVICE == 'cuda' else None
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")
    
    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE, scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
        
        # Update scheduler
        scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Print GPU memory
        if Config.DEVICE == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, f"{Config.SAVE_DIR}/best_model.pth")
            print(f"✓ Saved best model! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        print()
        
        # Early stopping
        if patience_counter >= Config.EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"No improvement for {Config.EARLY_STOP_PATIENCE} consecutive epochs")
            break
        
        # Clear cache
        if Config.DEVICE == 'cuda':
            torch.cuda.empty_cache()
    
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {Config.SAVE_DIR}/best_model.pth")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()