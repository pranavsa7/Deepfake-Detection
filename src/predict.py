import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import sys

# ==================== MODEL ARCHITECTURE (Same as training) ====================

class FrequencyBranch(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
    
    def forward(self, x):
        return self.backbone(x)


class PatchFusion(nn.Module):
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


# ==================== INFERENCE ====================

class DeepfakeInference:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.image_size = 224
        self.num_patches = 6
        
        # Load model
        print("Loading model from checkpoints/best_model.pth...")
        self.model = UniversalDeepfakeDetector().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        # Transform
        self.transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def extract_patches(self, image):
        """Extract grid patches from image"""
        h, w = image.shape[:2]
        patches = []
        stride = self.image_size // 2
        
        for y in range(0, h - self.image_size + 1, stride):
            for x in range(0, w - self.image_size + 1, stride):
                patch = image[y:y+self.image_size, x:x+self.image_size]
                patches.append(patch)
                
                if len(patches) >= self.num_patches:
                    return patches
        
        # If image too small, resize entire image
        if len(patches) == 0:
            resized = cv2.resize(image, (self.image_size, self.image_size))
            patches.append(resized)
        
        return patches
    
    def predict(self, image_path):
        """Predict if image is real or fake"""
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract patches
        patches = self.extract_patches(image)
        
        # Transform patches
        transformed_patches = []
        for patch in patches[:self.num_patches]:
            transformed = self.transform(image=patch)
            transformed_patches.append(transformed['image'])
        
        # Pad if necessary
        while len(transformed_patches) < self.num_patches:
            transformed_patches.append(torch.zeros(3, self.image_size, self.image_size))
        
        # Stack and add batch dimension
        patches_tensor = torch.stack(transformed_patches).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits, attention = self.model(patches_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
            # Class 0 = REAL, Class 1 = FAKE
            real_prob = probabilities[0][0].item() * 100
            fake_prob = probabilities[0][1].item() * 100
            
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        return {
            'predicted_class': predicted_class,
            'prediction': 'FAKE' if predicted_class == 1 else 'REAL',
            'real_probability': real_prob,
            'fake_probability': fake_prob,
            'confidence': max(real_prob, fake_prob)
        }


# ==================== MAIN ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"❌ Error: Image not found: {args.image}")
        sys.exit(1)
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Get image filename
    image_name = Path(args.image).name
    
    print(f"📸 Detected IMAGE input: {image_name}")
    print("🔮 Analyzing...\n")
    
    # Initialize inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = DeepfakeInference(args.checkpoint, device=device)
    
    # Predict
    result = detector.predict(args.image)
    
    # Print results
    print("=" * 40)
    print("🔍 ANALYSIS REPORT")
    print("=" * 40)
    print(f"Input Type:        IMAGE")
    print(f"Real Probability:  {result['real_probability']:.2f}%")
    print(f"Fake Probability:  {result['fake_probability']:.2f}%")
    print("-" * 40)
    
    if result['prediction'] == 'REAL':
        print("✅ RESULT: REAL CONTENT")
    else:
        print("⚠️  RESULT: FAKE/MANIPULATED CONTENT")
    
    print("=" * 40)
    print()


if __name__ == "__main__":
    main()