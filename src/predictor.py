import torch
import torch.nn as nn
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from facenet_pytorch import MTCNN
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# ==================== 1. MODEL ARCHITECTURE (Must match train.py) ====================

class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, 256)

    def compute_fft(self, x):
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        fft = torch.fft.fft2(gray)
        magnitude = torch.abs(torch.fft.fftshift(fft))
        magnitude = torch.log(magnitude + 1e-8)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        return magnitude.unsqueeze(1)

    def forward(self, x):
        return self.fc(self.conv_blocks(self.compute_fft(x)).view(x.size(0), -1))

class SpatialBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
    def forward(self, x): return self.backbone(x)

class PatchFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(1280 + 256, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True)
        )
    def forward(self, s, f): return self.fusion(torch.cat([s, f], dim=1))

class AttentionAggregation(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(128)
    def forward(self, x):
        attn_out, weights = self.attention(x, x, x)
        return (self.norm(x + attn_out)).mean(dim=1), weights.mean(dim=1)

class UniversalDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_branch = SpatialBranch()
        self.frequency_branch = FrequencyBranch()
        self.patch_fusion = PatchFusion()
        self.attention_aggregation = AttentionAggregation()
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, patches):
        feats = [self.patch_fusion(self.spatial_branch(patches[:, i]), self.frequency_branch(patches[:, i])) for i in range(patches.shape[1])]
        agg_feat, attn_weights = self.attention_aggregation(torch.stack(feats, dim=1))
        return self.classifier(agg_feat), attn_weights

# ==================== 2. INFERENCE CLASS ====================

class DeepfakeInference:
    def __init__(self, checkpoint_path, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load MTCNN for face cropping
        self.face_detector = MTCNN(keep_all=False, select_largest=True, device=self.device)
        
        # Load Model
        self.model = UniversalDeepfakeDetector().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        self.image_size = 224
        self.num_patches = 6

    def preprocess(self, image_path):
        """Read image, find face, crop it"""
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            return None, None
            
        # Detect face
        boxes, _ = self.face_detector.detect(image)
        
        if boxes is not None:
            # Crop with margin
            box = boxes[0]
            margin = (box[2] - box[0]) * 0.2
            box = [max(0, box[0]-margin), max(0, box[1]-margin), 
                   min(image.width, box[2]+margin), min(image.height, box[3]+margin)]
            image = image.crop(box)
        
        # Convert to numpy (OpenCV format) for patch extraction
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), True

    def extract_patches_and_coords(self, image):
        """Get 6 patches and their coordinates for the heatmap"""
        h, w = image.shape[:2]
        patches = []
        coords = []
        
        # Resize to ensure we can get patches (slightly larger than 224)
        image = cv2.resize(image, (256, 256))
        
        stride = 224 // 2
        for y in range(0, 256 - 224 + 1, stride):
            for x in range(0, 256 - 224 + 1, stride):
                patch = image[y:y+224, x:x+224]
                patches.append(patch)
                coords.append((x, y))
                if len(patches) >= self.num_patches: break
            if len(patches) >= self.num_patches: break
            
        # Fallback if loop fails
        while len(patches) < self.num_patches:
            patches.append(cv2.resize(image, (224, 224)))
            coords.append((0,0))
            
        return patches, coords, image

    def generate_heatmap(self, image, coords, weights):
        """Draw red boxes on high-attention areas"""
        overlay = image.copy()
        weights = weights[0].cpu().numpy()
        
        # Normalize weights 0-1
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        
        for i, (x, y) in enumerate(coords):
            if i >= len(weights): break
            score = weights[i]
            
            # If attention is high, draw red box
            if score > 0.3: 
                intensity = int(255 * score)
                # Draw filled rectangle
                cv2.rectangle(overlay, (x, y), (x+224, y+224), (0, 0, 255), -1)
        
        # Blend
        return cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

    def predict(self, image_path):
        # 1. Get Face
        face_img, found = self.preprocess(image_path)
        if face_img is None: return None
        
        # 2. Get Patches
        patches, coords, resized_face = self.extract_patches_and_coords(face_img)
        
        # 3. Prepare Tensor
        tensors = [self.transform(image=p)['image'] for p in patches]
        batch = torch.stack(tensors).unsqueeze(0).to(self.device)
        
        # 4. Predict
        with torch.no_grad():
            logits, attn_weights = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            fake_prob = probs[0][1].item() * 100
            
        # 5. Generate Heatmap
        heatmap = self.generate_heatmap(resized_face, coords, attn_weights)
        
        return {
            'prediction': 'FAKE' if fake_prob > 50 else 'REAL',
            'confidence': fake_prob if fake_prob > 50 else (100 - fake_prob),
            'heatmap': heatmap,  # <--- THIS IS THE KEY YOU WERE MISSING
            'face_found': found
        }

# ==================== 3. MAIN (Run this file to test) ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    args = parser.parse_args()
    
    detector = DeepfakeInference(args.checkpoint)
    result = detector.predict(args.image)
    
    if result:
        print(f"Result: {result['prediction']} ({result['confidence']:.2f}%)")
        
        # Show Heatmap
        plt.imshow(cv2.cvtColor(result['heatmap'], cv2.COLOR_BGR2RGB))
        plt.title(f"{result['prediction']} - {result['confidence']:.1f}%")
        plt.axis('off')
        plt.show()
    else:
        print("Failed to process image")

if __name__ == "__main__":
    main()