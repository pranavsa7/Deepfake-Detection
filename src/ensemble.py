import torch
import torch.nn as nn
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from facenet_pytorch import MTCNN
from PIL import Image
import sys

# ================= 1. FRIEND'S MODEL (Face Specialist) =================
class FaceBranch(nn.Module):
    def __init__(self, num_classes=2):
        super(FaceBranch, self).__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ================= 2. YOUR MODEL (Context Specialist) =================
# These classes must match your training code EXACTLY

class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, 256)

    def compute_fft(self, x):
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        fft = torch.fft.fft2(gray)
        mag = torch.log(torch.abs(torch.fft.fftshift(fft)) + 1e-8)
        return ((mag - mag.min()) / (mag.max() - mag.min() + 1e-8)).unsqueeze(1)

    def forward(self, x): 
        return self.fc(self.conv_blocks(self.compute_fft(x)).view(x.size(0), -1))

class SpatialBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
    def forward(self, x): return self.backbone(x)

class PatchFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(1280 + 256, 512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(True)
        )
    def forward(self, s, f): return self.fusion(torch.cat([s, f], dim=1))

# --- THIS CLASS WAS MISSING IN THE PREVIOUS VERSION ---
class AttentionAggregation(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(128)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + attn_output).mean(dim=1)
# ------------------------------------------------------

class UniversalDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_branch = SpatialBranch()
        self.frequency_branch = FrequencyBranch()
        self.patch_fusion = PatchFusion()
        # Restored the wrapper class usage:
        self.attention_aggregation = AttentionAggregation()
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, patches):
        feats = [self.patch_fusion(self.spatial_branch(patches[:, i]), self.frequency_branch(patches[:, i])) for i in range(patches.shape[1])]
        feats_stack = torch.stack(feats, dim=1)
        # Using the wrapper class's forward method:
        agg_feat = self.attention_aggregation(feats_stack)
        return self.classifier(agg_feat)

# ================= 3. ENSEMBLE PREDICTOR =================

class EnsemblePredictor:
    def __init__(self, my_ckpt, friend_ckpt, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Initializing Ensemble on {self.device}...")

        # --- Load YOUR Model ---
        print("Loading Scene Model...")
        self.my_model = UniversalDeepfakeDetector().to(self.device)
        self.my_model.load_state_dict(torch.load(my_ckpt, map_location=self.device)['model_state_dict'])
        self.my_model.eval()
        
        self.my_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # --- Load FRIEND'S Model ---
        print("Loading Face Model...")
        self.friend_model = FaceBranch().to(self.device)
        try:
            self.friend_model.load_state_dict(torch.load(friend_ckpt, map_location=self.device))
        except:
            ckpt = torch.load(friend_ckpt, map_location=self.device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                self.friend_model.load_state_dict(ckpt['model_state_dict'])
            else:
                self.friend_model.load_state_dict(ckpt)
                
        self.friend_model.eval()
        self.friend_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=self.device)

    def extract_patches_robust(self, image):
        """
        Cuts image into grid patches to preserve high-frequency details.
        """
        h, w = image.shape[:2]
        patches = []
        
        # 1. Global View
        patches.append(cv2.resize(image, (224, 224)))
        
        # 2. Grid Patches
        stride = 224 // 2
        count = 0
        
        # Resize if too small
        if h < 224 or w < 224:
            image = cv2.resize(image, (400, 400))
            h, w = image.shape[:2]

        for y in range(0, h - 224 + 1, stride):
            for x in range(0, w - 224 + 1, stride):
                patch = image[y:y+224, x:x+224]
                patches.append(patch)
                count += 1
                if count >= 5: break
            if count >= 5: break
            
        while len(patches) < 6:
            patches.append(cv2.resize(image, (224, 224)))
            
        return patches[:6]

    def predict(self, image_path):
        try:
            img_pil = Image.open(image_path).convert('RGB')
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except:
            return None

        print(f"\n🔍 Analyzing: {image_path}")

        # === PATH A: FACE MODEL (0=Fake, 1=Real) ===
        face_prob = 0.0
        face_found = False
        boxes, _ = self.mtcnn.detect(img_pil)
        
        if boxes is not None:
            face_found = True
            box = boxes[0]
            margin = (box[2] - box[0]) * 0.1
            crop_box = [
                max(0, box[0]-margin), max(0, box[1]-margin), 
                min(img_pil.width, box[2]+margin), min(img_pil.height, box[3]+margin)
            ]
            face_crop = img_pil.crop(crop_box)
            friend_input = self.friend_transform(face_crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.friend_model(friend_input)
                probs = torch.softmax(logits, dim=1)
                face_prob = probs[0][0].item() * 100 # Friend's 0=Fake
            
            print(f"   👤 Face Specialist: {face_prob:.2f}% Fake Confidence")
        else:
            print("   👤 Face Specialist: No face found.")

        # === PATH B: SCENE MODEL (1=Fake) ===
        # Use robust patches so frequency branch works
        patches = self.extract_patches_robust(img_cv)
            
        patch_tensors = [self.my_transform(image=p)['image'] for p in patches]
        my_input = torch.stack(patch_tensors).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.my_model(my_input)
            my_prob = torch.softmax(logits, dim=1)[0][1].item() * 100 # Your 1=Fake
            
        print(f"   🌍 Scene Specialist: {my_prob:.2f}% Fake Confidence")

        # === VOTING ===
        if face_found:
            final_score = (face_prob * 0.6) + (my_prob * 0.4)
            method = "Weighted Ensemble"
        else:
            final_score = my_prob
            method = "Scene Only (No Face Found)"

        return {
            "prediction": "FAKE" if final_score > 50 else "REAL",
            "confidence": final_score if final_score > 50 else (100 - final_score),
            "face_score": face_prob,
            "scene_score": my_prob,
            "method": method
        }

if __name__ == "__main__":
    import sys
    MY_MODEL_PATH = "checkpoints/best_model.pth"
    FRIEND_MODEL_PATH = "checkpoints/face_branch_final.pth" 
    
    test_img = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"

    try:
        ensemble = EnsemblePredictor(MY_MODEL_PATH, FRIEND_MODEL_PATH)
        result = ensemble.predict(test_img)
        if result:
            print(f"\n🏆 FINAL VERDICT: {result['prediction']} ({result['confidence']:.2f}%)")
    except Exception as e:
        print(f"\n❌ Error: {e}")