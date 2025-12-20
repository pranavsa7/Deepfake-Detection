import torch
import os
from pathlib import Path
from datetime import datetime

# Path to your saved model
CHECKPOINT_PATH = "checkpoints/best_model.pth"

def inspect_checkpoint():
    path_obj = Path(CHECKPOINT_PATH)
    
    # 1. Check if file exists
    if not path_obj.exists():
        print(f"❌ Error: File not found at {CHECKPOINT_PATH}")
        print("   This means training likely crashed before the first validation step (Epoch 0).")
        return

    # 2. Check File Timestamp (When did it stop?)
    mod_time = os.path.getmtime(path_obj)
    time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
    
    print("="*40)
    print(f"🕵️ MODEL AUTOPSY REPORT")
    print("="*40)
    print(f"📁 File:       {CHECKPOINT_PATH}")
    print(f"🕒 Last Saved: {time_str}")
    print("-" * 40)

    try:
        # 3. Load the internal dictionary
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        
        # Check if it has the metadata keys we added in the training script
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch'] + 1  # +1 because computer counts from 0
            val_acc = checkpoint.get('val_acc', 0.0)
            val_loss = checkpoint.get('val_loss', 0.0)
            
            print(f"✅ Training Status:  Saved at Epoch {epoch}")
            print(f"🎯 Best Accuracy:    {val_acc:.2f}%")
            print(f"📉 Validation Loss:  {val_loss:.4f}")
            
            # Did it finish?
            if epoch >= 30: # Assuming you set 30 epochs
                print("\n🎉 CONCLUSION: Training COMPLETED successfully!")
            else:
                print(f"\n⚠️ CONCLUSION: Training stopped early (at Epoch {epoch}/30).")
                print("   (This is the best model found *before* the crash/stop).")
        else:
            print("⚠️ Metadata not found. This might be an old model file.")
            
    except Exception as e:
        print(f"❌ Corrupt File: Could not load the model. Error: {e}")

if __name__ == "__main__":
    inspect_checkpoint()