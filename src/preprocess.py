import cv2
import torch
from facenet_pytorch import MTCNN
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

# ================= YOUR PATHS =================
SOURCE_VIDEOS = r"D:\Deepfake_Project\dataset" 
OUTPUT_DIR = r"D:\Deepfake_Project\processed_dataset"
# ==============================================

# Configuration
FRAMES_PER_VIDEO = 5  # Extract 5 frames from each video
FACE_SIZE = 224  # Size for extracted faces (matches training config)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_frames_from_video(video_path, num_frames=5):
    """
    Extract evenly spaced frames from video
    Returns list of frames as numpy arrays
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return []
    
    # Calculate frame indices to extract (evenly spaced)
    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames


def preprocess_dataset():
    """
    Main preprocessing function
    Extracts multiple frames from each video and detects faces
    """
    print("=" * 70)
    print("DEEPFAKE VIDEO PREPROCESSING")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Source: {SOURCE_VIDEOS}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Frames per video: {FRAMES_PER_VIDEO}")
    print(f"Face size: {FACE_SIZE}x{FACE_SIZE}")
    print("=" * 70 + "\n")
    
    # Initialize MTCNN for face detection
    mtcnn = MTCNN(
        keep_all=False,  # Only keep largest face
        select_largest=True,
        margin=20,  # Add margin around face
        post_process=False,
        device=DEVICE,
        image_size=FACE_SIZE  # Resize to training size
    )
    
    source_path = Path(SOURCE_VIDEOS)
    dest_path = Path(OUTPUT_DIR)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Find all videos
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    videos = []
    for ext in video_extensions:
        videos.extend(list(source_path.rglob(ext)))
    
    print(f"Found {len(videos)} videos to process\n")
    
    # Statistics
    stats = {
        'total_videos': len(videos),
        'successful_videos': 0,
        'failed_videos': 0,
        'total_faces_extracted': 0,
        'videos_with_no_faces': 0,
        'errors': [],
        'start_time': datetime.now().isoformat()
    }
    
    # Process each video
    for video_path in tqdm(videos, desc="Processing videos"):
        try:
            # Setup output directory (preserve folder structure)
            relative_path = video_path.relative_to(source_path)
            save_dir = dest_path / relative_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract frames
            frames = extract_frames_from_video(video_path, FRAMES_PER_VIDEO)
            
            if len(frames) == 0:
                stats['failed_videos'] += 1
                stats['errors'].append({
                    'video': str(video_path.name),
                    'error': 'Could not read frames'
                })
                continue
            
            # Process each frame
            faces_found = 0
            for frame_idx, frame in enumerate(frames):
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save path for this frame
                save_name = save_dir / f"{video_path.stem}_frame{frame_idx:02d}.jpg"
                
                # Skip if already processed
                if save_name.exists():
                    faces_found += 1
                    continue
                
                try:
                    # Detect and save face
                    # MTCNN expects numpy array, returns tensor or None
                    face_tensor = mtcnn(frame_rgb, save_path=str(save_name))
                    
                    if face_tensor is not None:
                        faces_found += 1
                        stats['total_faces_extracted'] += 1
                    
                except Exception as e:
                    # Face detection failed for this frame, skip it
                    continue
            
            # Update statistics
            if faces_found > 0:
                stats['successful_videos'] += 1
            else:
                stats['videos_with_no_faces'] += 1
                stats['errors'].append({
                    'video': str(video_path.name),
                    'error': 'No faces detected in any frame'
                })
                
        except Exception as e:
            stats['failed_videos'] += 1
            stats['errors'].append({
                'video': str(video_path.name),
                'error': str(e)
            })
    
    # Finalize statistics
    stats['end_time'] = datetime.now().isoformat()
    
    # Print summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total videos: {stats['total_videos']}")
    print(f"Successfully processed: {stats['successful_videos']}")
    print(f"Videos with no faces: {stats['videos_with_no_faces']}")
    print(f"Failed videos: {stats['failed_videos']}")
    print(f"Total faces extracted: {stats['total_faces_extracted']}")
    print(f"Average faces per successful video: {stats['total_faces_extracted'] / max(stats['successful_videos'], 1):.1f}")
    print("=" * 70)
    
    # Show first few errors
    if stats['errors']:
        print(f"\nFirst 10 errors/issues:")
        for i, error in enumerate(stats['errors'][:10]):
            print(f"  {i+1}. {error['video']}: {error['error']}")
        
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    
    # Save statistics to JSON
    stats_file = dest_path / "preprocessing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to: {stats_file}")
    print(f"Processed faces saved to: {OUTPUT_DIR}\n")
    
    return stats


def analyze_processed_data():
    """
    Analyze the preprocessed dataset
    Call this after preprocessing to see what you have
    """
    print("\n" + "=" * 70)
    print("ANALYZING PROCESSED DATASET")
    print("=" * 70)
    
    dest_path = Path(OUTPUT_DIR)
    
    if not dest_path.exists():
        print(f"Error: Output directory not found: {OUTPUT_DIR}")
        return
    
    # Count images by category
    categories = {}
    for img_path in dest_path.rglob("*.jpg"):
        category = img_path.parent.name
        if category not in categories:
            categories[category] = []
        categories[category].append(img_path)
    
    print(f"\nFound {len(categories)} categories:\n")
    
    total_images = 0
    for category, images in sorted(categories.items()):
        print(f"  {category}: {len(images)} images")
        total_images += len(images)
    
    print(f"\nTotal images: {total_images}")
    print("=" * 70 + "\n")
    
    return categories


if __name__ == "__main__":
    # Run preprocessing
    stats = preprocess_dataset()
    
    # Analyze results
    analyze_processed_data()
    
    print("✓ All done! You can now use the processed dataset for training.")