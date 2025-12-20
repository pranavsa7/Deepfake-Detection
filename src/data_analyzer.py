"""
Dataset Analyzer
before training
"""

from pathlib import Path
from collections import defaultdict
import json

class Config:
    """UPDATE THESE PATHS TO YOUR ACTUAL DATASET LOCATIONS"""
    
    # Face datasets (videos)
    FACE_REAL_PATH = r"D:\Deepfake_Project\dataset\real_face"  # Your 890 real face videos
    FACE_FAKE_PATH = r"D:\Deepfake_Project\dataset\fake_face"  # Your 890 fake face videos (FF++ or CelebDF)
    
    # Scene datasets (images)
    SCENE_REAL_PATH = r"D:\Deepfake_Project\dataset\real_scene"  # Your 5913 real scene images
    SCENE_FAKE_PATH = r"D:\Deepfake_Project\dataset\fake_scene"  # Your subset from COCO Fake


def analyze_dataset(root_path, dataset_name):
    """Analyze a single dataset folder"""
    if not Path(root_path).exists():
        return None
    
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    stats = {
        'name': dataset_name,
        'path': str(root_path),
        'total_files': 0,
        'videos': 0,
        'images': 0,
        'video_files': [],
        'image_files': [],
        'subfolders': [],
        'file_extensions': defaultdict(int)
    }
    
    root = Path(root_path)
    
    # Count files recursively
    for file_path in root.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            stats['file_extensions'][ext] += 1
            
            if ext in video_exts:
                stats['videos'] += 1
                if len(stats['video_files']) < 5:  # Store first 5 examples
                    stats['video_files'].append(str(file_path.relative_to(root)))
            elif ext in image_exts:
                stats['images'] += 1
                if len(stats['image_files']) < 5:  # Store first 5 examples
                    stats['image_files'].append(str(file_path.relative_to(root)))
    
    stats['total_files'] = stats['videos'] + stats['images']
    
    # Find subfolders
    for item in root.iterdir():
        if item.is_dir():
            stats['subfolders'].append(item.name)
    
    return stats


def print_section(title):
    """Pretty print section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_stats(stats):
    """Pretty print dataset statistics"""
    if stats is None:
        print("  ❌ Path not found or inaccessible")
        return
    
    print(f"\n  📁 Dataset: {stats['name']}")
    print(f"  📍 Path: {stats['path']}")
    print(f"  📊 Total Files: {stats['total_files']}")
    print(f"     ├─ Videos: {stats['videos']}")
    print(f"     └─ Images: {stats['images']}")
    
    if stats['subfolders']:
        print(f"  📂 Subfolders ({len(stats['subfolders'])}): {', '.join(stats['subfolders'][:5])}")
        if len(stats['subfolders']) > 5:
            print(f"     ... and {len(stats['subfolders']) - 5} more")
    
    if stats['file_extensions']:
        print(f"  📄 File Types:")
        for ext, count in sorted(stats['file_extensions'].items(), key=lambda x: x[1], reverse=True):
            print(f"     ├─ {ext}: {count}")
    
    if stats['video_files']:
        print(f"  🎬 Sample Videos:")
        for vf in stats['video_files']:
            print(f"     ├─ {vf}")
    
    if stats['image_files']:
        print(f"  🖼️  Sample Images:")
        for img in stats['image_files']:
            print(f"     ├─ {img}")


def main():
    """Main analysis function"""
    
    print_section("DEEPFAKE DATASET ANALYZER")
    print("\nAnalyzing your datasets... This may take a few minutes.\n")
    
    # Analyze each dataset
    datasets = {}
    
    print("📊 Scanning datasets...")
    
    # Real faces
    print("\n[1/5] Analyzing real face videos...")
    datasets['real_faces'] = analyze_dataset(Config.FACE_REAL_PATH, "Real Face Videos")
    
    # Fake faces
    print("[2/5] Analyzing fake face videos...")
    datasets['fake_faces'] = analyze_dataset(Config.FACE_FAKE_PATH, "Fake Face Videos")
    
    # Real scenes
    print("[4/5] Analyzing real scene images...")
    datasets['real_scenes'] = analyze_dataset(Config.SCENE_REAL_PATH, "Real Scene Images")
    
    # Fake scenes
    print("[5/5] Analyzing fake scene images...")
    datasets['fake_scenes'] = analyze_dataset(Config.SCENE_FAKE_PATH, "Fake Scene Images (COCO Fake Subset)")
    
    # Print detailed results
    print_section("DETAILED RESULTS")
    
    print("\n FACE DATASETS (Videos)")
    print_stats(datasets['real_faces'])
    print_stats(datasets['fake_faces'])
    
    print("\n\n SCENE DATASETS (Images)")
    print_stats(datasets['real_scenes'])
    print_stats(datasets['fake_scenes'])
    
    # Summary statistics
    print_section("SUMMARY")
    
    # Calculate totals
    total_real_faces = datasets['real_faces']['videos'] if datasets['real_faces'] else 0
    total_fake_faces = datasets['fake_faces']['videos'] if datasets['fake_faces'] else 0
    
    total_real_scenes = datasets['real_scenes']['images'] if datasets['real_scenes'] else 0
    total_fake_scenes = datasets['fake_scenes']['images'] if datasets['fake_scenes'] else 0
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"\n  Face Videos:")
    print(f"     ├─ Real: {total_real_faces:,}")
    print(f"     ├─ Fake: {total_fake_faces:,}")
    print(f"     └─ Total: {total_real_faces + total_fake_faces:,}")
    print(f"     └─ Balance Ratio: 1:{total_fake_faces/total_real_faces:.2f}" if total_real_faces > 0 else "")
    
    print(f"\n  Scene Images:")
    print(f"     ├─ Real: {total_real_scenes:,}")
    print(f"     ├─ Fake: {total_fake_scenes:,}")
    print(f"     └─ Total: {total_real_scenes + total_fake_scenes:,}")
    if total_real_scenes > 0:
        ratio = total_fake_scenes / total_real_scenes
        print(f"     └─ Balance Ratio: 1:{ratio:.2f}")
        
        if ratio > 2.0:
            recommended = int(total_real_scenes * 1.3)
            print(f"\n  ⚠️  IMBALANCE DETECTED!")
            print(f"     Fake scenes are {ratio:.1f}x more than real scenes")
            print(f"     Recommendation: Limit fake scenes to ~{recommended:,} during training")
        else:
            print(f"\n  ✅ Scene data is reasonably balanced!")
    
    print(f"\n  Grand Total:")
    print(f"     └─ {total_real_faces + total_fake_faces + total_real_scenes + total_fake_scenes:,} files")
    
    # Estimated training samples (assuming 5 frames per video)
    frames_per_video = 5
    estimated_face_frames = (total_real_faces + total_fake_faces) * frames_per_video
    estimated_total_samples = estimated_face_frames + total_real_scenes + total_fake_scenes
    
    print(f"\n📊 ESTIMATED TRAINING SAMPLES:")
    print(f"     ├─ From face videos: ~{estimated_face_frames:,} frames ({frames_per_video} frames/video)")
    print(f"     ├─ From scene images: {total_real_scenes + total_fake_scenes:,}")
    print(f"     └─ Total: ~{estimated_total_samples:,} samples")
    
    # Training time estimate
    print(f"\n⏱️  ESTIMATED TRAINING TIME (40 epochs):")
    samples_per_hour = {
        'RTX 3090': 5000,
        'RTX 4090': 7000,
        'A100': 10000,
        'T4': 2000
    }
    
    for gpu, speed in samples_per_hour.items():
        hours = (estimated_total_samples * 40) / speed
        days = hours / 24
        print(f"     ├─ {gpu}: ~{hours:.1f} hours ({days:.1f} days)")
    
    # Save results to JSON
    print_section("SAVING RESULTS")
    
    output_file = "dataset_analysis.json"
    output_data = {
        'datasets': datasets,
        'summary': {
            'total_real_faces': total_real_faces,
            'total_fake_faces': total_fake_faces,
            'total_real_scenes': total_real_scenes,
            'total_fake_scenes': total_fake_scenes,
            'estimated_samples': estimated_total_samples
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_file}")
    
    # Warnings and recommendations
    print_section("RECOMMENDATIONS")
    
    warnings = []
    recommendations = []
    
    # Check for missing datasets
    if datasets['real_faces'] is None:
        warnings.append("Real face dataset path not found!")
    if datasets['fake_faces'] is None:
        warnings.append("Fake face dataset path not found!")
    if datasets['real_scenes'] is None:
        warnings.append("Real scene dataset path not found!")
    if datasets['fake_scenes'] is None:
        warnings.append("Fake scene dataset path not found!")
    
    # Check face balance
    if total_real_faces > 0 and total_fake_faces > 0:
        face_ratio = max(total_real_faces, total_fake_faces) / min(total_real_faces, total_fake_faces)
        if face_ratio > 1.5:
            recommendations.append(f"Face data is imbalanced ({face_ratio:.1f}:1). Consider balancing or using weighted sampling.")
    
    # Check scene balance
    if total_real_scenes > 0 and total_fake_scenes > 0:
        scene_ratio = total_fake_scenes / total_real_scenes
        if scene_ratio > 2.0:
            recommendations.append(f"Scene data is heavily imbalanced ({scene_ratio:.1f}:1 fake:real).")
            recommendations.append(f"The training code will automatically limit fake scenes to {int(total_real_scenes * 1.3):,}")
    
    # Check total dataset size
    if estimated_total_samples < 10000:
        recommendations.append("Small dataset detected. Consider data augmentation and longer training.")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for w in warnings:
            print(f"   • {w}")
    
    if recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for r in recommendations:
            print(f"   • {r}")
    
    if not warnings and not recommendations:
        print("\n✅ Everything looks good! You're ready to train.")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()