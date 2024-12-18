import numpy as np
import cv2
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Select random frames from dataset')
    parser.add_argument('--src-dir', type=str, required=True,
                      help='Source directory containing frames')
    parser.add_argument('--dst-dir', type=str, required=True,
                      help='Destination directory for frames')
    parser.add_argument('--ratio', type=float, default=0.1,
                      help='Ratio of frames to select (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    return parser.parse_args()

def count_frames(src_dir):
    """Count total available frames with 4-digit format"""
    return len(list(Path(src_dir).glob('frame_????.png')))

def select_frames(src_dir, dst_dir, ratio, seed=42):
    # Setup
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Count frames and calculate selection size
    total_frames = count_frames(src_dir)
    num_frames = int(total_frames * ratio)
    
    if total_frames == 0:
        raise ValueError(f"No frames found in {src_dir}")
    
    # Select frames
    np.random.seed(seed)
    selected = np.random.randint(1, total_frames + 1, num_frames)
    
    # Copy frames with 4-digit format
    for idx in tqdm(selected, desc="Copying frames"):
        src_file = src_path / f"frame_{str(idx).zfill(4)}.png"
        dst_file = dst_path / f"frame_{str(idx).zfill(4)}.png"
        try:
            shutil.copy2(src_file, dst_file)
        except Exception as e:
            print(f"Error copying frame {idx}: {e}")

if __name__ == "__main__":
    args = parse_args()
    select_frames(args.src_dir, args.dst_dir, args.ratio, args.seed)
