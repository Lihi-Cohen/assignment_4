from torchvision.datasets import Flowers102
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil


import os
import shutil
import numpy as np
from torchvision.datasets import Flowers102
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def setup_data_from_torchvision(root_dir='./data', output_dir='./dataset_custom', seed=42):
    print("1. Downloading/Loading Flowers102 dataset...")
    d1 = Flowers102(root=root_dir, split='train', download=True)
    d2 = Flowers102(root=root_dir, split='val', download=True)
    d3 = Flowers102(root=root_dir, split='test', download=True)

    # Combine all images and labels
    all_image_paths = d1._image_files + d2._image_files + d3._image_files
    all_labels = d1._labels + d2._labels + d3._labels

    print(f"Total images found: {len(all_image_paths)}")

    # 2. Perform the Assignment-Specific Split (50% Train, 25% Val, 25% Test)
    print("2. Splitting data (50/25/25):")

    # First split: 50% Train vs 50% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_image_paths, all_labels,
        test_size=0.5,
        random_state=seed,
        stratify=all_labels
    )

    # Second split: Temp into Val (25%) and Test (25%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=seed,
        stratify=y_temp
    )

    # 3. Save to Disk (Required for YOLOv5)
    # YOLO requires: dataset/train/class_x/image.jpg

    def save_split(paths, labels, split_name):
        dest_root = os.path.join(output_dir, split_name)
        if os.path.exists(dest_root):
            shutil.rmtree(dest_root) # Clean previous runs

        print(f"Saving {split_name} data...")
        for src_path, label in tqdm(zip(paths, labels), total=len(paths)):
            # Create class folder (0-101)
            class_dir = os.path.join(dest_root, f"class_{label:03d}")
            os.makedirs(class_dir, exist_ok=True)

            # Copy file
            file_name = os.path.basename(src_path)
            shutil.copy2(src_path, os.path.join(class_dir, file_name))

    save_split(X_train, y_train, 'train')
    save_split(X_val, y_val, 'val')
    save_split(X_test, y_test, 'test')

    print(f"Done! Data ready at: {output_dir}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Run the setup
setup_data_from_torchvision(seed=42, output_dir="./dataset_1")
setup_data_from_torchvision(seed=100, output_dir="./dataset_2")

