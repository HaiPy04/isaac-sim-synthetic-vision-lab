import os
import cv2
import numpy as np
import glob
import shutil
import random

# --- CONFIGURATION ---
INPUT_FOLDERS = {
    'red': '_red_dataset',     
    'blue': '_blue_dataset',
    'green': '_green_dataset',
    'yellow': '_yellow_dataset',   
    'white': '_white_dataset',    # Fixed typo "datasett"
    'black': '_black_dataset',
}

OUTPUT_FOLDER = "yolo_dataset_final"

# CHANGED: 0.9 is better for 6k images
TRAIN_RATIO = 0.9 

# CRITICAL FIX: Added missing classes
CLASS_MAP = {
    'red': 0,
    'blue': 1,
    'green': 2,
    'yellow': 3,
    'white': 4,
    'black': 5
}

def main():
    # --- Step 1: Setup Folders ---
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_FOLDER, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, split, 'labels'), exist_ok=True)

    # --- Step 2: Get File List from All Folders ---
    all_files = []
    print("Scanning folders...")
    for color, folder in INPUT_FOLDERS.items():
        # Look for images
        rgb_files = sorted(glob.glob(os.path.join(folder, "rgb_*.png")))
        
        if not rgb_files:
            print(f"WARNING: No files found in {folder} for color {color}")
            continue
            
        for path in rgb_files:
            all_files.append({
                'path': path,
                'color': color
            })

    if not all_files:
        print("Error: No files found. Check your INPUT_FOLDERS paths.")
        return

    # Shuffle for random split
    random.seed(42)
    random.shuffle(all_files)

    split_idx = int(len(all_files) * TRAIN_RATIO)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Total Images: {len(all_files)}. Split: {len(train_files)} Train, {len(val_files)} Val.")

    # --- Step 3: Processing Loop ---
    def process_files(files, split_name):
        for entry in files:
            rgb_path = entry['path']
            color = entry['color']
            class_id = CLASS_MAP[color]

            # --- CRITICAL FIX: RENAME FILE ---
            # Old name: rgb_000.png
            # New name: red_rgb_000.png
            original_name = os.path.basename(rgb_path)
            new_filename = f"{color}_{original_name}" 

            # Construct the matching Mask path
            # We assume mask is in the same source folder as the RGB image
            file_id = original_name.split('_')[-1]  # "000.png"
            mask_filename = f"semantic_segmentation_{file_id}"
            mask_path = os.path.join(os.path.dirname(rgb_path), mask_filename)

            if not os.path.exists(mask_path):
                print(f"Skipping {new_filename}: Mask not found at {mask_path}")
                continue

            # Load images
            img = cv2.imread(rgb_path)
            mask = cv2.imread(mask_path) 
            
            if img is None or mask is None:
                print(f"Error reading file: {rgb_path}")
                continue

            h, w, _ = img.shape

            # --- EXTRACT POLYGONS ---
            # Binary mask: everything that is not black is target
            # Note: Ensure your mask colors are actually distinct from background (0,0,0)
            binary_mask = cv2.inRange(mask, (1, 1, 1), (255, 255, 255))

            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            yolo_lines = []
            for cnt in contours:
                if cv2.contourArea(cnt) < 1: 
                    continue

                polygon = cnt.flatten().astype(float)
                polygon[0::2] /= w  # x / width
                polygon[1::2] /= h  # y / height

                # Format: class_id x1 y1 x2 y2 ...
                poly_str = " ".join([f"{coord:.6f}" for coord in polygon])
                yolo_lines.append(f"{class_id} {poly_str}")

            # --- SAVE FILES ---
            if yolo_lines:
                # 1. Copy Image with NEW NAME
                dest_img_path = os.path.join(OUTPUT_FOLDER, split_name, 'images', new_filename)
                shutil.copy(rgb_path, dest_img_path)
                
                # 2. Write Label with NEW NAME
                # red_rgb_000.png -> red_rgb_000.txt
                label_name = os.path.splitext(new_filename)[0] + ".txt"
                with open(os.path.join(OUTPUT_FOLDER, split_name, 'labels', label_name), "w") as f:
                    f.write("\n".join(yolo_lines))

    print("Processing Train set...")
    process_files(train_files, 'train')
    
    print("Processing Val set...")
    process_files(val_files, 'val')

    # --- Step 4: Create YAML Config ---
    yaml_content = f"""
path: {os.path.abspath(OUTPUT_FOLDER)}
train: train/images
val: val/images

nc: {len(CLASS_MAP)}
names: {list(CLASS_MAP.keys())}
"""
    with open(os.path.join(OUTPUT_FOLDER, "dataset.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"\nSuccess! Dataset generated at: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
