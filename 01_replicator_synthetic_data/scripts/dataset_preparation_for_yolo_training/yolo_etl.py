import os
import cv2
import numpy as np
import glob
import shutil
import random

# ---------------- CONFIG ----------------
INPUT_FOLDER = "omni_dataset"
OUTPUT_FOLDER = "yolo_dataset_final"
TRAIN_RATIO = 0.9

# Final class names
CLASS_NAMES = ['black', 'white', 'yellow', 'blue', 'green', 'red']

CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

# 🔥 Mapping from mask color → class name (FROM YOUR ANALYSIS)
MASK_TO_CLASS = {
    (25, 82, 255): 'yellow',
    (25, 255, 82): 'blue',
    (140, 25, 255): 'green',
    (140, 255, 25): 'red',
    (255, 25, 197): 'black',
    (255, 197, 25): 'white',
}

# ----------------------------------------

def setup_dirs():
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)

    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_FOLDER, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, split, 'labels'), exist_ok=True)

def get_all_files():
    rgb_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "rgb_*.png")))

    data = []
    for rgb_path in rgb_files:
        file_id = os.path.basename(rgb_path).split('_')[-1]
        mask_path = os.path.join(INPUT_FOLDER, f"semantic_segmentation_{file_id}")

        if os.path.exists(mask_path):
            data.append((rgb_path, mask_path))
        else:
            print(f"Missing mask for {rgb_path}")

    return data

def process_split(files, split_name):
    for rgb_path, mask_path in files:
        img = cv2.imread(rgb_path)
        mask = cv2.imread(mask_path)

        if img is None or mask is None:
            continue

        h, w, _ = img.shape

        # Convert mask to RGB
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Find unique colors
        unique_colors = np.unique(mask.reshape(-1, 3), axis=0)

        yolo_lines = []

        for color in unique_colors:
            color_tuple = tuple(color.tolist())

            if color_tuple == (0, 0, 0):
                continue

            if color_tuple not in MASK_TO_CLASS:
                print(f"Unknown color: {color_tuple}")
                continue

            class_name = MASK_TO_CLASS[color_tuple]
            class_id = CLASS_MAP[class_name]

            # Create binary mask for this object class
            binary = np.all(mask == color, axis=2).astype(np.uint8) * 255

            # Find contours (individual objects)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) < 10:
                    continue

                polygon = cnt.flatten().astype(float)

                polygon[0::2] /= w
                polygon[1::2] /= h

                poly_str = " ".join([f"{p:.6f}" for p in polygon])
                yolo_lines.append(f"{class_id} {poly_str}")

        # Save only if objects found
        if yolo_lines:
            filename = os.path.basename(rgb_path)

            dest_img = os.path.join(OUTPUT_FOLDER, split_name, "images", filename)
            shutil.copy(rgb_path, dest_img)

            label_name = os.path.splitext(filename)[0] + ".txt"
            with open(os.path.join(OUTPUT_FOLDER, split_name, "labels", label_name), "w") as f:
                f.write("\n".join(yolo_lines))

def create_yaml():
    yaml = f"""
path: {os.path.abspath(OUTPUT_FOLDER)}
train: train/images
val: val/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(os.path.join(OUTPUT_FOLDER, "dataset.yaml"), "w") as f:
        f.write(yaml)

def main():
    setup_dirs()

    data = get_all_files()
    random.seed(42)
    random.shuffle(data)

    split_idx = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Total: {len(data)} | Train: {len(train_data)} | Val: {len(val_data)}")

    process_split(train_data, "train")
    process_split(val_data, "val")

    create_yaml()

    print("\n✅ YOLO dataset ready!")

if __name__ == "__main__":
    main()
