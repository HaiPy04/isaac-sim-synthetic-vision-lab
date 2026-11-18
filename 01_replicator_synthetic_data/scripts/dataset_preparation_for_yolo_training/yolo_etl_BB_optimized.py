import os
import cv2
import numpy as np
import glob
import shutil
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ---------------- CONFIG ----------------
INPUT_FOLDER = "omni_dataset_wet_dry" 
OUTPUT_FOLDER = "yolo_dataset_final_wet_dry"
TRAIN_RATIO = 0.9

CLASS_NAMES = ['dry', 'wet']
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

# MASK COLORS IN BGR (OpenCV default) to save a cvtColor step!
# Original RGB (140, 25, 255) -> BGR (255, 25, 140)
# Original RGB (140, 255, 25) -> BGR (25, 255, 140)
MASK_TO_CLASS_BGR = {
    (255, 25, 140): 'dry',
    (25, 255, 140): 'wet',
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
    return data

def process_single_image(pair, split_name):
    """Processes a single image-mask pair. Run in parallel across CPU cores."""
    rgb_path, mask_path = pair
    img = cv2.imread(rgb_path)
    mask = cv2.imread(mask_path)

    if img is None or mask is None:
        return False

    h, w, _ = img.shape
    yolo_lines = []

    # 🔥 OPTIMIZATION: Instead of checking every unique pixel color,
    # target only the specific class colors we expect to be there.
    for color_bgr, class_name in MASK_TO_CLASS_BGR.items():
        class_id = CLASS_MAP[class_name]

        # Highly optimized NumPy vectorized exact-match masking
        binary = (mask[:, :, 0] == color_bgr[0]) & \
                 (mask[:, :, 1] == color_bgr[1]) & \
                 (mask[:, :, 2] == color_bgr[2])
        
        binary = binary.astype(np.uint8) * 255

        # Skip if the class color doesn't exist in this frame
        if not np.any(binary):
            continue

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 10:
                continue

            bx, by, box_w, box_h = cv2.boundingRect(cnt)

            x_center_norm = (bx + (box_w / 2.0)) / w
            y_center_norm = (by + (box_h / 2.0)) / h
            width_norm = box_w / w
            height_norm = box_h / h

            box_str = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
            yolo_lines.append(box_str)

    if yolo_lines:
        filename = os.path.basename(rgb_path)
        
        # Save image directly via OpenCV (Faster disk write than shutil.copy on some filesystems)
        dest_img = os.path.join(OUTPUT_FOLDER, split_name, "images", filename)
        cv2.imwrite(dest_img, img)

        label_name = os.path.splitext(filename)[0] + ".txt"
        dest_label = os.path.join(OUTPUT_FOLDER, split_name, "labels", label_name)
        with open(dest_label, "w") as f:
            f.write("\n".join(yolo_lines))
        return True
        
    return False

def create_yaml():
    yaml = f"""path: {os.path.abspath(OUTPUT_FOLDER)}
train: train/images
val: val/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}"""
    with open(os.path.join(OUTPUT_FOLDER, "dataset.yaml"), "w") as f:
        f.write(yaml.strip())

def main():
    setup_dirs()
    data = get_all_files()
    random.seed(42)
    random.shuffle(data)

    split_idx = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Total pairs found: {len(data)} | Train: {len(train_data)} | Val: {len(val_data)}")
    
    # Automatically get maximum available CPU cores
    num_workers = multiprocessing.cpu_count()
    print(f"🚀 Launching Parallel ETL Engine using {num_workers} CPU cores...")

    # Parallelize processing
    processed_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = []
        for pair in train_data:
            futures.append(executor.submit(process_single_image, pair, "train"))
        for pair in val_data:
            futures.append(executor.submit(process_single_image, pair, "val"))
            
        # Track progress dynamically
        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                processed_count += 1
            if i % 500 == 0 or i == len(data):
                print(f"Processed {i}/{len(data)} files...")

    create_yaml()
    print(f"\n✅ YOLO Object Detection dataset ready! Generated labels for {processed_count} images.")

if __name__ == "__main__":
    main()
