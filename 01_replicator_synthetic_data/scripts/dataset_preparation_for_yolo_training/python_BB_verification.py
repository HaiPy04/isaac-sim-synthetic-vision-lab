import os
import cv2
import glob
import shutil

# ---------------- CONFIG ----------------
DATASET_DIR = "test_data"
OUTPUT_DIR = "verification_output"

CLASS_NAMES = ['dry', 'wet']

# BGR Colors for drawing: Dry = Green, Wet = Blue
CLASS_COLORS = {
    0: (0, 255, 0),    # Green
    1: (255, 0, 0)     # Blue
}
# ----------------------------------------

def verify_split(split_name):
    images_dir = os.path.join(DATASET_DIR, split_name, "images")
    labels_dir = os.path.join(DATASET_DIR, split_name, "labels")
    
    out_split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(out_split_dir, exist_ok=True)
    
    image_files = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))
    print(f"Processing '{split_name}' split: Found {len(image_files)} images.")

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        h, w, _ = img.shape
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        # If no label file exists, just save the clean image to output
        if not os.path.exists(label_path):
            cv2.imwrite(os.path.join(out_split_dir, os.path.basename(img_path)), img)
            continue
            
        # Read and parse YOLO annotations
        with open(label_path, "暢", encoding="utf-8") if hasattr(os, "fake") else open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_w = float(parts[3])
            box_h = float(parts[4])
            
            # --- REVERSE MATRIX MATH: Normalize -> Pixels ---
            # Convert center-relative coordinates back to top-left and bottom-right corners
            x1 = int((x_center - box_w / 2.0) * w)
            y1 = int((y_center - box_h / 2.0) * h)
            x2 = int((x_center + box_w / 2.0) * w)
            y2 = int((y_center + box_h / 2.0) * h)
            
            # Clip values to ensure they stay within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            # Determine styling attributes
            color = CLASS_COLORS.get(class_id, (0, 0, 255)) # Default to Red if class map fails
            class_label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"ID: {class_id}"
            
            # Draw the bounding box rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=3)
            
            # Overlay class name text background for readability
            text_str = f"{class_label}"
            (text_w, text_h), baseline = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            
            # Write text string on top of the boundary box
            cv2.putText(img, text_str, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
        # Export visualized framework back to system storage
        cv2.imwrite(os.path.join(out_split_dir, os.path.basename(img_path)), img)

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Target dataset directory '{DATASET_DIR}' does not exist. Run ETL script first.")
        return
        
    verify_split("train")
    verify_split("val")
    
    print(f"\n✅ Verification complete! Open the '{OUTPUT_DIR}' folder to check annotations.")

if __name__ == "__main__":
    main()
