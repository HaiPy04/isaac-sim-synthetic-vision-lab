from ultralytics import YOLO
import cv2
import os
import glob

# --- CONFIGURATION ---
# 1. Path to your trained model
# (Check this path matches where your train.py saved the weights)
MODEL_PATH = "models/phase1_rouge_jaune.pt"

# 2. Input folder containing images (jpg, png, etc.)
INPUT_FOLDER = "Colin_targets"

# 3. Output folder to save the visualized results
OUTPUT_FOLDER = "inference_results_phase1"

# 4. Confidence Threshold (0.25 is default, lower it if detections are missed)
CONF_THRESHOLD = 0.2
def main():
    # --- Check Paths ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found.")
        # Create it to be helpful if it's missing
        os.makedirs(INPUT_FOLDER, exist_ok=True)
        print(f"Created empty folder '{INPUT_FOLDER}'. Please put images inside.")
        return

    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # --- Load Model ---
    print(f"Loading model: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # --- Find Images ---
    # Look for common image formats
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    for ext in extensions:
        # glob.glob gives us list of files matching pattern
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    
    if not image_files:
        print(f"No images found in {INPUT_FOLDER}")
        return

    print(f"Found {len(image_files)} images. Starting inference...")

    # --- Run Inference ---
    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        # Run prediction
        # save=True automatically saves to runs/..., but we want custom control
        # so we use result.plot() instead.
        results = model.predict(
            source=img_path, 
            conf=CONF_THRESHOLD, 
            imgsz=1024,
            device='0' # Change to 0 if you fixed your CUDA
        )

        for result in results:
            # result.plot() creates a BGR numpy array with boxes/masks drawn
            annotated_frame = result.plot()

            # Save to our custom output folder
            save_path = os.path.join(OUTPUT_FOLDER, f"pred_{filename}")
            cv2.imwrite(save_path, annotated_frame)
            
            # Optional: Print what was detected
            if result.masks:
                print(f"Detected target in: {filename}")
            else:
                print(f"No target found in: {filename}")

    print(f"\n=== Done! Check the '{OUTPUT_FOLDER}' folder for results. ===")

if __name__ == "__main__":
    main()
