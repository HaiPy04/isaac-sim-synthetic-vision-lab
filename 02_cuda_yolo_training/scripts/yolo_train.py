from ultralytics import YOLO
import os

def main():
    # --- Configuration ---
    # 1. Path to the yaml file generated in the previous step
    # Ensure this points correctly to where your conversion script saved it
    DATA_YAML = "yolo_dataset_final_red_yellow/dataset.yaml" 
    
    # 2. Model Selection
    # 'yolov8n-seg.pt' = Nano (Fastest, smallest)
    # 'yolov8s-seg.pt' = Small (Better accuracy, slower)
    MODEL_WEIGHTS = "yolov8n.pt"
    
    # 3. Training Parameters
    EPOCHS = 100           # 50 is plenty for 50 images to see convergence
    BATCH_SIZE = 16        # Adjust based on your GPU VRAM (8 or 16 is safe)
    IMG_SIZE = 640        # Standard YOLO resolution
    DEVICE = '0'            # 0 = First GPU, 'cpu' = CPU

    # --- Check Paths ---
    if not os.path.exists(DATA_YAML):
        print(f"Error: Could not find {DATA_YAML}. Did you run the conversion script?")
        return

    print(f"=== Starting Training: {MODEL_WEIGHTS} on {DATA_YAML} ===")

    # --- Load Model ---
    # Load a pretrained YOLOv8n-seg model
    model = YOLO(MODEL_WEIGHTS)

    # --- Train ---
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        
        # Optimization for small datasets
        workers=4,        # Number of dataloader workers
        project="yolo_training_runs", # Folder to save results
        name="test_run_nano",         # Subfolder name
        exist_ok=True,    # Overwrite if exists
        
        # Augmentation (Optional: disable some for synthetic data if it looks weird)
        # For now, default augmentations are fine.
    )

    print("=== Training Complete ===")
    print(f"Best model saved at: yolo_training_runs/test_run_nano/weights/best.pt")

    # --- Validation / Quick Test ---
    print("\n=== Running Validation ===")
    metrics = model.val()  # Evaluate model performance on the validation set
    print(f"Mean Average Precision (mAP50-95): {metrics.box.map:.4f}")

if __name__ == '__main__':
    # This guard is required for multiprocessing on Windows/Linux
    main()
