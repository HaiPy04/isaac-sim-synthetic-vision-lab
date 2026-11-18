# Lab 2: CUDA-Accelerated YOLO Training & Inference

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20CUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF)](https://docs.ultralytics.com/)
[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA%2012.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

This laboratory module guides you through configuring a CUDA-accelerated deep learning environment, auditing GPU and cuDNN availability, and training and evaluating a **YOLOv8** perception model on the synthetic dataset generated in Lab 1.

---

## 1. Learning Objectives

By completing this lab, you will master:
1. **CUDA Environment Validation**: Running automated diagnostics to verify PyTorch GPU binding, CUDA compute capability, device count, and cuDNN tensor acceleration.
2. **YOLOv8 Model Configuration**: Selecting pretrained nano/small architectures (`yolov8n.pt`) and binding training loops to custom OpenUSD synthetic dataset schemas.
3. **Hyperparameter Tuning & Training**: Executing the Ultralytics training loop with hardware-tuned batch sizes, multi-worker dataloaders (`workers=4`), and mixed-precision execution on NVIDIA GPU `0`.
4. **Validation & Perception Metrics**: Evaluating post-training model performance using mean Average Precision ($mAP_{50-95}$ and $mAP_{50}$).
5. **High-Resolution Inference**: Deploying trained checkpoints (`best.pt`) on multi-format target imagery (`*.png`, `*.jpg`, `*.webp`) at high inference resolutions (`1024x1024`) with confidence filtering.

---

## 2. Environment Setup

### 2.1 Conda Environment Creation
Create an isolated Conda environment containing PyTorch with CUDA acceleration and the Ultralytics framework:

```bash
# Create and activate Python 3.10 environment
conda create -n yolo_cuda python=3.10 -y
conda activate yolo_cuda

# Install PyTorch with CUDA support (adjust cuda version matching your NVIDIA Driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Ultralytics perception framework and computer vision libraries
pip install ultralytics opencv-python numpy
```

---

### 2.2 CUDA & Hardware Audit (`scripts/verify_cuda.py`)

The repository includes [`scripts/verify_cuda.py`](file:///home/haipy/Desktop/isaac-sim-synthetic-vision-lab/02_cuda_yolo_training/scripts/verify_cuda.py) to audit your hardware before launching long training jobs.

> [!NOTE]
> If `scripts/verify_cuda.py` is unpopulated (0 bytes) in your clone, use the following production diagnostic script:

```python
import sys
import torch

def verify_environment():
    print("=" * 60)
    print("NVIDIA CUDA & PyTorch Environment Verification")
    print("=" * 60)
    print(f"Python Interpreter : {sys.executable}")
    print(f"PyTorch Version    : {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available     : {cuda_available}")
    
    if not cuda_available:
        print("\n[ERROR] CUDA is not available to PyTorch.")
        print("Please verify NVIDIA Driver installation and PyTorch CUDA build.")
        sys.exit(1)
        
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()
    cudnn_enabled = torch.backends.cudnn.enabled
    
    print(f"CUDA Version       : {cuda_version}")
    print(f"cuDNN Version      : {cudnn_version} (Enabled: {cudnn_enabled})")
    print(f"GPU Device Count   : {device_count}")
    print(f"Active Device ID   : {current_device} ({device_name})")
    
    # Simple tensor allocation benchmark on GPU
    try:
        x = torch.rand((1000, 1000), device="cuda")
        y = torch.matmul(x, x)
        print(f"GPU Tensor Test    : PASSED (Allocated {y.shape} on {y.device})")
    except Exception as e:
        print(f"[ERROR] GPU Tensor allocation failed: {e}")
        sys.exit(1)
        
    print("=" * 60)
    print("Hardware is verified and ready for YOLO training.")
    print("=" * 60)

if __name__ == "__main__":
    verify_environment()
```

Run the audit script:
```bash
python scripts/verify_cuda.py
```

Expected diagnostic output:
```text
============================================================
NVIDIA CUDA & PyTorch Environment Verification
============================================================
Python Interpreter : /home/haipy/miniconda3/envs/yolo_cuda/bin/python
PyTorch Version    : 2.2.0+cu121
CUDA Available     : True
CUDA Version       : 12.1
cuDNN Version      : 8902 (Enabled: True)
GPU Device Count   : 1
Active Device ID   : 0 (NVIDIA RTX A5000)
GPU Tensor Test    : PASSED (Allocated torch.Size([1000, 1000]) on cuda:0)
============================================================
Hardware is verified and ready for YOLO training.
============================================================
```

---

## 3. Training & Inference Workflow

The training module contains two core scripts:
```text
02_cuda_yolo_training/scripts/
├── verify_cuda.py       # Hardware and framework audit
├── yolo_train.py        # Model training loop and validation evaluation
└── yolo_inference.py    # Multi-image perception deployment
```

### 3.1 Model Training (`scripts/yolo_train.py`)

[`scripts/yolo_train.py`](file:///home/haipy/Desktop/isaac-sim-synthetic-vision-lab/02_cuda_yolo_training/scripts/yolo_train.py) configures and initiates the Ultralytics training engine.

#### Parameter Configuration (Directly from Script)
- **Dataset Configuration (`DATA_YAML`)**: `"yolo_dataset_final_red_yellow/dataset.yaml"`
- **Model Architecture (`MODEL_WEIGHTS`)**: `"yolov8n.pt"` (pretrained Nano checkpoint automatically downloaded from Ultralytics)
- **Training Epochs (`EPOCHS`)**: `100` (sufficient for fast convergence on synthetic perception datasets)
- **Batch Size (`BATCH_SIZE`)**: `16` (optimized for 8GB–16GB VRAM GPUs; reduce to `8` if encountering CUDA out-of-memory errors)
- **Input Resolution (`IMG_SIZE`)**: `640` ($640 \times 640$ pixels)
- **Device (`DEVICE`)**: `'0'` (binds execution to NVIDIA GPU 0)
- **Dataloader Workers (`workers`)**: `4` parallel CPU worker threads
- **Output Project (`project`)**: `"yolo_training_runs"`
- **Experiment Subfolder (`name`)**: `"test_run_nano"`
- **Overwrite Protection (`exist_ok`)**: `True`

#### Execution Command
```bash
python scripts/yolo_train.py
```

#### Training Execution Flow
```text
1. Validates existence of dataset.yaml.
2. Initializes YOLO("yolov8n.pt") with pretrained COCO weights.
3. Automatically downloads weights if not present locally.
4. Executes 100 epochs of transfer learning and fine-tuning on synthetic drone/target data.
5. Saves best model weights to: yolo_training_runs/test_run_nano/weights/best.pt.
6. Executes post-training validation on val/ split and outputs Mean Average Precision (mAP50-95).
```

---

### 3.2 Visual Inference Pipeline (`scripts/yolo_inference.py`)

[`scripts/yolo_inference.py`](file:///home/haipy/Desktop/isaac-sim-synthetic-vision-lab/02_cuda_yolo_training/scripts/yolo_inference.py) loads trained weights and runs inference over raw test images.

#### Parameter Configuration (Directly from Script)
- **Model Checkpoint (`MODEL_PATH`)**: `"models/phase1_rouge_jaune.pt"` (points to trained checkpoint)
- **Input Directory (`INPUT_FOLDER`)**: `"Colin_targets"` (folder scanned for test imagery)
- **Output Directory (`OUTPUT_FOLDER`)**: `"inference_results_phase1"`
- **Confidence Threshold (`CONF_THRESHOLD`)**: `0.2` (filters low-confidence false positives)
- **Inference Resolution (`imgsz`)**: `1024` ($1024 \times 1024$ pixels for detecting small, distant airborne targets)
- **Compute Device (`device`)**: `'0'`
- **Supported Formats**: `*.jpg`, `*.jpeg`, `*.png`, `*.bmp`, `*.webp`

#### Execution Command
```bash
python scripts/yolo_inference.py
```

#### Inference Logic
The script loops over every discovered target image, runs `model.predict()`, renders bounding boxes and labels via `result.plot()`, and writes labeled predictions to `inference_results_phase1/pred_{filename}` using OpenCV.

---

## 4. Configuration & Path Assumptions Checklist

Verify these relative path configurations before executing:

| Script | Variable | Configured Value | Description & Adjustment |
| :--- | :--- | :--- | :--- |
| [`yolo_train.py`](file:///home/haipy/Desktop/isaac-sim-synthetic-vision-lab/02_cuda_yolo_training/scripts/yolo_train.py#L8) | `DATA_YAML` | `"yolo_dataset_final_red_yellow/dataset.yaml"` | Ensure this matches the `dataset.yaml` generated in Lab 1 (e.g., `yolo_dataset_final/dataset.yaml` or `yolo_dataset_final_wet_dry/dataset.yaml`). |
| [`yolo_inference.py`](file:///home/haipy/Desktop/isaac-sim-synthetic-vision-lab/02_cuda_yolo_training/scripts/yolo_inference.py#L9) | `MODEL_PATH` | `"models/phase1_rouge_jaune.pt"` | Update to `"yolo_training_runs/test_run_nano/weights/best.pt"` to test your newly trained model. |
| [`yolo_inference.py`](file:///home/haipy/Desktop/isaac-sim-synthetic-vision-lab/02_cuda_yolo_training/scripts/yolo_inference.py#L12) | `INPUT_FOLDER` | `"Colin_targets"` | Target directory containing evaluation images. If missing, the script automatically creates an empty folder. |
| [`yolo_inference.py`](file:///home/haipy/Desktop/isaac-sim-synthetic-vision-lab/02_cuda_yolo_training/scripts/yolo_inference.py#L15) | `OUTPUT_FOLDER` | `"inference_results_phase1"` | Destination folder for visual prediction overlays. |

---

## 5. Expected Artifacts & Metrics

After completing the training and inference pipeline, your directory will contain:

```text
02_cuda_yolo_training/
├── yolo_training_runs/
│   └── test_run_nano/
│       ├── weights/
│       │   ├── best.pt                    # Highest validation mAP checkpoint
│       │   └── last.pt                    # Final epoch checkpoint
│       ├── args.yaml                      # Exact hyperparameters used during training
│       ├── results.csv                    # Epoch-by-epoch loss and mAP telemetry
│       ├── confusion_matrix.png           # Classification accuracy matrix
│       └── val_batch0_pred.jpg            # Visual validation batch predictions
└── inference_results_phase1/
    ├── pred_target_01.png                 # Annotated inference image
    └── pred_target_02.png                 # Annotated inference image
```
