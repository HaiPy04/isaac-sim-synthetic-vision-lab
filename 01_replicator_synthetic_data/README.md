# Lab 1: Synthetic Data Generation & Parallel YOLO ETL Pipeline

[![NVIDIA Replicator](https://img.shields.io/badge/Omniverse-Replicator-76B900?logo=nvidia&logoColor=white)](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Python Multiprocessing](https://img.shields.io/badge/Python-ProcessPoolExecutor-blue)](https://docs.python.org/3/library/concurrent.futures.html)

This laboratory module focuses on generating photorealistic, domain-randomized synthetic data in **NVIDIA Isaac Sim** using **Omni.Replicator**, degrading optical captures with physical sensor noise via `carb.settings`, and running high-throughput, multi-core ETL pipelines to produce training-ready **YOLOv8** object detection and polygon instance segmentation datasets.

---

## 1. Learning Objectives

By completing this lab, you will master:
1. **Scene Querying & Prim Manipulation**: Referencing existing stage prims and attaching runtime camera randomizers.
2. **Semantic Annotation**: Tagging OpenUSD prims with semantic class schemas (`class: drone`) so Omniverse annotators track ground truth.
3. **Physical Optical Sensor Degradation**: Configuring real-time RTX post-processing effects through the Carbonite configuration interface (`carb.settings`) to simulate sensor ISO grain, scanline artifacts, optical chromatic aberration, and physical shutter motion blur.
4. **Synthetic Data Orchestration**: Defining trigger graphs (`rep.trigger.on_frame`), render products (1080p), and initializing Replicator's `BasicWriter` for multi-channel output (RGB, Semantic Segmentation, 2D Bounding Boxes).
5. **High-Throughput Parallel ETL**: Converting color-coded semantic segmentation masks into normalized bounding boxes (`x_center`, `y_center`, `w`, `h`) using Python's `concurrent.futures.ProcessPoolExecutor` and NumPy vectorization.
6. **Polygon Instance Segmentation Extraction**: Extracting object contours via OpenCV to output normalized polygon segmentation strings for YOLO-Seg.
7. **Quality Assurance & Dataset Verification**: Reversing normalization math to draw visual bounding boxes and merging multi-class captures into a consolidated `dataset.yaml`.

---

## 2. Script-by-Script Breakdown

The pipeline is partitioned into two subdirectories:
```text
01_replicator_synthetic_data/scripts/
├── dataset_creation/                         # Executed via Isaac Sim ./python.sh
│   ├── replicator_noise.py                   # Sensor noise & motion blur capture
│   ├── replicator_prim_onStage.py            # Clean orbital pursuit capture
│   └── replicator_semantic.py                # USD prim semantic tagging
└── dataset_preparation_for_yolo_training/    # Executed via ML Conda Python
    ├── yolo_etl_BB_optimized.py              # Parallelized bounding box ETL (multi-core)
    ├── yolo_etl.py                           # Single-threaded polygon segmentation ETL
    ├── python_BB_verification.py             # Reverse-matrix bounding box QA overlay
    └── yolo_folders_merge_dataset.py         # Multi-folder color-run merger
```

### 2.1 Dataset Creation (Isaac Sim Environment)

#### `scripts/dataset_creation/replicator_semantic.py`
A minimal semantic tagging script. It targets an OpenUSD prim at `/World/shahed_drone` and applies semantic metadata using `rep.modify.semantics([("class", "drone")])`. This semantic tag informs the Replicator annotator pipeline to recognize the drone mesh as a discrete object class during rendering.

> [!NOTE]
> Ensure the prim path targeted matches your stage hierarchy. `replicator_noise.py` queries `/World/shahed_unwrapped`, whereas `replicator_semantic.py` queries `/World/shahed_drone`.

#### `scripts/dataset_creation/replicator_prim_onStage.py`
Captures clean synthetic data of the target drone prim `/World/shahed_unwrapped`.
- **Environment Randomization**: Instantiates a `DomeLight` initialized with an HDRI texture (`bambanani_sunset_4k.exr`). Every frame, `randomize_environment()` rotates the dome light `[0, 360]` degrees and randomizes intensity between `500` and `2500` nits.
- **Orbital Camera**: Randomizes camera position across `[-15, -15, 2]` to `[15, 15, 12]` while maintaining a dynamic `look_at` constraint locked to the target drone.
- **Render Product & Writer**: Captures 1080p (`1920x1080`) frames using `BasicWriter` with `rgb=True` and `semantic_segmentation=True`.

#### `scripts/dataset_creation/replicator_noise.py`
Extends the data generator with realistic physical optical sensor degradation applied directly to the RTX rendering pipeline via `carb.settings`:
- **RTX Motion Blur**:
  ```python
  settings.set("/rtx/post/motionblur/enabled", True)
  settings.set("/rtx/post/motionblur/maxBlurDiameterFraction", 0.05)
  settings.set("/rtx/post/motionblur/numSamples", 8)
  ```
  Coupled with a physical camera shutter window (`shutter:open = 0.0`, `shutter:close = 0.02`, randomized up to `0.035` s), this generates realistic optical streaks based on camera and drone velocity.
- **Sensor ISO Noise & Scanlines**:
  ```python
  settings.set("/rtx/post/tvNoise/enabled", True)
  settings.set("/rtx/post/tvNoise/grainAmount", 0.15)      # Simulates high-ISO sensor gain
  settings.set("/rtx/post/tvNoise/scanlineSpread", 0.002)  # Simulates rolling shutter sensor lines
  ```
- **Chromatic Aberration**:
  ```python
  settings.set("/rtx/post/chromaticAberration/enabled", True)
  settings.set("/rtx/post/chromaticAberration/strength", 0.03)
  ```
- **Writer Output**: Exports RGB images, semantic segmentation masks, and tight 2D bounding boxes (`bounding_box_2d_tight=True`) to disk.

---

### 2.2 Dataset Preparation & ETL (ML Conda Environment)

#### `scripts/dataset_preparation_for_yolo_training/yolo_etl_BB_optimized.py`
High-performance, multi-core bounding-box ETL engine:
- **Optimization Strategy**: Eliminates color space conversion bottlenecks by defining target segmentation mask colors directly in **BGR** format:
  ```python
  MASK_TO_CLASS_BGR = {
      (255, 25, 140): 'dry',  # Original RGB: (140, 25, 255)
      (25, 255, 140): 'wet',  # Original RGB: (140, 255, 25)
  }
  ```
- **NumPy Exact Matching**: Extracts binary masks with vectorized channel comparisons rather than inspecting every unique pixel color:
  ```python
  binary = (mask[:, :, 0] == color_bgr[0]) & \
           (mask[:, :, 1] == color_bgr[1]) & \
           (mask[:, :, 2] == color_bgr[2])
  ```
- **Parallel Processing**: Spawns worker processes using `ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())`, processing hundreds of frames per second.
- **Normalization Math**: Computes bounding rects via `cv2.boundingRect()` and normalizes coordinates:
  $$\text{x\_center} = \frac{x + \frac{w}{2}}{W}, \quad \text{y\_center} = \frac{y + \frac{h}{2}}{H}, \quad \text{width} = \frac{w}{W}, \quad \text{height} = \frac{h}{H}$$
- **Output**: Generates YOLO detection labels: `<class_id> <x_center> <y_center> <w> <h>`.

#### `scripts/dataset_preparation_for_yolo_training/yolo_etl.py`
Polygon instance segmentation ETL pipeline for **YOLO-Seg**:
- Converts OpenCV BGR masks to RGB (`cv2.COLOR_BGR2RGB`).
- Evaluates mask colors using `MASK_TO_CLASS` mappings (`black`, `white`, `yellow`, `blue`, `green`, `red`).
- Extracts object contours via `cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`.
- Normalizes individual polygon vertex points $[(x_i / W), (y_i / H)]$ and writes polygon strings:
  `<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>`.

#### `scripts/dataset_preparation_for_yolo_training/python_BB_verification.py`
Quality assurance script that verifies YOLO annotation accuracy:
- Reads normalized label files and denormalizes bounding box coordinates back to absolute pixel values:
  $$x_1 = \text{int}\left(\left(x_{\text{center}} - \frac{w}{2}\right) \times W\right), \quad y_1 = \text{int}\left(\left(y_{\text{center}} - \frac{h}{2}\right) \times H\right)$$
  $$x_2 = \text{int}\left(\left(x_{\text{center}} + \frac{w}{2}\right) \times W\right), \quad y_2 = \text{int}\left(\left(y_{\text{center}} + \frac{h}{2}\right) \times H\right)$$
- Draws bounding boxes using `cv2.rectangle` and overlays text labels (`dry`: Green, `wet`: Blue) into `verification_output/`.

#### `scripts/dataset_preparation_for_yolo_training/yolo_folders_merge_dataset.py`
Consolidates distinct synthetic data generation batches (e.g., captures performed across separate color runs: `_red_dataset`, `_blue_dataset`, `_green_dataset`, `_yellow_dataset`, `_white_dataset`, `_black_dataset`):
- Prepends class name prefixes to image and label filenames (e.g., `rgb_0001.png` $\to$ `red_rgb_0001.png`) to prevent collision.
- Maps all sources into a unified 6-class dictionary (`CLASS_MAP = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'white': 4, 'black': 5}`).
- Automatically executes a 90/10 train/val split and outputs `yolo_dataset_final/dataset.yaml`.

---

## 3. Configuration & Hardcoded Paths (Sanitization Checklist)

Before executing the scripts on a new workstation, inspect and adjust the following hardcoded paths embedded in the source:

| Script | Line(s) | Hardcoded Path / Assumption | Recommended Production Fix |
| :--- | :--- | :--- | :--- |
| [`replicator_noise.py`](scripts/dataset_creation/replicator_noise.py#L31-L36) | 31, 36 | `HDRI_FOLDER = "/home/haipy/Desktop/isaac/background"`<br/>Fallback: `"/home/haipy/Desktop/isaac/bambanani_sunset_4k.exr"` | Pass via CLI arg (`--hdri-dir`) or use relative repo path |
| [`replicator_noise.py`](scripts/dataset_creation/replicator_noise.py#L43) | 43, 112 | Queries stage prim `/World/shahed_unwrapped` | Verify stage mesh naming before launching script |
| [`replicator_noise.py`](scripts/dataset_creation/replicator_noise.py#L126) | 126 | `output_dir="/home/haipy/Desktop/isaac/target_drone_noise_gaussian"` | Configure via environment variable or CLI argument |
| [`replicator_prim_onStage.py`](scripts/dataset_creation/replicator_prim_onStage.py#L11-L16) | 11, 16, 97 | Same HDRI paths and `output_dir="/home/haipy/Desktop/isaac/target_drone_dataset_test"` | Update to a local workspace directory |
| [`replicator_semantic.py`](scripts/dataset_creation/replicator_semantic.py#L4) | 4 | Targets `/World/shahed_drone` | Rename target to `/World/shahed_unwrapped` if using the sample stage |
| [`yolo_etl.py`](scripts/dataset_preparation_for_yolo_training/yolo_etl.py#L9-L10) | 9, 10 | Expects `omni_dataset/` in current working directory | Point `INPUT_FOLDER` to your Replicator output folder |
| [`yolo_etl_BB_optimized.py`](scripts/dataset_preparation_for_yolo_training/yolo_etl_BB_optimized.py#L11-L12) | 11, 12 | Expects `omni_dataset_wet_dry/` in current working directory | Point `INPUT_FOLDER` to your Replicator output folder |
| [`python_BB_verification.py`](scripts/dataset_preparation_for_yolo_training/python_BB_verification.py#L7) | 7 | Expects `DATASET_DIR = "test_data"` | Set to `yolo_dataset_final_wet_dry` or target output folder |
| [`yolo_folders_merge_dataset.py`](scripts/dataset_preparation_for_yolo_training/yolo_folders_merge_dataset.py#L9-L16) | 9-16 | Expects relative folders `_red_dataset`, `_blue_dataset`, etc. | Ensure folders exist or modify dictionary mappings |

---

## 4. Step-by-Step Execution Commands

### 4.1 Synthetic Data Generation (Isaac Sim Runtime)

1. Launch Isaac Sim with your target OpenUSD drone stage loaded:
   ```bash
   cd ~/.local/share/ov/pkg/isaac-sim-4.2.0  # Or /path/to/isaac-sim
   ```
2. Run the semantic annotation tagger:
   ```bash
   ./python.sh /path/to/isaac-sim-synthetic-vision-lab/01_replicator_synthetic_data/scripts/dataset_creation/replicator_semantic.py
   ```
3. Execute the sensor degradation capture script:
   ```bash
   ./python.sh /path/to/isaac-sim-synthetic-vision-lab/01_replicator_synthetic_data/scripts/dataset_creation/replicator_noise.py
   ```

### 4.2 YOLO Dataset ETL Pipeline (Host Conda Environment)

1. Activate your machine learning Conda environment:
   ```bash
   conda activate yolo_cuda
   cd /path/to/isaac-sim-synthetic-vision-lab/01_replicator_synthetic_data/scripts/dataset_preparation_for_yolo_training
   ```
2. Execute the multi-core bounding-box ETL:
   ```bash
   python yolo_etl_BB_optimized.py
   ```
3. *(Alternative)* Run the polygon segmentation ETL for instance segmentation:
   ```bash
   python yolo_etl.py
   ```
4. Verify bounding box annotations visually:
   ```bash
   python python_BB_verification.py
   ```
5. *(Optional)* If merging multiple single-class generation batches:
   ```bash
   python yolo_folders_merge_dataset.py
   ```

---

## 5. Expected Directory Output & `dataset.yaml` Schema

### Resulting Directory Structure
Upon running `yolo_etl_BB_optimized.py` or `yolo_etl.py`, the output directory (e.g., `yolo_dataset_final_wet_dry/` or `yolo_dataset_final/`) will be structured as follows:

```text
yolo_dataset_final_wet_dry/
├── dataset.yaml
├── train/
│   ├── images/
│   │   ├── rgb_0000.png
│   │   ├── rgb_0001.png
│   │   └── ...
│   └── labels/
│       ├── rgb_0000.txt
│       ├── rgb_0001.txt
│       └── ...
└── val/
    ├── images/
    │   ├── rgb_0090.png
    │   └── ...
    └── labels/
        ├── rgb_0090.txt
        └── ...
```

### Sample `dataset.yaml` Schema
Generated automatically by the ETL scripts:

```yaml
path: /home/haipy/Desktop/isaac-sim-synthetic-vision-lab/01_replicator_synthetic_data/scripts/dataset_preparation_for_yolo_training/yolo_dataset_final_wet_dry
train: train/images
val: val/images

nc: 2
names: ['dry', 'wet']
```

For the 6-class merged dataset generated by `yolo_folders_merge_dataset.py`:
```yaml
path: /home/haipy/Desktop/isaac-sim-synthetic-vision-lab/01_replicator_synthetic_data/scripts/dataset_preparation_for_yolo_training/yolo_dataset_final
train: train/images
val: val/images

nc: 6
names: ['red', 'blue', 'green', 'yellow', 'white', 'black']
```
