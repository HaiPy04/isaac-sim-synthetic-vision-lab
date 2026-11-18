# Troubleshooting & Diagnostic Guide

[![NVIDIA Developer](https://img.shields.io/badge/NVIDIA-Developer%20Support-76B900?logo=nvidia&logoColor=white)](https://forums.developer.nvidia.com/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-Diagnostics-green)](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)

This document provides concrete diagnostic workflows and resolutions for common pitfalls encountered across the **NVIDIA Isaac Sim**, **Omni.Replicator**, **PyTorch/CUDA**, and **OpenUSD** toolchains used in this repository.

---

## 1. Isaac Sim Python Environment Isolation

### Symptom
Running scripts from `01_replicator_synthetic_data/scripts/dataset_creation/` from a standard terminal or Conda environment fails immediately:
```text
Traceback (most recent call last):
  File "replicator_noise.py", line 1, in <module>
    import omni.replicator.core as rep
ModuleNotFoundError: No module named 'omni'
```
or
```text
ModuleNotFoundError: No module named 'carb'
```

### Root Cause
`omni` and `carb` are proprietary C++/Python bindings bundled exclusively inside NVIDIA Omniverse / Isaac Sim. Standard system Python or standalone Conda environments do not have these wheels installed.

### Solution
Always execute synthetic data generation scripts using Isaac Sim's bundled Python wrapper script:

```bash
# Locate your Isaac Sim installation directory
cd ~/.local/share/ov/pkg/isaac-sim-4.2.0  # Adjust version (e.g., 4.x or 5.x)

# Run via Isaac Sim python wrapper
./python.sh /path/to/isaac-sim-synthetic-vision-lab/01_replicator_synthetic_data/scripts/dataset_creation/replicator_noise.py
```

> [!TIP]
> If running in headless mode on a remote server or container without a physical display, add the `--headless` flag or set your headless environment variable:
> ```bash
> ./python.sh /path/to/script.py --headless
> ```

---

## 2. `carb.settings` RTX Post-Processing Key Inconsistencies

### Symptom
In [`replicator_noise.py`](file:///home/haipy/Desktop/isaac-sim-synthetic-vision-lab/01_replicator_synthetic_data/scripts/dataset_creation/replicator_noise.py#L12-L27), post-processing effects (`tvNoise`, `motionblur`, `chromaticAberration`) execute without raising Python exceptions, but the output RGB images contain no grain, blur, or aberration.

### Root Cause
1. **Setting Namespace Discrepancies**: Different versions of Omniverse Kit and Isaac Sim (e.g., Isaac Sim 2023.x vs. 4.x vs. 5.x) migrate post-processing settings between global Carbonite settings and Hydra/Camera-specific render attributes.
2. **Render Product Overrides**: If a `render_product` or active camera has explicit camera attributes enabled, global `/rtx/post/...` keys may be bypassed.

### Diagnostics & Verification
Verify active Carbonite settings directly in Python:
```python
import carb.settings
settings = carb.settings.get_settings()

# Check active values
print("Motion Blur Enabled:", settings.get("/rtx/post/motionblur/enabled"))
print("TV Noise Enabled   :", settings.get("/rtx/post/tvNoise/enabled"))
print("Grain Amount       :", settings.get("/rtx/post/tvNoise/grainAmount"))
```

### Resolution
If global settings are ignored, apply sensor attributes directly to the camera prim inside the Replicator context:
```python
with camera:
    rep.modify.attribute("focalLength", 24.0)
    rep.modify.attribute("shutter:open", 0.0)
    rep.modify.attribute("shutter:close", 0.02)  # Physical motion blur shutter
```
For persistent global overrides across frames, confirm setting initialization occurs *before* instantiating render products:
```python
settings = carb.settings.get_settings()
settings.set("/rtx/post/tvNoise/enabled", True)
settings.set("/rtx/post/tvNoise/grainAmount", 0.15)
settings.set("/rtx/post/motionblur/enabled", True)
```

---

## 3. OpenCV BGR vs. RGB Color-Mask Mismatches in Replicator ETL

### Symptom
The ETL script executes without crashing, but outputs `Generated labels for 0 images`, and label files in `labels/` are empty (0 bytes).

### Root Cause
1. **Color Space Inversion**: OpenCV's `cv2.imread()` loads images in **BGR** format by default.
2. If your mask mapping expects RGB tuples (as in [`yolo_etl.py`](file:///home/haipy/Desktop/isaac-sim-synthetic-vision-lab/01_replicator_synthetic_data/scripts/dataset_preparation_for_yolo_training/yolo_etl.py#L19-L26)):
   ```python
   # RGB tuple: Red channel is 140, Blue is 255
   (140, 25, 255): 'green'
   ```
   Failing to run `cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)` causes OpenCV to evaluate the BGR array `(255, 25, 140)`. The tuple check fails for every pixel, resulting in no detected contours.
3. Conversely, in [`yolo_etl_BB_optimized.py`](file:///home/haipy/Desktop/isaac-sim-synthetic-vision-lab/01_replicator_synthetic_data/scripts/dataset_preparation_for_yolo_training/yolo_etl_BB_optimized.py#L21-L24), the script intentionally keeps the BGR ordering to skip the conversion step:
   ```python
   MASK_TO_CLASS_BGR = {
       (255, 25, 140): 'dry',  # Must match BGR ordering from cv2.imread!
       (25, 255, 140): 'wet',
   }
   ```
   If someone pastes an RGB color tuple here, the vector mask returns `False` everywhere.

### Diagnostic Script: Inspect Actual Mask Colors
Create a one-line scratch script to inspect the exact color tuples in your generated masks:
```python
import cv2
import numpy as np

# Load a sample semantic segmentation mask
mask = cv2.imread("path/to/omni_dataset/semantic_segmentation_0000.png")
print("BGR unique colors:\n", np.unique(mask.reshape(-1, 3), axis=0))

mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
print("RGB unique colors:\n", np.unique(mask_rgb.reshape(-1, 3), axis=0))
```

### Resolution
Ensure that:
- If checking against RGB tuples: apply `cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)`.
- If using `yolo_etl_BB_optimized.py`: configure `MASK_TO_CLASS_BGR` with the exact BGR channel values output by the diagnostic script.

---

## 4. PyTorch CUDA Driver & Toolkit Mismatches

### Symptom
Running [`scripts/verify_cuda.py`](file:///home/haipy/Desktop/isaac-sim-synthetic-vision-lab/02_cuda_yolo_training/scripts/verify_cuda.py) or `yolo_train.py` reports:
```text
[ERROR] CUDA is not available to PyTorch.
```
or
```text
RuntimeError: No CUDA GPUs are available
```
or Ultralytics falls back to CPU training:
```text
WARNING ⚠️ 'device=0' is invalid. Using 'device=cpu' instead.
```

### Root Cause
1. **PyTorch Installed Without CUDA**: A standard `pip install torch` on Linux often installs the CPU-only wheel unless the explicit CUDA index URL is supplied.
2. **Driver Compatibility**: The installed NVIDIA display driver is older than the CUDA runtime requirements of the PyTorch wheel (e.g., CUDA 12.1 requires NVIDIA Driver $\ge 525.60.13$).
3. **Container / Sandbox Isolation**: The terminal session lacks access to the `/dev/nvidia*` device nodes.

### Diagnostic Workflow
1. Check NVIDIA Driver status:
   ```bash
   nvidia-smi
   ```
   Confirm Driver version and GPU name appear.
2. Audit PyTorch build:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Version:', torch.version.cuda)"
   ```
   If `CUDA available` is `False` and `torch.version.cuda` is `None`, your PyTorch installation lacks CUDA binaries.

### Resolution
Reinstall PyTorch built against CUDA 12.1 or 11.8:
```bash
conda activate yolo_cuda
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 5. Missing Primvars & Shader Fallbacks in OpenUSD Thermal Stage

### Symptom
When opening [`OpenUSD stage/thermal_material_stage.usd`](file:///home/haipy/Desktop/isaac-sim-synthetic-vision-lab/03_thermal_sensor_omnigraph/OpenUSD%20stage/thermal_material_stage.usd) in Isaac Sim:
- The `/Chess_Bishop` mesh appears completely black or magenta (unshaded) in Thermal mode.
- The console reports:
  ```text
  [Warning] [rtx.material] Texture file not found: /home/haipy/Desktop/isaac/bambanani_sunset_4k.exr
  ```
  or
  ```text
  [Warning] [omni.hydra] Primvar 'temperature' not found on prim /Chess_Bishop/Chess_Figures_BISHOP
  ```

### Root Cause
1. **Missing Primvars on Custom Meshes**: The thermal shader (`/World/Looks/Thermal`) relies on `primvars:temperature` and `primvars:emissivity`. If you import a new USD asset that lacks these custom attributes, the shader receives `0.0`, resulting in zero radiance (pitch black).
2. **Hardcoded Asset Paths**: The `DomeLight` references an external EXR background on a local workstation path (`/home/haipy/Desktop/isaac/...`).

### Resolution

#### 1. Fix Missing HDRI Sky Path
In Isaac Sim GUI:
1. Select `/Environment/DomeLight` in the Stage tree.
2. In the Property panel under **Dome Light**, locate `Texture File`.
3. Set it to any local HDRI or standard Omniverse asset (e.g., `omniverse://localhost/NVIDIA/Assets/Skies/Clear/noon_sky.hdr`).

#### 2. Author Thermal Primvars on New Prims
To apply thermal attributes to any newly imported mesh via Python in Isaac Sim (`./python.sh` or Script Editor):

```python
from pxr import Usd, UsdGeom, Sdf

# Get current open stage
stage = omni.usd.get_context().get_stage()

# Target prim
mesh_prim = stage.GetPrimAtPath("/World/YourNewObject")
geom = UsdGeom.Gprim(mesh_prim)

# Author temperature (in Kelvin or custom normalized units)
temp_primvar = geom.CreatePrimvar("temperature", Sdf.ValueTypeNames.Float)
temp_primvar.Set(350.0)
temp_primvar.SetInterpolation("constant")

# Author surface emissivity (0.0 to 1.0)
emiss_primvar = geom.CreatePrimvar("emissivity", Sdf.ValueTypeNames.Float)
emiss_primvar.Set(0.92)
emiss_primvar.SetInterpolation("constant")

print("Thermal primvars successfully authored on:", mesh_prim.GetPath())
```
