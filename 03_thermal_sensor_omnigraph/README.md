# Lab 3: OpenUSD Thermal Sensor Stage & OmniGraph View Switching

[![OpenUSD](https://img.shields.io/badge/OpenUSD-v23.11-007ACC)](https://openusd.org/)
[![NVIDIA MDL](https://img.shields.io/badge/NVIDIA-Material%20Definition%20Language-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/mdl)
[![OmniGraph](https://img.shields.io/badge/Omniverse-OmniGraph-orange)](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html)

This laboratory module explores physics-based synthetic multi-spectral vision in **NVIDIA Isaac Sim**. You will inspect and operate an **OpenUSD** stage configured with procedural **MDL (Material Definition Language)** thermal infrared shaders, custom geometry `primvars` (temperature and emissivity), and an **OmniGraph (ActionGraph)** execution network that dynamically switches the viewport between standard RGB and simulated Long-Wave Infrared (LWIR) thermal sensor modes.

---

## 1. Learning Objectives

By completing this lab, you will master:
1. **OpenUSD Stage Composition**: Inspecting prim hierarchies, custom scopes (`/Looks`), cameras (`/World/Thermal_camera`), and geometric meshes in OpenUSD.
2. **Procedural Thermal Physics via MDL**: Modeling infrared radiant emission based on Stefan-Boltzmann radiation principles ($E \propto \varepsilon \cdot T^4$) using procedural math nodes in Omniverse Material Graph.
3. **Custom USD Primvars**: Binding arbitrary per-prim simulation properties (`temperature`, `emissivity`, `Max_temp`) that feed procedural shaders at render time.
4. **OmniGraph Visual Scripting**: Creating an event-driven `ActionGraph` that processes keyboard triggers to dynamically alter prim attributes, material bindings, and RTX post-processing parameters in real time.
5. **Multi-Spectral Perception Pipelines**: Generating paired ground-truth RGB and simulated Thermal Infrared imagery for multi-modal sensor fusion algorithms.

---

## 2. OpenUSD Stage Architecture

The stage is located at:
[`OpenUSD stage/thermal_material_stage.usd`](file:///home/haipy/Desktop/isaac-sim-synthetic-vision-lab/03_thermal_sensor_omnigraph/OpenUSD%20stage/thermal_material_stage.usd)

### Stage Prim Hierarchy

```text
/World (defaultPrim) [Xform]
├── Looks [Scope]
│   ├── Thermal [Material]                   <-- Custom procedural thermal MDL shader
│   └── Ash_Planks [Material]                <-- PBR surface material
├── Thermal_camera [Camera]                  <-- Synthetic sensor camera
├── ActionGraph [OmniGraph]                  <-- Runtime event graph for view switching
│   ├── on_keyboard_input [OmniGraphNode]    <-- Trigger key listener (RGB mode)
│   ├── on_keyboard_input_01 [OmniGraphNode] <-- Trigger key listener (Thermal mode)
│   ├── write_prim_attribute [OmniGraphNode] <-- Assigns material binding
│   ├── write_prim_attribute_01 to 05        <-- Multi-prim attribute assignment
│   ├── write_setting [OmniGraphNode]        <-- Alters RTX render settings
│   ├── write_setting_01 [OmniGraphNode]     <-- Alters RTX render settings
│   ├── constant_bool [OmniGraphNode]        <-- Boolean state toggle
│   └── constant_bool_01 [OmniGraphNode]     <-- Boolean state toggle
├── Chess_Bishop [Xform]
│   ├── Chess_Figures_BISHOP [Mesh]          <-- Target object carrying thermal primvars
│   └── Looks [Scope]
│       ├── White [Material] -> Shader       <-- Default visible spectrum PBR material
│       └── Black [Material]
└── Environment [Xform]
    ├── defaultLight [DistantLight]          <-- Solar illumination
    ├── Plane [Mesh]                         <-- Ground plane
    └── DomeLight [DomeLight]                <-- Sky dome HDRI
```

---

## 3. Thermal Radiation Modeling & Material Graph

### 3.1 Physics of Synthetic Thermal Vision
In the long-wave infrared (LWIR, 8–14 $\mu m$) spectrum, optical sensors measure emitted thermal radiation rather than reflected ambient photons. The total radiant emittance ($W/m^2$) of a grey-body surface is governed by the **Stefan-Boltzmann Law**:

$$E = \varepsilon \cdot \sigma \cdot T^4$$

where:
- $\varepsilon \in [0.0, 1.0]$ is surface **emissivity** (dimensionless material property).
- $\sigma = 5.670374 \times 10^{-8} \, \text{W}/(\text{m}^2 \cdot \text{K}^4)$ is the Stefan-Boltzmann constant.
- $T$ is absolute **temperature** in Kelvin.

### 3.2 Material Graph Inspection (`/World/Looks/Thermal`)
Open the **Material Graph** panel in Isaac Sim and double-click `/World/Looks/Thermal`.

![Thermal Material Graph](image_guide/Screenshot%20from%202026-11-18%2011-33-29.png)

#### Mathematical Network Architecture:
1. **Primvar Readers**:
   - Reads `primvars:temperature` (object surface temperature).
   - Reads `primvars:emissivity` (surface emissivity factor).
   - Reads `inputs:Max_temp` (normalization ceiling).
2. **Procedural Math Nodes**:
   - **Normalize**: Scales object temperature relative to the maximum scene temperature:
     $$T_{\text{norm}} = \frac{T - T_{\text{min}}}{T_{\text{max}} - T_{\text{min}}}$$
   - **Power Node (`pow`)**: Approximates non-linear radiative scaling ($T^4$ curve).
   - **Emissivity Modulator**: Multiplies normalized temperature by surface emissivity:
     $$I_{\text{thermal}} = \varepsilon \cdot (T_{\text{norm}})^4$$
3. **Surface & Emission Output**:
   - Feeds $I_{\text{thermal}}$ into the **Emissive Color** and **Diffuse Tint** ports of the MDL surface shader.
   - Hot objects emit bright, high-value pixels (radiating white/light grey), while cold background objects remain dark.

---

## 4. OmniGraph View Switching Walkthrough

### 4.1 Baseline RGB Mode
When the stage is initially loaded in Isaac Sim, the `/Chess_Bishop` mesh is bound to standard visible PBR materials (`/Chess_Bishop/Looks/White`), illuminated by sunlight and the sky dome.

![RGB Viewport Baseline](image_guide/Screenshot%20from%202026-11-18%2011-32-56.png)

In the screenshot above:
- Viewport displays standard RGB illumination.
- The stage hierarchy panel (top right) exposes `/World/ActionGraph`.
- The ActionGraph editor (bottom) shows the keyboard event listener connected to `write_prim_attribute` and `write_setting` nodes.

---

### 4.2 Simulated Thermal IR Mode
Pressing the trigger key configured in the ActionGraph triggers the runtime graph:
1. `on_keyboard_input` captures the key event.
2. `write_prim_attribute` rebinds `/Chess_Bishop/Chess_Figures_BISHOP` material binding to `/World/Looks/Thermal`.
3. `write_setting` toggles the viewport post-processing settings, suppressing ambient visible light and isolating thermal emission.

![Simulated Thermal Viewport](image_guide/Screenshot%20from%202026-11-18%2011-33-02.png)

In the thermal viewport:
- Background grass, terrain, and distant trees drop into deep black/dark tones (ambient background temperature $\approx 290\,\text{K}$).
- The target bishop glows with intense white thermal luminance ($\approx 350\,\text{K}$ high-heat target signature).
- Emissivity variations create subtle surface temperature gradients across the object contours.

---

## 5. Step-by-Step Practical Walkthrough

### Step 1: Open the Stage in Isaac Sim
1. Launch Isaac Sim with GUI:
   ```bash
   cd ~/.local/share/ov/pkg/isaac-sim-4.2.0
   ./isaac-sim.sh
   ```
2. Open `File` $\to$ `Open` and select:
   `03_thermal_sensor_omnigraph/OpenUSD stage/thermal_material_stage.usd`

### Step 2: Inspect & Modify Temperature Primvars
1. In the **Stage** panel, navigate to `/Chess_Bishop/Chess_Figures_BISHOP`.
2. In the **Property** panel below, locate the **Primvars** section:
   - Notice the custom attributes:
     - `primvars:temperature`: Numeric value representing surface heat.
     - `primvars:emissivity`: Float between `0.0` and `1.0`.
3. Modify `primvars:temperature` (e.g., lower it to match ambient, or raise it to simulate an operating engine or heated payload).

### Step 3: Test Runtime OmniGraph Toggle
1. Press the **Play** button on the left Isaac Sim simulation toolbar to start the OmniGraph execution engine.
2. Press the configured trigger key (or manually toggle the execution in Action Graph) to switch between the RGB PBR view and the Thermal IR view.
3. Observe real-time shader re-evaluation as the bishop transitions between visible and infrared modalities.

---

## 6. Configuration & Asset Portability Notes

When transferring this stage to a new machine or compute cluster, verify the following:

| Component | Path / Asset Reference in USD | Potential Portability Issue | Resolution |
| :--- | :--- | :--- | :--- |
| **DomeLight HDRI** | Points to external `bambanani_sunset_4k.exr` | File path may fail if the user's local directory differs from `/home/haipy/Desktop/...` | Re-point the `inputs:texture:file` attribute on `/Environment/DomeLight` to your local HDRI folder. |
| **MDL Shader References** | References Omniverse standard core MDL libraries | Custom MDL shader search paths must include NVIDIA core shader directories | Isaac Sim includes standard MDLs automatically; verify in `Preferences -> Materials -> Search Paths`. |
| **Folder Naming** | `OpenUSD stage/` contains a space | Shell commands failing without quoting | Always wrap paths in double quotes in terminal commands (`"OpenUSD stage/thermal_material_stage.usd"`). |
