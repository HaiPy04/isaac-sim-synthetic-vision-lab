import omni.replicator.core as rep
import carb.settings
import random
import os
import glob

print("=== Starting Shahed Target Drone Synthetic Data Generation with Sensor Noise ===")

# ------------------------------------------------
# 0. CONFIGURE RTX SENSOR & RENDER SETTINGS
# ------------------------------------------------
settings = carb.settings.get_settings()

# Enable Physical Motion Blur in RTX Renderer
settings.set("/rtx/post/motionblur/enabled", True)
settings.set("/rtx/post/motionblur/maxBlurDiameterFraction", 0.05)
settings.set("/rtx/post/motionblur/numSamples", 8)

# Enable Sensor ISO Grain / Noise Simulation
settings.set("/rtx/post/tvNoise/enabled", True)
settings.set("/rtx/post/tvNoise/grainAmount", 0.15)       # Simulates sensor ISO grain
settings.set("/rtx/post/tvNoise/scanlineSpread", 0.002)   # Simulates rolling shutter / line artifacts

# Enable Chromatic Aberration & Lens Blur
settings.set("/rtx/post/chromaticAberration/enabled", True)
settings.set("/rtx/post/chromaticAberration/strength", 0.03)

# ------------------------------------------------
# 1. SETUP PATHS & LOAD FILES
# ------------------------------------------------
HDRI_FOLDER = "/home/haipy/Desktop/isaac/background"
hdri_files = sorted(glob.glob(os.path.join(HDRI_FOLDER, "*")))

if len(hdri_files) == 0:
    print(f"WARNING: No HDRIs found in {HDRI_FOLDER}. Using fallback.")
    hdri_files = ["/home/haipy/Desktop/isaac/bambanani_sunset_4k.exr"]

print(f"Loaded {len(hdri_files)} HDRI background skies.")

# ------------------------------------------------
# 2. REFERENCE EXISTING SHAHED DRONE
# ------------------------------------------------
shahed_prim = rep.get.prims(path_pattern="/World/shahed_unwrapped")

# ------------------------------------------------
# 3. HDRI DOME LIGHT (Sky & Sun Lighting)
# ------------------------------------------------
dome_light = rep.create.light(
    light_type="Dome",
    texture=hdri_files[0],
    intensity=1000
)

# ------------------------------------------------
# 4. INTERCEPTOR CAMERA (Physical Optical Settings)
# ------------------------------------------------
camera = rep.create.camera()

with camera:
    rep.modify.attribute("focalLength", 24.0)
    #rep.modify.attribute("fStop", 2.8)                    # Slight optical depth-of-field blur
    rep.modify.attribute("focusDistance", 10.0)
    rep.modify.attribute("exposure", 0.0)
    
    # Motion Blur Shutter Timing (Physical Shutter Window)
    rep.modify.attribute("shutter:open", 0.0)
    rep.modify.attribute("shutter:close", 0.02)           # 1/50s shutter speed creates physical motion streak

# ------------------------------------------------
# 5. RANDOMIZERS
# ------------------------------------------------
@rep.randomizer.register
def randomize_shahed_flight():
    with shahed_prim:
        rep.modify.pose(
            position=rep.distribution.uniform((-1.5, -1.5, -1.0), (1.5, 1.5, 1.0)),
            rotation=rep.distribution.uniform((-20.0, -180.0, -20.0), (20.0, 180.0, 20.0)),
        )

@rep.randomizer.register
def randomize_camera_sensor_noise():
    # Dynamically vary ISO noise, shutter speed, and exposure per frame
    with camera:
        rep.modify.attribute("exposure", rep.distribution.uniform(-0.8, 1.2))
        rep.modify.attribute("fStop", rep.distribution.uniform(1.8, 4.0))
        # Varying shutter timing randomizes motion blur intensity across velocities
        rep.modify.attribute("shutter:close", rep.distribution.uniform(0.008, 0.035))

@rep.randomizer.register
def randomize_environment():
    with dome_light:
        rep.modify.attribute(
            "inputs:texture:file",
            rep.distribution.choice(hdri_files)
        )
        rep.modify.pose(rotation=rep.distribution.uniform((0, 0, 0), (0, 360, 0)))
        rep.modify.attribute("inputs:intensity", rep.distribution.uniform(500, 2500))

# ------------------------------------------------
# 6. TRIGGER GRAPH (Frame Loop)
# ------------------------------------------------
NUM_FRAMES = 10

with rep.trigger.on_frame(num_frames=NUM_FRAMES):
    # Dynamic camera position with tracking look_at
    with camera:
        rep.modify.pose(
            position=rep.distribution.uniform(
                (-15.0, -15.0,  2.0),
                ( 15.0,  15.0, 12.0)
            ),
            look_at="/World/shahed_unwrapped"
        )
    
    rep.randomizer.randomize_camera_sensor_noise()
    #rep.randomizer.randomize_shahed_flight()
    rep.randomizer.randomize_environment()

# ------------------------------------------------
# 7. WRITER SETUP & EXECUTION
# ------------------------------------------------
render_product = rep.create.render_product(camera, (1920, 1080))

writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir="/home/haipy/Desktop/isaac/target_drone_noise_gaussian",
    rgb=True,
    semantic_segmentation=True,
    bounding_box_2d_tight=True
)
writer.attach([render_product])

rep.orchestrator.run()
print(f"=== Generation triggered: {NUM_FRAMES} frames with sensor noise & motion blur targeting /World/shahed_unwrapped ===")
