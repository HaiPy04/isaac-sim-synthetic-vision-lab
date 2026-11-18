import omni.replicator.core as rep
import random
import os
import glob

print("=== Starting Shahed Target Drone Synthetic Data Generation ===")

# ------------------------------------------------
# 0. SETUP PATHS & LOAD FILES
# ------------------------------------------------
HDRI_FOLDER = "/home/haipy/Desktop/isaac/background"
hdri_files = sorted(glob.glob(os.path.join(HDRI_FOLDER, "*")))

if len(hdri_files) == 0:
    print(f"WARNING: No HDRIs found in {HDRI_FOLDER}. Using fallback.")
    hdri_files = ["/home/haipy/Desktop/isaac/bambanani_sunset_4k.exr"]

print(f"Loaded {len(hdri_files)} HDRI background skies.")

# ------------------------------------------------
# 1. REFERENCE EXISTING SHAHED DRONE
# ------------------------------------------------
# Get the already existing unwrapped Shahed prim in the stage
shahed_prim = rep.get.prims(path_pattern="/World/shahed_unwrapped")

# ------------------------------------------------
# 2. HDRI DOME LIGHT (Sky & Sun Lighting)
# ------------------------------------------------
dome_light = rep.create.light(
    light_type="Dome",
    texture=hdri_files[0],
    intensity=1000
)

# ------------------------------------------------
# 3. INTERCEPTOR CAMERA
# ------------------------------------------------
camera = rep.create.camera()

with camera:
    rep.modify.attribute("focalLength", 24.0)
    rep.modify.attribute("fStop", 0.0)

# ------------------------------------------------
# 4. RANDOMIZERS
# ------------------------------------------------
@rep.randomizer.register
def randomize_shahed_flight():
    with shahed_prim:
        # Randomize target position (simulate mid-air flight offsets)
        rep.modify.pose(
            position=rep.distribution.uniform((-1.5, -1.5, -1.0), (1.5, 1.5, 1.0)),
            # Randomize pitch, roll, and yaw representing flight attitude
            rotation=rep.distribution.uniform((-20.0, -180.0, -20.0), (20.0, 180.0, 20.0)),
            #scale=(1.0, 1.0, 1.0)
        )

@rep.randomizer.register
def randomize_environment():
    with dome_light:
        rep.modify.attribute(
            "inputs:texture:file",
            rep.distribution.choice(hdri_files)
        )
        # Randomize sun angle around the vertical axis
        rep.modify.pose(rotation=rep.distribution.uniform((0, 0, 0), (0, 360, 0)))
        rep.modify.attribute("inputs:intensity", rep.distribution.uniform(500, 2500))

# ------------------------------------------------
# 5. TRIGGER GRAPH (Frame Loop)
# ------------------------------------------------
NUM_FRAMES = 10

with rep.trigger.on_frame(num_frames=NUM_FRAMES):
    # Orbit/Pursuit camera randomizer looking directly at the target
    with camera:
        rep.modify.pose(
            position=rep.distribution.uniform(
                (-15.0, -15.0,  2.0),
                ( 15.0,  15.0, 12.0)
            ),
            look_at="/World/shahed_unwrapped"
        )
    
    # Apply dynamic flight and atmospheric lighting variations
    #rep.randomizer.randomize_shahed_flight()
    rep.randomizer.randomize_environment()

# ------------------------------------------------
# 6. WRITER SETUP & EXECUTION
# ------------------------------------------------
# Set output resolution (4K: 3840x2160 or 1080p: 1920x1080)
render_product = rep.create.render_product(camera, (1920, 1080))

writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir="/home/haipy/Desktop/isaac/target_drone_dataset_test",
    rgb=True,
    semantic_segmentation=True,
)
writer.attach([render_product])

rep.orchestrator.run()
print(f"=== Generation triggered: {NUM_FRAMES} frames targeting /World/shahed_unwrapped ===")
