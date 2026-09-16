# --- START OF FILE globals.py ---

import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

# Canonical media extensions, defined once so the file dialogs and
# has_image_extension never drift. GIF is intentionally excluded: OpenCV's
# cv2.imread/imwrite (the only image I/O this app uses) cannot decode or
# encode GIF on 4.10 or 4.11, so offering it would silently fail. WEBP works
# via the libwebp bundled with opencv-python.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mkv")

# Face Mapping Data
source_target_map: List[Dict[str, Any]] = [] # Stores detailed map for image/video processing
simple_map: Dict[str, Any] = {}             # Stores simplified map (embeddings/faces) for live/simple mode

# Paths
source_path: str | None = None
target_path: str | None = None
output_path: str | None = None
# Saved UI conveniences. ``source_path`` may point to a generated still from
# a video, while these remember the media the user actually selected.
last_source_media_path: str | None = None
last_source_cache_path: str | None = None
# A multi-frame embedding is valid only for the cached frame selected from a
# source video.  It must never survive a switch to an ordinary source photo.
source_profile_embedding: Any | None = None
source_profile_from_video: bool = False
last_camera_name: str | None = None
quality_profile: str = "Quality"

# Processing Options
frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False         # Process all detected faces with default source
map_faces: bool = False          # Use source_target_map or simple_map for specific swaps
poisson_blend: bool = False      # Enable Poisson Blending for smoother face swaps
color_correction: bool = False   # Enable color correction (implementation specific)
nsfw_filter: bool = False

# Video Output Options
video_encoder: str | None = None
video_quality: int | None = None # Typically a CRF value or bitrate

# Live Mode Options
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None # Placeholder for UI element if needed
webcam_preview_running: bool = False
show_fps: bool = False
# Send the processed live frame to Unity Video Capture for OBS.
virtual_camera: bool = False
# Replace the live camera room with the selected still image.  The feature is
# deliberately separate from the virtual-camera switch: it affects the live
# preview, the clean output window and Unity Video Capture alike.
virtual_background: bool = True
virtual_background_path: str | None = os.path.join(
    os.path.dirname(ROOT_DIR), "assets", "backgrounds", "presidential-office.webp"
)
# Smart FPS changes only how often the virtual-background matte is refreshed.
# It never silently disables the user's selected background or face model.
smart_fps: bool = True
smart_fps_minimum: int = 18
virtual_background_interval: int = 2

# System Configuration
max_memory: int | None = None        # Memory limit in GB? (Needs clarification)
execution_providers: List[str] = []  # e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads: int | None = None # Number of threads for CPU execution
headless: bool | None = None         # Run without UI?
log_level: str = "error"             # Logging level (e.g., 'debug', 'info', 'warning', 'error')

# Face Processor UI Toggles (Example)
fp_ui: Dict[str, bool] = {"face_enhancer": False, "face_enhancer_gpen256": False, "face_enhancer_gpen512": False, "face_enhancer_gpen1024": False}

# Face Swapper Specific Options
face_swapper_enabled: bool = True # General toggle for the swapper processor
opacity: float = 1.0              # Blend factor for the swapped face (0.0-1.0)
sharpness: float = 0.0            # Sharpness enhancement for swapped face (0.0-1.0+)

# Final colour controls are applied to the outgoing live frame.  Neutral
# defaults take a zero-cost fast path, so they do not reduce FPS unless used.
brightness: float = 0.0           # -100..100, applied after face processing
contrast: float = 1.0             # 0.50..1.50
saturation: float = 1.0           # 0.00..2.00
gamma: float = 1.0                # 0.50..1.80
digital_vibrance: float = 0.0     # -100..100, adaptive saturation like NVIDIA Digital Vibrance
texture_preservation: float = 0.0   # 0..100, optional post-GPEN facial microtexture
# TensorRT GPEN is faster but changes identity detail in the complete live
# face-swap pipeline. Keep the visually validated ONNX/CUDA implementation as
# the default; the optional engine remains available only for future testing.
use_gpen256_tensorrt: bool = False

# Mouth Mask Options
mouth_mask: bool = False           # Enable mouth area masking/pasting
show_mouth_mask_box: bool = False  # Visualize the mouth mask area (for debugging)
mask_feather_ratio: int = 12       # Denominator for feathering calculation (higher = smaller feather)
mask_down_size: float = 0.1        # Expansion factor for lower lip mask (relative)
mask_size: float = 1.0             # Expansion factor for upper lip mask (relative)
mouth_mask_size: float = 0.0       # Mouth mask size (0-100; 0=off, 100=mouth to chin)

# Extend the face-swap alpha to cover the hairline and lower jaw/upper neck.
# This is a soft geometric blend, not a separate hair-generation model.
full_head_coverage: bool = True
# Live mask geometry selected in the UI. It is independent of the identity
# model and can be changed while LIVE is running.
mask_profile: str = "Chin"
# Experimental landmark-shaped alpha. Disabled until it is validated across
# more webcam poses; the affine-locked soft mask is the stable live default.
adaptive_mask: bool = True
# Preview-only A/B comparison and optional live performance readout.  Neither
# option changes the frame sent to the virtual camera.
preview_original: bool = False
show_diagnostics: bool = False
# Experimental TensorRT LivePortrait mode.  In this mode source_path is a
# source video and the normal 128px face swapper is bypassed.
full_head_mode: bool = False

# The trained DeepFaceLab identity is optional and deliberately off by
# default.  It replaces only the detected face crop; the normal INSwapper
# engine remains available unchanged.
trained_identity_mode: bool = False
trained_identity_morph: float = 1.0

# --- START: Added for Frame Interpolation ---
enable_interpolation: bool = True # Toggle temporal smoothing
interpolation_weight: float = 0  # Blend weight for current frame (0.0-1.0). Lower=smoother.
# --- END: Added for Frame Interpolation ---

# --- END OF FILE globals.py ---

import threading
dml_lock = threading.Lock()
