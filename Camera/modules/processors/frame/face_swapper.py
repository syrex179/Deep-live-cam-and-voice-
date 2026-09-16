from typing import Any, List, Optional, Tuple

# --- TensorRT/CUDA DLL preload for Windows ---
# Must happen before ONNX Runtime/InsightFace creates any sessions.
_TRY_ROOT = os.path.dirname(os.path.abspath(__file__)) if "os" in globals() else None
import copy
import os as _os_boot
try:
    _ROOT = _os_boot.path.dirname(_os_boot.path.dirname(_os_boot.path.dirname(_os_boot.path.dirname(_os_boot.path.abspath(__file__)))))
    _SITE = _os_boot.path.join(_ROOT, "venv", "Lib", "site-packages")
    _DLL_DIRS = [
        _os_boot.path.join(_SITE, "tensorrt_libs"),
        _os_boot.path.join(_SITE, "nvidia", "cublas", "bin"),
        _os_boot.path.join(_SITE, "nvidia", "cuda_runtime", "bin"),
        _os_boot.path.join(_SITE, "nvidia", "cudnn", "bin"),
        _os_boot.path.join(_SITE, "nvidia", "cuda_nvrtc", "bin"),
        _os_boot.path.join(_SITE, "nvidia", "cufft", "bin"),
        _os_boot.path.join(_SITE, "nvidia", "curand", "bin"),
        _os_boot.path.join(_SITE, "nvidia", "nvjitlink", "bin"),
    ]
    for _d in _DLL_DIRS:
        if _os_boot.path.isdir(_d):
            try:
                _os_boot.add_dll_directory(_d)
            except Exception:
                pass
            _os_boot.environ["PATH"] = _d + _os_boot.pathsep + _os_boot.environ.get("PATH", "")
except Exception:
    pass
import cv2
import insightface
import logging
import threading
import numpy as np
import platform
import modules.globals
import modules.processors.frame.core
from modules import imread_unicode, imwrite_unicode
# Lazy wrapper to avoid a circular import: core -> ui -> face_swapper -> core.
def update_status(*args, **kwargs):
    from modules.core import update_status as _core_update_status
    return _core_update_status(*args, **kwargs)
from modules.face_analyser import get_one_face, get_many_faces, default_source_face
from modules.typing import Face, Frame
from modules.utilities import (
    is_image,
    is_video,
)
from modules.cluster_analysis import find_closest_centroid
from modules.gpu_processing import gpu_gaussian_blur, gpu_sharpen, gpu_add_weighted, gpu_resize
from modules.platform_info import OPENVINO_PROVIDER_CONFIG
import os
from collections import deque
import time

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-SWAPPER"

# --- START: Added for Interpolation ---
PREVIOUS_FRAME_RESULT = None # Stores the final processed frame from the previous step
# --- END: Added for Interpolation ---

# --- Poisson blend (ported from deep-live-cam-gumroad-edition) ---
# Root-cause fix for the "wobble": the blend mask is NOT built from the
# independently-detected 106-pt landmarks (they jitter sub-pixel every frame
# and seamlessClone is hyper-sensitive to its mask boundary). Instead it is
# derived from the swap's OWN affine transform (M) + the swapped pixels
# (bgr_fake), so the mask is locked exactly to where the swapped face was
# placed — no independent jitter source, no EMA, no lag. The mask is cached
# when the face is nearly still so an identical array is reused (zero wobble).
_ELLIPTICAL_MASK_CACHE: dict = {}
_poisson_cached_mask: Optional[np.ndarray] = None
_poisson_cached_key: Optional[tuple] = None


def _mask_geometry(size: int) -> tuple[tuple[int, int], tuple[int, int], Optional[float]]:
    """Return the selected live mask geometry in aligned-face coordinates."""
    profile = str(getattr(modules.globals, "mask_profile", "Chin"))
    if profile == "Tight":
        # A conventional face-only oval: it deliberately fades above the
        # lower jaw and is useful when facial-hair transfer looks too wide.
        return ((size // 2, int(round(size * .47))),
                (int(size * .44), int(size * .47)), .72)
    if profile == "Full":
        return ((size // 2, int(round(size * .50))),
                (int(size * .495), int(round(size * .60))), .98)
    # Chin: covers the jaw and beard while retaining a slightly wider soft
    # transition than the Full preset.
    return ((size // 2, int(round(size * .50))),
            (int(size * .495), int(round(size * .60))), .92)


def _create_elliptical_mask(size: Tuple[int, int]) -> np.ndarray:
    """Fixed, heavily-blurred elliptical mask in aligned-face space.

    Geometry-based (not content-adaptive) and cached by size — identical
    every frame for the same model input size, so it contributes no jitter.
    """
    global _ELLIPTICAL_MASK_CACHE
    full_head = bool(getattr(modules.globals, "full_head_coverage", True))
    profile = str(getattr(modules.globals, "mask_profile", "Chin"))
    cache_key = (size, full_head, profile)
    if cache_key in _ELLIPTICAL_MASK_CACHE:
        return _ELLIPTICAL_MASK_CACHE[cache_key]
    h, w = size
    if full_head:
        # The aligned crop contains a little of the hairline at its top and
        # lower jaw/upper neck at its bottom. Shift and extend the oval so
        # those pixels participate, while preserving a soft edge instead of
        # turning the result into a visible rectangular source-photo paste.
        # The default 128px swap crop contains useful generated chin/beard
        # pixels below the old oval.  Include that lower part of the model
        # output, rather than painting a static source texture over the live
        # camera beard.  The edge is still blurred and remains above chest.
        center, axes, lower_fade_start = _mask_geometry(min(h, w))
    else:
        center = (w // 2, h // 2)
        axes = (int(w * 0.47), int(h * 0.49))
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    if full_head:
        # Keep the generated beard and the small upper-neck zone.  Starting
        # this fade at 80% cut the mask across a long beard: the target's
        # lower beard/neck showed through instead of being replaced.  The
        # fade now occupies only the final 2% of the aligned crop, where it
        # quietly joins the live body without a horizontal seam. This keeps
        # the complete generated beard visible during the mask-only test.
        if lower_fade_start is not None:
            start = int(round(h * lower_fade_start))
            t = np.linspace(0.0, 1.0, h - start, dtype=np.float32)
            mask[start:, :] *= (1.0 - (t * t * (3.0 - 2.0 * t)))[:, None]
    if h * w < 65536:
        mask = cv2.GaussianBlur(mask, (31, 31), 12)
    else:
        mask = gpu_gaussian_blur(mask, (31, 31), 12)
    _ELLIPTICAL_MASK_CACHE[cache_key] = mask
    return mask


def _apply_fast_affine_blend(
    swapped_frame: Frame,
    original_frame: Frame,
    affine_matrix: np.ndarray,
    bgr_fake: np.ndarray,
) -> Optional[Frame]:
    """Fast affine-locked face blend for live mode.

    This replaces the older erosion + Gaussian + multiple ROI operations.
    The cached aligned-face alpha mask is warped once into the current ROI,
    then blended with OpenCV. It stays frame-local and never reprojects an
    old frame, so quick head motion cannot make the face slide.
    """
    try:
        h, w = swapped_frame.shape[:2]
        fh, fw = bgr_fake.shape[:2]
        inv = cv2.invertAffineTransform(affine_matrix)
        corners = np.array(
            [[0, 0], [fw - 1, 0], [fw - 1, fh - 1], [0, fh - 1]],
            dtype=np.float32,
        )
        transformed = corners @ inv.T
        px1 = max(0, int(np.floor(transformed[:, 0].min())))
        py1 = max(0, int(np.floor(transformed[:, 1].min())))
        px2 = min(w, int(np.ceil(transformed[:, 0].max())) + 1)
        py2 = min(h, int(np.ceil(transformed[:, 1].max())) + 1)
        rw, rh = px2 - px1, py2 - py1
        if rw <= 8 or rh <= 8:
            return None

        roi_aff = inv.copy()
        roi_aff[0, 2] -= px1
        roi_aff[1, 2] -= py1

        # _create_elliptical_mask is cached by model input size. For the
        # 128x128 INSwapper output this is tiny; only the warp changes per frame.
        aligned_alpha = _create_elliptical_mask((fh, fw))
        alpha = cv2.warpAffine(
            aligned_alpha, roi_aff, (rw, rh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        alpha8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint16)
        inv8 = 255 - alpha8
        swap_roi = swapped_frame[py1:py2, px1:px2].astype(np.uint16)
        original_roi = original_frame[py1:py2, px1:px2].astype(np.uint16)
        blended = (
            swap_roi * alpha8[:, :, None] +
            original_roi * inv8[:, :, None] +
            127
        ) // 255
        swapped_frame[py1:py2, px1:px2] = blended.astype(np.uint8)
        return swapped_frame
    except Exception:
        return None


def _preserve_target_eye_detail(
    swapped_frame: Frame,
    original_frame: Frame,
    target_face: Face,
) -> Frame:
    """Restore the current camera eye regions after the face swap.

    INSwapper can retain the source portrait's forward-looking gaze even when
    the target turns or looks sideways.  The detector's current-frame 5-point
    landmarks let us place two small, feathered eye masks without temporal
    caching, so gaze and blinking follow the camera naturally.
    """
    try:
        kps = getattr(target_face, "_syrex_original_kps", None)
        if kps is None:
            kps = getattr(target_face, "kps", None)
        if kps is None or len(kps) < 2:
            return swapped_frame
        kps = np.asarray(kps, dtype=np.float32)
        left_eye, right_eye = kps[0], kps[1]
        eye_distance = float(np.linalg.norm(right_eye - left_eye))
        if not np.isfinite(eye_distance) or eye_distance < 12.0:
            return swapped_frame

        # Covers the eye, eyelids and a small amount of surrounding skin.
        radius_x = max(5, int(round(eye_distance * 0.24)))
        radius_y = max(4, int(round(eye_distance * 0.105)))
        angle = float(np.degrees(np.arctan2(
            right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
        )))
        angle_rad = np.radians(angle)
        extent_x = int(np.ceil(np.hypot(radius_x * np.cos(angle_rad), radius_y * np.sin(angle_rad))))
        extent_y = int(np.ceil(np.hypot(radius_x * np.sin(angle_rad), radius_y * np.cos(angle_rad))))
        feather = max(3, int(round(eye_distance * 0.04)))
        h, w = swapped_frame.shape[:2]

        for eye in (left_eye, right_eye):
            cx, cy = int(round(float(eye[0]))), int(round(float(eye[1])))
            x0 = max(0, cx - extent_x - feather)
            y0 = max(0, cy - extent_y - feather)
            x1 = min(w, cx + extent_x + feather + 1)
            y1 = min(h, cy + extent_y + feather + 1)
            if x1 <= x0 or y1 <= y0:
                continue

            mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.ellipse(
                mask, (cx - x0, cy - y0), (radius_x, radius_y), angle,
                0, 360, 255, -1,
            )
            kernel = 2 * feather + 1
            alpha = cv2.GaussianBlur(mask, (kernel, kernel), 0)
            # Eye Protection slider: 0 = off, 100 = full target-eye restore.
            eye_strength = max(0.0, min(100.0, float(getattr(modules.globals, "eye_protection", 85.0))))
            alpha = ((alpha.astype(np.uint16) * eye_strength) // 100).astype(np.uint8)
            alpha_3ch = cv2.merge((alpha, alpha, alpha))
            blended = cv2.add(
                cv2.multiply(original_frame[y0:y1, x0:x1], alpha_3ch, scale=1.0 / 255.0),
                cv2.multiply(swapped_frame[y0:y1, x0:x1], cv2.bitwise_not(alpha_3ch), scale=1.0 / 255.0),
            )
            swapped_frame[y0:y1, x0:x1] = blended
    except Exception:
        pass
    return swapped_frame


def _apply_poisson_blend(
    swapped_frame: Frame, original_frame: Frame,
    target_face: Face, affine_matrix: np.ndarray = None,
    bgr_fake: np.ndarray = None,
) -> Frame:
    """Live-safe fast blend used by the Poisson Blend toggle.

    The old seamlessClone fallback could consume 10–20 ms on 1280x960 live
    frames. Live processing now uses only the affine-locked fast path. If the
    user selects Performance mode, this optional blend is skipped altogether
    to prioritize FPS.
    """
    try:
        if getattr(modules.globals, "performance_mode", "Balanced") == "Performance":
            return swapped_frame
        if affine_matrix is not None and bgr_fake is not None:
            fast_result = _apply_fast_affine_blend(
                swapped_frame, original_frame, affine_matrix, bgr_fake,
            )
            if fast_result is not None:
                return fast_result
    except Exception:
        pass
    return swapped_frame


# --- START: Mac M1-M5 Optimizations ---
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.machine() == 'arm64'
FRAME_CACHE = deque(maxlen=3)  # Cache for frame reuse
FACE_DETECTION_CACHE = {}  # Cache face detections
LAST_DETECTION_TIME = 0
DETECTION_INTERVAL = 0.033  # ~30 FPS detection rate for live mode
FRAME_SKIP_COUNTER = 0
ADAPTIVE_QUALITY = True
# --- END: Mac M1-M5 Optimizations ---

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)

def pre_check() -> bool:
    # Use models_dir instead of abs_dir to save to the correct location
    download_directory_path = models_dir

    # Make sure the models directory exists, catch permission errors if they occur
    try:
        os.makedirs(download_directory_path, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create directory {download_directory_path} due to permission error: {e}")
        return False

    from modules.model_downloader import ensure_any

    variants = ["inswapper_128.onnx", "inswapper_128_fp16.onnx"]
    if _HAS_TORCH_CUDA:
        variants.reverse()
    if ensure_any(variants) is None:
        update_status(
            "Could not obtain the inswapper model. Place inswapper_128.onnx in "
            "the models folder manually or check your internet connection.",
            NAME,
        )
        return False
    return True


def pre_start() -> bool:
    # Check for either model variant
    fp16_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
    fp32_path = os.path.join(models_dir, "inswapper_128.onnx")
    if not os.path.exists(fp16_path) and not os.path.exists(fp32_path):
        update_status(f"Model not found in {models_dir}. Please download inswapper_128.onnx.", NAME)
        return False

    # Try to get the face swapper to ensure it loads correctly
    if get_face_swapper() is None:
        # Error message already printed within get_face_swapper
        return False

    return True


_tensorrt_session = None
_tensorrt_enabled = False

def _init_tensorrt_session(model_path: str, swapper) -> bool:
    """Replace INSwapper's inference session with TensorRT FP16.

    Keeps the same INSwapper object, input/output names and paste-back path.
    TensorRT is used only for the 128x128 swap model; if it fails, caller can
    fall back to the existing CUDA-graph path.
    """
    global _tensorrt_session, _tensorrt_enabled
    try:
        import onnxruntime as ort
        trt_cache = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "trt_cache_inswapper")
        os.makedirs(trt_cache, exist_ok=True)
        providers = [
            ("TensorrtExecutionProvider", {
                "device_id": 0,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": trt_cache,
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": trt_cache,
            }),
            ("CUDAExecutionProvider", {"device_id": 0}),
            ("CPUExecutionProvider", {}),
        ]
        sess = ort.InferenceSession(model_path, providers=providers)
        actual = sess.get_providers()
        if not actual or actual[0] != "TensorrtExecutionProvider":
            print(f"[{NAME}] TensorRT session did not load as primary EP: {actual}")
            return False

        # INSwapper.get() expects the same session API and the same input/output
        # names. The model shape is static: target [1,3,128,128], source [1,512].
        swapper.session = sess
        _tensorrt_session = sess
        _tensorrt_enabled = True
        print(f"[{NAME}] TensorRT FP16 session initialized | providers={actual}")
        print(f"[{NAME}] TensorRT cache={trt_cache}")
        return True
    except Exception as e:
        print(f"[{NAME}] TensorRT FP16 init failed: {e}")
        _tensorrt_enabled = False
        return False

def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            # Prefer FP16 on GPUs with Tensor Cores (Turing+) — half the
            # memory bandwidth, faster inference.  Fall back to FP32 for
            # older GPUs (e.g. GTX 16xx) where FP16 can produce NaN.
            fp32_path = os.path.join(models_dir, "inswapper_128.onnx")
            fp16_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
            use_fp16 = _HAS_TORCH_CUDA and os.path.exists(fp16_path)
            if use_fp16:
                model_path = fp16_path
            elif os.path.exists(fp32_path):
                model_path = fp32_path
            else:
                if not pre_check():
                    return None
                model_path = fp16_path if os.path.exists(fp16_path) else fp32_path
                if not os.path.exists(model_path):
                    update_status(f"No inswapper model found in {models_dir}.", NAME)
                    return None
            # On Apple Silicon, rewrite Pad(reflect) → Slice+Concat so
            # CoreML can run the entire model in a single partition on
            # the Neural Engine instead of bouncing between CPU and ANE.
            if IS_APPLE_SILICON:
                from modules.onnx_optimize import optimize_for_coreml
                model_path = optimize_for_coreml(model_path)

            update_status(f"Loading face swapper model from: {model_path}", NAME)
            try:
                providers_config = []
                for p in modules.globals.execution_providers:
                    if p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
                        # Enhanced CoreML configuration for M1-M5
                        providers_config.append((
                            "CoreMLExecutionProvider",
                            {
                                "ModelFormat": "MLProgram",
                                "MLComputeUnits": "ALL",  # Use Neural Engine + GPU + CPU
                                "SpecializationStrategy": "FastPrediction",
                                "AllowLowPrecisionAccumulationOnGPU": 1,
                                "EnableOnSubgraphs": 1,
                            }
                        ))
                    elif p == "CUDAExecutionProvider":
                        # Use bare provider — ONNX Runtime defaults are
                        # fastest on modern GPUs (Blackwell/sm_120).
                        providers_config.append(p)
                    elif p == "OpenVINOExecutionProvider":
                        providers_config.append(OPENVINO_PROVIDER_CONFIG)
                    else:
                        providers_config.append(p)
                FACE_SWAPPER = insightface.model_zoo.get_model(
                    model_path,
                    providers=providers_config,
                )
                # Prefer TensorRT FP16 for the swap model. If TRT fails, keep
                # the proven CUDA-graph path as a safe fallback.
                _trt_ok = False
                if _HAS_TORCH_CUDA and platform.system() == "Windows":
                    _trt_ok = _init_tensorrt_session(model_path, FACE_SWAPPER)
                if not _trt_ok and _HAS_TORCH_CUDA and any(
                    p == "CUDAExecutionProvider" or
                    (isinstance(p, tuple) and p[0] == "CUDAExecutionProvider")
                    for p in providers_config
                ):
                    _init_cuda_graph_session(model_path, FACE_SWAPPER)
                update_status("Face swapper model loaded successfully.", NAME)
            except Exception as e:
                update_status(f"Error loading face swapper model: {e}", NAME)
                FACE_SWAPPER = None
                return None
    return FACE_SWAPPER


_HAS_TORCH_CUDA = False
try:
    import torch
    if torch.cuda.is_available():
        _HAS_TORCH_CUDA = True
except ImportError:
    pass

# Cache for paste-back. The feather amount is part of the key so the
# Edge Softness slider changes the real alpha mask.
_paste_cache = {
    'soft_alpha': None,
    'alpha_key': None,
}

# Bounded state for optional stabilization. Large motion snaps immediately,
# so this is not a stale-frame reprojection cache.
_STAB_STATE = {}


def _get_soft_alpha(size: int) -> np.ndarray:
    """Return a feathered elliptical mask in aligned-face space."""
    feather = max(0.0, min(100.0, float(getattr(modules.globals, 'mask_feather', 50.0))))
    full_head = bool(getattr(modules.globals, "full_head_coverage", True))
    profile = str(getattr(modules.globals, "mask_profile", "Chin"))
    key = (int(size), round(feather, 2), full_head, profile)
    if _paste_cache['alpha_key'] == key and _paste_cache['soft_alpha'] is not None:
        return _paste_cache['soft_alpha']

    if full_head:
        center, axes, lower_fade_start = _mask_geometry(size)
    else:
        center = (size // 2, size // 2)
        axes = (int(size * 0.47), int(size * 0.49))
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    if full_head:
        if lower_fade_start is not None:
            start = int(round(size * lower_fade_start))
            t = np.linspace(0.0, 1.0, size - start, dtype=np.float32)
            lower_fade = 1.0 - (t * t * (3.0 - 2.0 * t))
            mask[start:, :] = (mask[start:, :].astype(np.float32) * lower_fade[:, None]).astype(np.uint8)

    sigma = 2.0 + 20.0 * (feather / 100.0)
    kernel = max(3, int(round(sigma * 5.0)) | 1)
    mask = cv2.GaussianBlur(mask, (kernel, kernel), sigma)
    _paste_cache['soft_alpha'] = mask
    _paste_cache['alpha_key'] = key
    return mask


def _transform_points(points: np.ndarray, center: np.ndarray,
                      scale: float, angle_deg: float,
                      dx: float, dy: float) -> np.ndarray:
    if points is None:
        return points
    pts = np.asarray(points, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[1] != 2 or not np.all(np.isfinite(pts)):
        return points
    theta = np.deg2rad(float(angle_deg))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return ((pts - center) @ rot.T) * float(scale) + center + np.array([dx, dy], dtype=np.float32)


def prepare_target_face(face: Face, frame_shape=None, track_key=0) -> Face:
    """Apply only live-face stabilization.

    X/Y/Scale/Rotation are applied directly to the output affine in
    ``swap_face`` so these controls cannot be ignored by the detector's
    keypoint representation.
    """
    if face is None:
        return face
    stabil = max(0.0, min(90.0, float(getattr(modules.globals, 'face_stabilization', 0.0))))
    if stabil == 0.0:
        return face
    try:
        f = copy.copy(face)
        setattr(f, '_syrex_geometry_prepared', True)
        if getattr(face, 'kps', None) is not None:
            f._syrex_original_kps = np.asarray(face.kps, dtype=np.float32).copy()
        if getattr(face, 'landmark_2d_106', None) is not None:
            f._syrex_original_landmark_2d_106 = np.asarray(face.landmark_2d_106, dtype=np.float32).copy()
        bbox = np.asarray(getattr(face, 'bbox', None), dtype=np.float32).reshape(-1)
        if bbox.size < 4 or not np.all(np.isfinite(bbox)):
            return face
        x1,y1,x2,y2=map(float,bbox[:4])
        bw,bh=max(1.0,x2-x1),max(1.0,y2-y1)
        center=np.array([(x1+x2)*0.5,(y1+y2)*0.5],dtype=np.float32)
        prev=_STAB_STATE.get(track_key)
        if prev is not None:
            prev_center=np.asarray(prev,dtype=np.float32)
            delta=float(np.linalg.norm(center-prev_center))
            if delta <= max(10.0,0.18*max(bw,bh)):
                # Keep stabilization as a small landmark damper, never a
                # delayed head position. High values previously made the
                # swapped face visibly lag behind fast real head movement.
                current_weight=1.0-0.55*(stabil/90.0)
                center=prev_center*(1.0-current_weight)+center*current_weight
        _STAB_STATE[track_key]=center.copy()
        dx=float(center[0]-(x1+x2)*0.5)
        dy=float(center[1]-(y1+y2)*0.5)
        for attr in ('kps','landmark_2d_106'):
            pts=getattr(face,attr,None)
            if pts is not None:
                setattr(f,attr,(np.asarray(pts,dtype=np.float32)+np.array([dx,dy],dtype=np.float32)).astype(np.float32))
        f.bbox=np.array([x1+dx,y1+dy,x2+dx,y2+dy],dtype=np.float32)
        return f
    except Exception:
        return face

def clear_face_stabilization_state() -> None:
    _STAB_STATE.clear()


# CUDA graph swap session cache
_cuda_graph_session = {
    'session': None,
    'io_binding': None,
    'ort_input': None,
    'ort_latent': None,
    'recorded': False,
}
# Serializes CUDA-graph replay. The io_binding + ort_input/ort_latent are
# shared across threads and run_with_iobinding mutates GPU-side buffers;
# concurrent calls would produce wrong output.
_cuda_graph_lock = threading.Lock()


class _CudaGraphSessionAdapter:
    """Drop-in wrapper around an ONNX Runtime session.

    Routes ``.run()`` through CUDA graph replay when a recorded graph is
    available, and transparently proxies every other attribute to the
    underlying session so insightface's INSwapper sees an unchanged API.
    """

    def __init__(self, underlying):
        # Use object.__setattr__ to bypass our own __setattr__.
        object.__setattr__(self, "_underlying", underlying)

    def run(self, output_names, input_dict, **kwargs):
        if _cuda_graph_session['recorded']:
            try:
                keys = list(input_dict.keys())
                blob = input_dict[keys[0]]
                latent = input_dict[keys[1]]
                return [_cuda_graph_swap_inference(blob, latent)]
            except Exception:
                pass
        return self._underlying.run(output_names, input_dict, **kwargs)

    def __getattr__(self, name):
        return getattr(self._underlying, name)

    def __setattr__(self, name, value):
        setattr(self._underlying, name, value)


def _init_cuda_graph_session(model_path: str, swapper):
    """Create a CUDA-graph-enabled ONNX session for the swap model.

    CUDA graphs record the GPU kernel launch sequence once, then replay it
    with near-zero CPU overhead on subsequent runs.  Requires static input
    shapes (inswapper is always 1x3x128x128 + 1x512).
    """
    import onnxruntime as ort
    try:
        providers = [('CUDAExecutionProvider', {'enable_cuda_graph': '1'})]
        sess = ort.InferenceSession(model_path, providers=providers)

        # Pre-allocate GPU buffers with correct shapes
        inp_shape = (1, 3, swapper.input_size[1], swapper.input_size[0])
        latent_shape = (1, 512)
        dummy_inp = np.zeros(inp_shape, dtype=np.float32)
        dummy_lat = np.zeros(latent_shape, dtype=np.float32)

        ort_input = ort.OrtValue.ortvalue_from_numpy(dummy_inp, 'cuda', 0)
        ort_latent = ort.OrtValue.ortvalue_from_numpy(dummy_lat, 'cuda', 0)

        io = sess.io_binding()
        io.bind_ortvalue_input(swapper.input_names[0], ort_input)
        io.bind_ortvalue_input(swapper.input_names[1], ort_latent)
        io.bind_output(swapper.output_names[0], 'cuda', 0)

        # First run records the CUDA graph
        sess.run_with_iobinding(io)

        _cuda_graph_session['session'] = sess
        _cuda_graph_session['io_binding'] = io
        _cuda_graph_session['ort_input'] = ort_input
        _cuda_graph_session['ort_latent'] = ort_latent
        _cuda_graph_session['recorded'] = True

        # Wrap swapper.session in an adapter instead of rebinding
        # session.run. insightface's INSwapper.get() reads .run via the
        # session attribute, so either works; the adapter survives any
        # later attribute reads on the session and keeps the original
        # session object untouched.
        if not isinstance(swapper.session, _CudaGraphSessionAdapter):
            swapper.session = _CudaGraphSessionAdapter(swapper.session)

        import sys
        print(f"[{NAME}] CUDA graph session initialized (swap model)")
        sys.stdout.flush()
    except Exception as e:
        print(f"[{NAME}] CUDA graph init failed, using standard session: {e}")
        _cuda_graph_session['recorded'] = False


def _cuda_graph_swap_inference(blob: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """Run swap model via CUDA graph replay — minimal CPU overhead."""
    cg = _cuda_graph_session
    with _cuda_graph_lock:
        cg['ort_input'].update_inplace(blob)
        cg['ort_latent'].update_inplace(latent)
        cg['session'].run_with_iobinding(cg['io_binding'])
        return cg['io_binding'].get_outputs()[0].numpy()


def _apply_live_output_geometry(inv_M: np.ndarray, target_face: Face) -> np.ndarray:
    """Apply live Face X/Y/Scale/Rotation in final output-frame coordinates."""
    try:
        scale=max(0.60,min(1.40,float(getattr(modules.globals,'face_scale',1.0))))
        angle=max(-30.0,min(30.0,float(getattr(modules.globals,'face_rotation',0.0))))
        fx=max(-100.0,min(100.0,float(getattr(modules.globals,'face_offset_x',0.0))))
        fy=max(-100.0,min(100.0,float(getattr(modules.globals,'face_offset_y',0.0))))
        if scale==1.0 and abs(angle)<1e-6 and fx==0.0 and fy==0.0:
            return inv_M
        bbox=np.asarray(getattr(target_face,'bbox',None),dtype=np.float32).reshape(-1)
        if bbox.size<4 or not np.all(np.isfinite(bbox)):
            return inv_M
        x1,y1,x2,y2=map(float,bbox[:4])
        bw,bh=max(1.0,x2-x1),max(1.0,y2-y1)
        cx,cy=(x1+x2)*0.5,(y1+y2)*0.5
        dx=(fx/100.0)*bw*0.75
        dy=(fy/100.0)*bh*0.75
        th=np.deg2rad(angle)
        c,s=float(np.cos(th)),float(np.sin(th))
        T=np.array([[scale*c,-scale*s,cx+dx-scale*c*cx+scale*s*cy],
                    [scale*s,scale*c,cy+dy-scale*s*cx-scale*c*cy],
                    [0.0,0.0,1.0]],dtype=np.float32)
        H=np.vstack([np.asarray(inv_M,dtype=np.float32),[0.0,0.0,1.0]])
        return (T@H)[:2]
    except Exception:
        return inv_M

def _get_adaptive_face_alpha(size: int, target_face: Face, affine_matrix: np.ndarray) -> np.ndarray:
    """Refine the stable oval with the current face outline in crop space.

    The detector landmarks are transformed into the same 128px aligned space
    as the swap output.  That makes the mask follow the affine exactly, rather
    than creating an independently tracked full-frame edge that can wobble.
    """
    base = _get_soft_alpha(size)
    if not bool(getattr(modules.globals, "adaptive_mask", False)):
        return base
    try:
        landmarks = getattr(target_face, "landmark_2d_106", None)
        if landmarks is None:
            return base
        points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        if len(points) < 20 or not np.all(np.isfinite(points)):
            return base
        aligned = cv2.transform(points.reshape(1, -1, 2), affine_matrix)[0]
        valid = aligned[
            (aligned[:, 0] > -size * .25) & (aligned[:, 0] < size * 1.25)
            & (aligned[:, 1] > -size * .25) & (aligned[:, 1] < size * 1.25)
        ]
        if len(valid) < 20:
            return base
        left, top = valid.min(axis=0)
        right, bottom = valid.max(axis=0)
        width, height = right - left, bottom - top
        if width < size * .25 or height < size * .25:
            return base
        # The 106 landmarks end around the brows. Add a conservative forehead
        # cap before finding the hull; hair remains outside the mask.
        center_x = float((left + right) * .5)
        cap_y = max(0.0, float(top - height * .28))
        cap = np.array([
            [center_x - width * .30, cap_y],
            [center_x, max(0.0, cap_y - height * .04)],
            [center_x + width * .30, cap_y],
        ], dtype=np.float32)
        hull = cv2.convexHull(np.vstack([valid, cap]).astype(np.float32))
        shape = np.zeros((size, size), dtype=np.uint8)
        cv2.fillConvexPoly(shape, np.round(hull).astype(np.int32), 255)
        shape = cv2.GaussianBlur(shape, (9, 9), 2.0).astype(np.float32) / 255.0
        # Extend the lower contour just enough to keep the generated beard,
        # but do not leave a translucent oval outside the real face. That
        # translucent oval was the visible "square" on bare skin.
        chin = np.array([
            [center_x - width * .22, bottom + height * .06],
            [center_x, bottom + height * .14],
            [center_x + width * .22, bottom + height * .06],
        ], dtype=np.float32)
        hull = cv2.convexHull(np.vstack([valid, cap, chin]).astype(np.float32))
        shape = np.zeros((size, size), dtype=np.uint8)
        cv2.fillConvexPoly(shape, np.round(hull).astype(np.int32), 255)
        shape = cv2.GaussianBlur(shape, (11, 11), 2.6).astype(np.float32) / 255.0
        return base.astype(np.float32) * shape
    except Exception:
        return base


def _fast_paste_back(
    target_img: Frame, bgr_fake: np.ndarray, aimg: np.ndarray, M: np.ndarray,
    target_face: Optional[Face] = None, source_affine: Optional[np.ndarray] = None,
) -> Frame:
    """Paste bgr_fake back onto target_img via the inverse affine of M.

    Restricts work to the face bbox in output coordinates and warps a
    precomputed feathered alpha template per-frame instead of running a
    size-scaled erode+blur on the warped mask. Cost is O(crop_area) regardless
    of how much of the frame the face occupies.
    """
    h, w = target_img.shape[:2]
    face_h, face_w = aimg.shape[:2]
    # inswapper's aligned-face space is square (128x128). _get_soft_alpha
    # caches a single NxN template keyed by N, so fail loudly if that ever
    # stops being true rather than silently mis-warping the alpha mask.
    assert face_h == face_w, f"Expected square aligned face, got {face_h}x{face_w}"
    IM = cv2.invertAffineTransform(M)

    # Bbox in output coords from the affine corners of the aligned-face square.
    corners = np.array(
        [[0, 0], [face_w, 0], [face_w, face_h], [0, face_h]], dtype=np.float32
    )
    transformed = (IM[:, :2] @ corners.T).T + IM[:, 2]
    x1 = int(np.floor(transformed[:, 0].min()))
    x2 = int(np.ceil(transformed[:, 0].max()))
    y1 = int(np.floor(transformed[:, 1].min()))
    y2 = int(np.ceil(transformed[:, 1].max()))
    if x1 >= x2 or y1 >= y2:
        return target_img

    # Small interpolation margin only — the feather is baked into the template.
    pad = 2
    y1p, y2p = max(0, y1 - pad), min(h, y2 + pad + 1)
    x1p, x2p = max(0, x1 - pad), min(w, x2 + pad + 1)

    IM_crop = IM.copy()
    IM_crop[0, 2] -= x1p
    IM_crop[1, 2] -= y1p
    crop_w, crop_h = x2p - x1p, y2p - y1p

    # Build the adaptive shape in the unmodified detector affine. It is then
    # pasted through ``M`` together with the output, so Face X/Y/Scale/Rotate
    # remain perfectly in sync with the mask.
    soft_alpha = (
        _get_adaptive_face_alpha(face_h, target_face, source_affine)
        if target_face is not None and source_affine is not None
        else _get_soft_alpha(face_h)
    )
    bgr_fake_crop = cv2.warpAffine(bgr_fake, IM_crop, (crop_w, crop_h), borderMode=cv2.BORDER_REPLICATE)
    alpha_crop = cv2.warpAffine(soft_alpha, IM_crop, (crop_w, crop_h), borderValue=0)
    # The adaptive contour is calculated in float space, whereas the stable
    # oval is uint8.  The ROI mixer below intentionally uses the fast uint8
    # OpenCV path, so normalize both variants here before multiply/add.
    alpha_crop = np.clip(alpha_crop, 0, 255).astype(np.uint8, copy=False)

    target_crop = target_img[y1p:y2p, x1p:x2p]

    # IMPORTANT PERFORMANCE FIX:
    # Do the final small ROI blend on CPU with OpenCV. Uploading
    # bgr_fake_crop + target_crop to CUDA and downloading the blended
    # result adds a large synchronization/PCIe overhead for this
    # relatively small face ROI. The cv2 uint8 path avoids that.
    alpha_3c = cv2.merge([alpha_crop, alpha_crop, alpha_crop])
    inv_alpha = 255 - alpha_3c
    a_fake = cv2.multiply(bgr_fake_crop, alpha_3c, scale=1.0 / 255.0)
    a_tgt = cv2.multiply(target_crop, inv_alpha, scale=1.0 / 255.0)
    target_img[y1p:y2p, x1p:x2p] = cv2.add(a_fake, a_tgt)

    return target_img


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Optimized face swapping with better memory management and performance."""
    # A locally trained DFM model has its own identity decoder, so it does not
    # need a source image/embedding.  Keep this branch before INSwapper is
    # initialised: switching modes neither changes nor unloads the proven
    # ordinary face-swap engine.
    if getattr(modules.globals, "trained_identity_mode", False):
        from modules.processors.frame.dfm_identity import swap as swap_trained_identity
        return swap_trained_identity(temp_frame, target_face)

    face_swapper = get_face_swapper()
    if face_swapper is None:
        update_status("Face swapper model not loaded or failed to load. Skipping swap.", NAME)
        return temp_frame

    # Safety check for faces
    if source_face is None or target_face is None:
        return temp_frame
    if not hasattr(source_face, 'normed_embedding') or source_face.normed_embedding is None:
        return temp_frame

    # _fast_paste_back writes in-place on the GPU path.  Only copy when
    # mouth_mask or opacity < 1 need an unmodified original.
    opacity = getattr(modules.globals, "opacity", 1.0)
    opacity = max(0.0, min(1.0, opacity))
    mouth_mask_size = max(0.0, min(100.0, float(getattr(modules.globals, "mouth_mask_size", 0.0))))
    mouth_mask_enabled = bool(getattr(modules.globals, "mouth_mask", False)) or mouth_mask_size > 0.0
    poisson_blend_enabled = getattr(modules.globals, "poisson_blend", False)
    preserve_eye_detail = getattr(target_face, "kps", None) is not None
    # Poisson blend's seamlessClone needs the genuine pre-swap frame as its
    # destination. Without this, original_frame aliases temp_frame, which
    # _fast_paste_back mutates in place — so seamlessClone would blend the
    # swapped face onto the already-swapped frame (no visible effect).
    needs_original = (
        opacity < 1.0 or mouth_mask_enabled or poisson_blend_enabled
        or preserve_eye_detail
    )
    if needs_original:
        original_frame = temp_frame.copy()
    else:
        original_frame = temp_frame

    if temp_frame.dtype != np.uint8:
        temp_frame = np.clip(temp_frame, 0, 255).astype(np.uint8)

    try:
        _swap_t0 = time.perf_counter()
        if not temp_frame.flags['C_CONTIGUOUS']:
            temp_frame = np.ascontiguousarray(temp_frame)
        _prep_t = time.perf_counter()

        if not getattr(target_face, '_syrex_geometry_prepared', False):
            target_face = prepare_target_face(target_face, temp_frame.shape, track_key=0)

        # Use paste_back=False and our optimized paste-back
        _get_t0 = time.perf_counter()
        if any("DmlExecutionProvider" in p for p in modules.globals.execution_providers):
            with modules.globals.dml_lock:
                bgr_fake, M = face_swapper.get(
                    temp_frame, target_face, source_face, paste_back=False
                )
        else:
            bgr_fake, M = face_swapper.get(
                temp_frame, target_face, source_face, paste_back=False
            )
        _get_t1 = time.perf_counter()

        if bgr_fake is None:
            return original_frame

        if not isinstance(bgr_fake, np.ndarray):
            return original_frame

        if getattr(modules.globals, "color_match", False):
            try:
                aligned_target = cv2.warpAffine(
                    temp_frame, M, (face_swapper.input_size[0], face_swapper.input_size[1]),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                )
                matched = apply_color_transfer(bgr_fake, aligned_target)
                # 85% transfer: strong enough to correct webcam casts without
                # aggressively flattening the source face's local contrast.
                bgr_fake = cv2.addWeighted(matched, 0.85, bgr_fake, 0.15, 0.0)
            except Exception:
                pass

        # Pass a dummy aimg with correct shape — _fast_paste_back only uses aimg.shape
        # to create the white mask. Avoids redundant norm_crop2 (~0.6ms).
        _face_size = face_swapper.input_size[0]
        _aimg_dummy = np.empty((_face_size, _face_size, 3), dtype=np.uint8)

        try:
            _inv_for_paste = cv2.invertAffineTransform(M)
            _inv_for_paste = _apply_live_output_geometry(_inv_for_paste, target_face)
            M_for_paste = cv2.invertAffineTransform(_inv_for_paste)
        except Exception:
            M_for_paste = M

        _paste_t0 = time.perf_counter()
        swapped_frame = _fast_paste_back(
            temp_frame, bgr_fake, _aimg_dummy, M_for_paste,
            target_face=target_face, source_affine=M,
        )
        _paste_ms = (time.perf_counter() - _paste_t0) * 1000.0
        _swap_total_ms = (time.perf_counter() - _swap_t0) * 1000.0

        # Print occasionally without flooding the console.
        _now = time.time()
        if not hasattr(swap_face, "_prof_last"):
            swap_face._prof_last = _now
            swap_face._prof_n = 0
            swap_face._prof_get = 0.0
            swap_face._prof_paste = 0.0
            swap_face._prof_prep = 0.0
            swap_face._prof_ort = 0.0
            swap_face._prof_prep = 0.0
            swap_face._prof_ort = 0.0
        swap_face._prof_n += 1
        swap_face._prof_get += max(0.0, _swap_total_ms - _paste_ms)
        swap_face._prof_paste += _paste_ms
        swap_face._prof_prep += max(0.0, (_prep_t - _swap_t0) * 1000.0)
        swap_face._prof_ort += max(0.0, (_get_t1 - _get_t0) * 1000.0)
        if _now - swap_face._prof_last >= 1.0:
            _n = max(1, swap_face._prof_n)
            print(
                f"[SWAP DETAIL] EP={"TRT-FP16" if _tensorrt_enabled else "CUDA-GRAPH/other"} | prep={swap_face._prof_prep/_n:.2f}ms | "
                f"get={swap_face._prof_get/_n:.2f}ms | "
                f"get_call={swap_face._prof_ort/_n:.2f}ms | "
                f"paste={swap_face._prof_paste/_n:.2f}ms | "
                f"total={(_swap_total_ms):.1f}ms"
            )
            swap_face._prof_last = _now
            swap_face._prof_n = 0
            swap_face._prof_get = 0.0
            swap_face._prof_paste = 0.0

    except Exception as e:
        print(f"Error during face swap: {e}")
        return original_frame

    # --- Post-swap Processing (Masking, Opacity, etc.) ---
    # Now, work with the guaranteed uint8 'swapped_frame'

    mouth_ms = 0.0
    poisson_ms = 0.0
    eyes_ms = 0.0


    # --- Poisson Blending ---
    # Mask derived from the swap's own affine (M) + swapped pixels (bgr_fake),
    # so it tracks the swapped face exactly per-frame — no landmark jitter,
    # no EMA, no lag. See _apply_poisson_blend.
    if getattr(modules.globals, "poisson_blend", False):
        _poisson_t0 = time.perf_counter()
        swapped_frame = _apply_poisson_blend(
            swapped_frame, original_frame, target_face, M, bgr_fake
        )
        poisson_ms = (time.perf_counter() - _poisson_t0) * 1000.0

    if preserve_eye_detail:
        _eyes_t0 = time.perf_counter()
        swapped_frame = _preserve_target_eye_detail(
            swapped_frame, original_frame, target_face,
        )
        eyes_ms = (time.perf_counter() - _eyes_t0) * 1000.0

    # Do not replace or copy the whole mouth during Face Swap. The final
    # teeth-holdout runs after GPEN in the live worker.
    mouth_ms = 0.0

    # Timings only: identify the expensive live blending stage without
    # affecting the output.  This is intentionally separate from the base
    # TensorRT swap timing printed above.
    _blend_now = time.time()
    if not hasattr(swap_face, "_blend_profile_last"):
        swap_face._blend_profile_last = _blend_now
        swap_face._blend_profile_n = 0
        swap_face._blend_profile_mouth = 0.0
        swap_face._blend_profile_poisson = 0.0
        swap_face._blend_profile_eyes = 0.0
    swap_face._blend_profile_n += 1
    swap_face._blend_profile_mouth += mouth_ms
    swap_face._blend_profile_poisson += poisson_ms
    swap_face._blend_profile_eyes += eyes_ms
    if _blend_now - swap_face._blend_profile_last >= 1.0:
        _blend_n = max(1, swap_face._blend_profile_n)
        print(
            f"[BLEND DETAIL] mouth={swap_face._blend_profile_mouth/_blend_n:.2f}ms | "
            f"poisson={swap_face._blend_profile_poisson/_blend_n:.2f}ms | "
            f"eyes={swap_face._blend_profile_eyes/_blend_n:.2f}ms | "
            f"frames={_blend_n}",
            flush=True,
        )
        swap_face._blend_profile_last = _blend_now
        swap_face._blend_profile_n = 0
        swap_face._blend_profile_mouth = 0.0
        swap_face._blend_profile_poisson = 0.0
        swap_face._blend_profile_eyes = 0.0

    # Apply opacity blend between the original frame and the swapped frame
    if opacity >= 1.0:
        return swapped_frame.astype(np.uint8)

    # Blend the original_frame with the (potentially mouth-masked) swapped_frame
    final_swapped_frame = gpu_add_weighted(original_frame.astype(np.uint8), 1 - opacity, swapped_frame.astype(np.uint8), opacity, 0)
    return final_swapped_frame.astype(np.uint8)


# --- START: Mac M1-M5 Optimized Face Detection ---
def get_faces_optimized(frame: Frame, use_cache: bool = True) -> Optional[List[Face]]:
    """Optimized face detection for live mode on Apple Silicon"""
    global LAST_DETECTION_TIME, FACE_DETECTION_CACHE
    
    if not use_cache or not IS_APPLE_SILICON:
        # Standard detection
        if modules.globals.many_faces:
            return get_many_faces(frame)
        else:
            face = get_one_face(frame)
            return [face] if face else None
    
    # Adaptive detection rate for live mode
    current_time = time.time()
    time_since_last = current_time - LAST_DETECTION_TIME
    
    # Skip detection if too soon (adaptive frame skipping)
    if time_since_last < DETECTION_INTERVAL and FACE_DETECTION_CACHE:
        return FACE_DETECTION_CACHE.get('faces')
    
    # Perform detection
    LAST_DETECTION_TIME = current_time
    if modules.globals.many_faces:
        faces = get_many_faces(frame)
    else:
        face = get_one_face(frame)
        faces = [face] if face else None
    
    # Cache results
    FACE_DETECTION_CACHE['faces'] = faces
    FACE_DETECTION_CACHE['timestamp'] = current_time
    
    return faces
# --- END: Mac M1-M5 Optimized Face Detection ---

# --- START: Helper function for interpolation and sharpening ---
def apply_post_processing(current_frame: Frame, swapped_face_bboxes: List[np.ndarray]) -> Frame:
    """Applies sharpening and interpolation with Apple Silicon optimizations."""
    global PREVIOUS_FRAME_RESULT

    sharpness_value = getattr(modules.globals, "sharpness", 0.0)
    enable_interpolation = getattr(modules.globals, "enable_interpolation", False)

    # Skip copy when no post-processing is active
    if sharpness_value <= 0.0 and not enable_interpolation:
        PREVIOUS_FRAME_RESULT = None
        return current_frame

    processed_frame = current_frame.copy()

    # 1. Apply Sharpening (if enabled) with optimized kernel for Apple Silicon
    sharpness_value = getattr(modules.globals, "sharpness", 0.0)
    if sharpness_value > 0.0 and swapped_face_bboxes:
        height, width = processed_frame.shape[:2]
        for bbox in swapped_face_bboxes:
            # Ensure bbox is iterable and has 4 elements
            if not hasattr(bbox, '__iter__') or len(bbox) != 4:
                # print(f"Warning: Invalid bbox format for sharpening: {bbox}") # Debug
                continue
            x1, y1, x2, y2 = bbox
            # Ensure coordinates are integers and within bounds
            try:
                 x1, y1 = max(0, int(x1)), max(0, int(y1))
                 x2, y2 = min(width, int(x2)), min(height, int(y2))
            except ValueError:
                # print(f"Warning: Could not convert bbox coordinates to int: {bbox}") # Debug
                continue


            if x2 <= x1 or y2 <= y1:
                continue

            face_region = processed_frame[y1:y2, x1:x2]
            if face_region.size == 0:
                continue

            # Apply sharpening (GPU-accelerated when CUDA OpenCV is available)
            try:
                sigma = 2 if IS_APPLE_SILICON else 3
                sharpened_region = gpu_sharpen(face_region, strength=sharpness_value, sigma=sigma)
                processed_frame[y1:y2, x1:x2] = sharpened_region
            except cv2.error:
                pass


    # 2. Apply Interpolation (if enabled)
    enable_interpolation = getattr(modules.globals, "enable_interpolation", False)
    interpolation_weight = getattr(modules.globals, "interpolation_weight", 0.2)

    final_frame = processed_frame # Start with the current (potentially sharpened) frame

    if enable_interpolation and 0 < interpolation_weight < 1:
        if PREVIOUS_FRAME_RESULT is not None and PREVIOUS_FRAME_RESULT.shape == processed_frame.shape and PREVIOUS_FRAME_RESULT.dtype == processed_frame.dtype:
            # Perform interpolation
            try:
                 final_frame = gpu_add_weighted(
                    PREVIOUS_FRAME_RESULT, 1.0 - interpolation_weight,
                    processed_frame, interpolation_weight,
                    0
                 )
                 # Ensure final frame is uint8
                 final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
            except cv2.error as interp_e:
                 # print(f"Warning: OpenCV error during interpolation: {interp_e}") # Debug
                 final_frame = processed_frame # Use current frame if interpolation fails
                 PREVIOUS_FRAME_RESULT = None # Reset state if error occurs

            # Update the state for the next frame *with the interpolated result*
            PREVIOUS_FRAME_RESULT = final_frame.copy()
        else:
            # If previous frame invalid or doesn't match, use current frame and update state
            if PREVIOUS_FRAME_RESULT is not None and PREVIOUS_FRAME_RESULT.shape != processed_frame.shape:
                # print("Info: Frame shape changed, resetting interpolation state.") # Debug
                pass
            PREVIOUS_FRAME_RESULT = processed_frame.copy()
    else:
         # Interpolation is off or weight is invalid — no need to cache
         PREVIOUS_FRAME_RESULT = None


    return final_frame
# --- END: Helper function for interpolation and sharpening ---


def process_frame(source_face: Face, temp_frame: Frame, target_face: Face = None) -> Frame:
    """Process a single frame, swapping source_face onto detected target(s).

    Args:
        target_face: Pre-detected target face. When provided, skips the
            internal face detection call (saves ~30-40ms per frame).
            Ignored when many_faces mode is active.
    """
    if getattr(modules.globals, "opacity", 1.0) == 0:
        global PREVIOUS_FRAME_RESULT
        PREVIOUS_FRAME_RESULT = None
        return temp_frame

    processed_frame = temp_frame
    swapped_face_bboxes = []

    if modules.globals.many_faces:
        many_faces = get_many_faces(processed_frame)
        if many_faces:
            current_swap_target = processed_frame.copy()
            for face in many_faces:
                current_swap_target = swap_face(source_face, face, current_swap_target)
                if face is not None and hasattr(face, "bbox") and face.bbox is not None:
                    swapped_face_bboxes.append(face.bbox.astype(int))
            processed_frame = current_swap_target
    else:
        if target_face is None:
            target_face = get_one_face(processed_frame)
        if target_face:
            processed_frame = swap_face(source_face, target_face, processed_frame)
            if hasattr(target_face, "bbox") and target_face.bbox is not None:
                swapped_face_bboxes.append(target_face.bbox.astype(int))

    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)
    return final_frame


def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    """Handles complex mapping scenarios (map_faces=True) and live streams."""
    if getattr(modules.globals, "opacity", 1.0) == 0:
        # If opacity is 0, no swap happens, so no post-processing needed.
        # Also reset interpolation state if it was active.
        global PREVIOUS_FRAME_RESULT
        PREVIOUS_FRAME_RESULT = None
        return temp_frame

    processed_frame = temp_frame # Start with the input frame
    swapped_face_bboxes = [] # Keep track of where swaps happened

    # Determine source/target pairs based on mode
    source_target_pairs = []

    # Ensure maps exist before accessing them
    source_target_map = getattr(modules.globals, "source_target_map", None)
    simple_map = getattr(modules.globals, "simple_map", None)

    # Check if target is a file path (image or video) or live stream
    is_file_target = modules.globals.target_path and (is_image(modules.globals.target_path) or is_video(modules.globals.target_path))

    if is_file_target:
        # Processing specific image or video file with pre-analyzed maps
        if source_target_map:
            if modules.globals.many_faces:
                source_face = default_source_face() # Use default source for all targets
                if source_face:
                    for map_data in source_target_map:
                        if is_image(modules.globals.target_path):
                            target_info = map_data.get("target", {})
                            if target_info: # Check if target info exists
                                target_face = target_info.get("face")
                                if target_face:
                                    source_target_pairs.append((source_face, target_face))
                        elif is_video(modules.globals.target_path):
                             # Find faces for the current frame_path in video map
                             target_frames_data = map_data.get("target_faces_in_frame", [])
                             if target_frames_data: # Check if frame data exists
                                 target_frames = [f for f in target_frames_data if f and f.get("location") == temp_frame_path]
                                 for frame_data in target_frames:
                                     faces_in_frame = frame_data.get("faces", [])
                                     if faces_in_frame: # Check if faces exist
                                         for target_face in faces_in_frame:
                                             source_target_pairs.append((source_face, target_face))
            else: # Single face or specific mapping
                 for map_data in source_target_map:
                    source_info = map_data.get("source", {})
                    if not source_info:
                        continue # Skip if no source info
                    source_face = source_info.get("face")
                    if not source_face:
                        continue # Skip if no source defined for this map entry

                    if is_image(modules.globals.target_path):
                        target_info = map_data.get("target", {})
                        if target_info:
                           target_face = target_info.get("face")
                           if target_face:
                              source_target_pairs.append((source_face, target_face))
                    elif is_video(modules.globals.target_path):
                        target_frames_data = map_data.get("target_faces_in_frame", [])
                        if target_frames_data:
                           target_frames = [f for f in target_frames_data if f and f.get("location") == temp_frame_path]
                           for frame_data in target_frames:
                               faces_in_frame = frame_data.get("faces", [])
                               if faces_in_frame:
                                  for target_face in faces_in_frame:
                                      source_target_pairs.append((source_face, target_face))

    else:
        # Live stream or webcam processing (analyze faces on the fly)
        detected_faces = get_many_faces(processed_frame)
        if detected_faces:
            if modules.globals.many_faces:
                 source_face = default_source_face() # Use default source for all detected targets
                 if source_face:
                     for target_face in detected_faces:
                        source_target_pairs.append((source_face, target_face))
            elif simple_map:
                # Use simple_map (source_faces <-> target_embeddings)
                source_faces = simple_map.get("source_faces", [])
                target_embeddings = simple_map.get("target_embeddings", [])

                if source_faces and target_embeddings and len(source_faces) == len(target_embeddings):
                     # Match detected faces to the closest target embedding
                     if len(detected_faces) <= len(target_embeddings):
                          # More targets defined than detected - match each detected face
                          for detected_face in detected_faces:
                              if detected_face.normed_embedding is None:
                                  continue
                              closest_idx, _ = find_closest_centroid(target_embeddings, detected_face.normed_embedding)
                              if 0 <= closest_idx < len(source_faces):
                                  source_target_pairs.append((source_faces[closest_idx], detected_face))
                     else:
                          # More faces detected than targets defined - match each target embedding to closest detected face
                          detected_embeddings = [f.normed_embedding for f in detected_faces if f.normed_embedding is not None]
                          detected_faces_with_embedding = [f for f in detected_faces if f.normed_embedding is not None]
                          if not detected_embeddings:
                              return processed_frame # No embeddings to match

                          for i, target_embedding in enumerate(target_embeddings):
                              if 0 <= i < len(source_faces): # Ensure source face exists for this embedding
                                 closest_idx, _ = find_closest_centroid(detected_embeddings, target_embedding)
                                 if 0 <= closest_idx < len(detected_faces_with_embedding):
                                     source_target_pairs.append((source_faces[i], detected_faces_with_embedding[closest_idx]))
            else: # Fallback: if no map, use default source for the single detected face (if any)
                source_face = default_source_face()
                target_face = get_one_face(processed_frame, detected_faces) # Use faces already detected
                if source_face and target_face:
                    source_target_pairs.append((source_face, target_face))


    # Perform swaps based on the collected pairs
    current_swap_target = processed_frame.copy() # Apply swaps sequentially
    for source_face, target_face in source_target_pairs:
        if source_face and target_face:
            current_swap_target = swap_face(source_face, target_face, current_swap_target)
            if target_face is not None and hasattr(target_face, "bbox") and target_face.bbox is not None:
                swapped_face_bboxes.append(target_face.bbox.astype(int))
    processed_frame = current_swap_target # Assign final result


    # Apply sharpening and interpolation
    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)

    return final_frame


def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """
    Processes a list of frame paths (typically for video).
    Optimized with better memory management and caching.
    Iterates through frames, applies the appropriate swapping logic based on globals,
    and saves the result back to the frame path. Handles multi-threading via caller.
    """
    # Determine which processing function to use based on map_faces global setting
    use_v2 = getattr(modules.globals, "map_faces", False)
    source_face = None # Initialize source_face

    # --- Pre-load source face only if needed (Simple Mode: map_faces=False) ---
    if not use_v2:
        if not source_path or not os.path.exists(source_path):
            update_status(f"Error: Source path invalid or not provided for simple mode: {source_path}", NAME)
            # Log the error but allow proceeding; subsequent check will stop processing.
        else:
            try:
                source_img = imread_unicode(source_path)
                if source_img is None:
                    # Specific error for file reading failure
                    update_status(f"Error reading source image file {source_path}. Please check the path and file integrity.", NAME)
                else:
                    source_face = get_one_face(source_img)
                    if source_face is None:
                        # Specific message for no face detected after successful read
                        update_status(f"Warning: Successfully read source image {source_path}, but no face was detected. Swaps will be skipped.", NAME)
                    # Free memory immediately after extracting face
                    del source_img
            except Exception as e:
                # Print the specific exception caught
                import traceback
                print(f"{NAME}: Caught exception during source image processing for {source_path}:")
                traceback.print_exc() # Print the full traceback
                update_status(f"Error during source image reading or analysis {source_path}: {e}", NAME)
                # Log general exception during the process

    total_frames = len(temp_frame_paths)
    # update_status(f"Processing {total_frames} frames. Use V2 (map_faces): {use_v2}", NAME) # Optional Debug

    # --- Stop processing entirely if in Simple Mode and source face is invalid ---
    if not use_v2 and source_face is None:
        update_status("Halting video processing: Invalid or no face detected in source image for simple mode.", NAME)
        if progress:
            # Ensure the progress bar completes if it was started
            remaining_updates = total_frames - progress.n if hasattr(progress, 'n') else total_frames
            if remaining_updates > 0:
                progress.update(remaining_updates)
        return # Exit the function entirely

    # --- Process each frame path provided in the list ---
    # Note: In the current core.py multi_process_frame, temp_frame_paths will usually contain only ONE path per call.
    for i, temp_frame_path in enumerate(temp_frame_paths):
        # update_status(f"Processing frame {i+1}/{total_frames}: {os.path.basename(temp_frame_path)}", NAME) # Optional Debug

        # Read the target frame
        temp_frame = None
        try:
            temp_frame = imread_unicode(temp_frame_path)
            if temp_frame is None:
                print(f"{NAME}: Error: Could not read frame: {temp_frame_path}, skipping.")
                if progress:
                    progress.update(1)
                continue # Skip this frame if read fails
        except Exception as read_e:
            print(f"{NAME}: Error reading frame {temp_frame_path}: {read_e}, skipping.")
            if progress:
                progress.update(1)
            continue

        # Select processing function and execute
        result_frame = None
        try:
            if use_v2:
                # V2 uses global maps and needs the frame path for lookup in video mode
                # update_status(f"Using process_frame_v2 for: {os.path.basename(temp_frame_path)}", NAME) # Optional Debug
                result_frame = process_frame_v2(temp_frame, temp_frame_path)
            else:
                # Simple mode uses the pre-loaded source_face (already checked for validity above)
                # update_status(f"Using process_frame (simple) for: {os.path.basename(temp_frame_path)}", NAME) # Optional Debug
                result_frame = process_frame(source_face, temp_frame) # source_face is guaranteed to be valid here

            # Check if processing actually returned a frame
            if result_frame is None:
                 print(f"{NAME}: Warning: Processing returned None for frame {temp_frame_path}. Using original.")
                 result_frame = temp_frame

        except Exception as proc_e:
            print(f"{NAME}: Error processing frame {temp_frame_path}: {proc_e}")
            # import traceback # Optional for detailed debugging
            # traceback.print_exc()
            result_frame = temp_frame # Use original frame on processing error

        # Write the result back to the same frame path with optimized compression
        try:
            # Use PNG compression level 3 (faster) instead of default 9
            write_success = imwrite_unicode(temp_frame_path, result_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not write_success:
                print(f"{NAME}: Error: Failed to write processed frame to {temp_frame_path}")
        except Exception as write_e:
            print(f"{NAME}: Error writing frame {temp_frame_path}: {write_e}")
        
        # Free memory immediately after processing
        del temp_frame
        if result_frame is not None:
            del result_frame

        # Update progress bar
        if progress:
            progress.update(1)
        # else: # Basic console progress (optional)
        #     if (i + 1) % 10 == 0 or (i + 1) == total_frames: # Update every 10 frames or on last frame
        #        update_status(f"Processed frame {i+1}/{total_frames}", NAME)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Processes a single target image."""
    # --- Reset interpolation state for single image processing ---
    global PREVIOUS_FRAME_RESULT
    PREVIOUS_FRAME_RESULT = None
    # ---

    use_v2 = getattr(modules.globals, "map_faces", False)

    # Read target first
    try:
        target_frame = imread_unicode(target_path)
        if target_frame is None:
            update_status(f"Error: Could not read target image: {target_path}", NAME)
            return
    except Exception as read_e:
        update_status(f"Error reading target image {target_path}: {read_e}", NAME)
        return

    result = None
    try:
        if use_v2:
            if getattr(modules.globals, "many_faces", False):
                 update_status("Processing image with 'map_faces' and 'many_faces'. Using pre-analysis map.", NAME)
            # V2 processes based on global maps, doesn't need source_path here directly
            # Assumes maps are pre-populated. Pass target_path for map lookup.
            result = process_frame_v2(target_frame, target_path)

        else: # Simple mode
            try:
                source_img = imread_unicode(source_path)
                if source_img is None:
                    update_status(f"Error: Could not read source image: {source_path}", NAME)
                    return
                source_face = get_one_face(source_img)
                if not source_face:
                    update_status(f"Error: No face found in source image: {source_path}", NAME)
                    return
            except Exception as src_e:
                 update_status(f"Error reading or analyzing source image {source_path}: {src_e}", NAME)
                 return

            result = process_frame(source_face, target_frame)

        # Write the result if processing was successful
        if result is not None:
            write_success = imwrite_unicode(output_path, result)
            if write_success:
                update_status(f"Output image saved to: {output_path}", NAME)
            else:
                update_status(f"Error: Failed to write output image to {output_path}", NAME)
        else:
            # This case might occur if process_frame/v2 returns None unexpectedly
            update_status("Image processing failed (result was None).", NAME)

    except Exception as proc_e:
         update_status(f"Error during image processing: {proc_e}", NAME)
         # import traceback
         # traceback.print_exc()


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Sets up and calls the frame processing for video."""
    # --- Reset interpolation state before starting video processing ---
    global PREVIOUS_FRAME_RESULT
    PREVIOUS_FRAME_RESULT = None
    # ---

    mode_desc = "'map_faces'" if getattr(modules.globals, "map_faces", False) else "'simple'"
    if getattr(modules.globals, "map_faces", False) and getattr(modules.globals, "many_faces", False):
        mode_desc += " and 'many_faces'. Using pre-analysis map."
    update_status(f"Processing video with {mode_desc} mode.", NAME)

    # Pass the correct source_path (needed for simple mode in process_frames)
    # The core processing logic handles calling the right frame function (process_frames)
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames # Pass the newly modified process_frames
    )

# ==========================
# MASKING FUNCTIONS (Mostly unchanged, added safety checks and minor improvements)
# ==========================


def restore_target_mouth(frame: np.ndarray, target_frame: np.ndarray, face: Face, track_key: int = 0) -> np.ndarray:
    """Preserve the current swapped inner-mouth opening after GPEN.

    Only the teeth/mouth cavity inside the 106-point inner-lip contour is
    restored. This retains the face swapper's current expression while keeping
    the enhanced skin and swapped outer lips untouched.
    """
    if frame is None or target_frame is None or face is None:
        return frame
    try:
        size = max(0.0, min(100.0, float(getattr(modules.globals, 'mouth_mask_size', 0.0))))
        enabled = bool(getattr(modules.globals, 'mouth_mask', False)) or size > 0.0
        if not enabled or size <= 0.0:
            return frame

        # The face output can be moved/scaled/rotated after detection. Use
        # the current prepared landmarks, then apply that same output
        # transform, so the mouth holdout follows the rendered face.
        lm = getattr(face, 'landmark_2d_106', None)
        if lm is None:
            lm = getattr(face, '_syrex_original_landmark_2d_106', None)
        if not isinstance(lm, np.ndarray) or lm.shape[0] < 104:
            return frame
        lm = np.asarray(lm, dtype=np.float32)
        if not np.all(np.isfinite(lm)):
            return frame

        try:
            bbox = np.asarray(getattr(face, 'bbox', None), dtype=np.float32).reshape(-1)
            if bbox.size >= 4 and np.all(np.isfinite(bbox)):
                x1b, y1b, x2b, y2b = map(float, bbox[:4])
                bw, bh = max(1.0, x2b - x1b), max(1.0, y2b - y1b)
                cx, cy = (x1b + x2b) * 0.5, (y1b + y2b) * 0.5
                scale = max(0.60, min(1.40, float(getattr(modules.globals, 'face_scale', 1.0))))
                angle = np.deg2rad(max(-30.0, min(30.0, float(getattr(modules.globals, 'face_rotation', 0.0)))))
                dx = (max(-100.0, min(100.0, float(getattr(modules.globals, 'face_offset_x', 0.0)))) / 100.0) * bw * 0.75
                dy = (max(-100.0, min(100.0, float(getattr(modules.globals, 'face_offset_y', 0.0)))) / 100.0) * bh * 0.75
                c, s = float(np.cos(angle)), float(np.sin(angle))
                transform = np.array([[scale * c, -scale * s], [scale * s, scale * c]], dtype=np.float32)
                lm = ((lm - np.array([cx, cy], dtype=np.float32)) @ transform.T
                      + np.array([cx + dx, cy + dy], dtype=np.float32))
        except Exception:
            pass

        # The 106-point model orders mouth points as outer lip 52..63 and
        # inner mouth 64..71. Points 96..104 are an eyebrow, not the mouth.
        inner = lm[64:72].copy()
        upper_c = np.mean(lm[64:69], axis=0)
        lower_c = np.mean(lm[69:72], axis=0)
        mouth_h = float(np.linalg.norm(lower_c - upper_c))
        mouth_w = float(np.ptp(inner[:, 0]))
        closed_mouth = mouth_w < 10.0 or mouth_h < max(2.2, mouth_w * 0.035)
        if closed_mouth:
            # When lips are closed there is no useful inner-mouth polygon.
            # A small outer-lip fallback still protects the swapped lip
            # texture from GPEN, instead of silently doing nothing.
            inner = lm[52:64].copy()
            mouth_w = float(np.ptp(inner[:, 0]))
            mouth_h = float(np.ptp(inner[:, 1]))
            if mouth_w < 10.0 or mouth_h < 2.0:
                return frame

        h, w = frame.shape[:2]
        if target_frame.shape[:2] != (h, w):
            target_frame = cv2.resize(target_frame, (w, h), interpolation=cv2.INTER_AREA)

        center = inner.mean(axis=0)
        inset = 0.04 + 0.08 * (1.0 - size / 100.0)
        factor = (max(0.72, 0.92 - 0.12 * (1.0 - size / 100.0))
                  if not closed_mouth else 0.78)
        poly = center + (inner - center) * factor
        erode_px = max(1, int(round(min(w, h) * inset * 0.0035)))
        pad = erode_px + 3
        x0 = max(0, int(np.floor(poly[:, 0].min())) - pad)
        y0 = max(0, int(np.floor(poly[:, 1].min())) - pad)
        x1 = min(w, int(np.ceil(poly[:, 0].max())) + pad + 1)
        y1 = min(h, int(np.ceil(poly[:, 1].max())) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            return frame

        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        local_poly = np.round(poly - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [local_poly], 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        mask = cv2.erode(mask, kernel, iterations=1)
        # Preserve a tiny lip-edge feather.  The pixels inside the mask are
        # still restored at full strength, so the enhancer cannot introduce a
        # dark or synthetic mouth over the live expression.
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        alpha = (mask.astype(np.float32) / 255.0) * (size / 100.0)
        if not np.any(alpha > 0.02):
            return frame

        src_u8 = target_frame[y0:y1, x0:x1]
        soft = cv2.GaussianBlur(src_u8, (0, 0), 0.65)
        src = cv2.addWeighted(src_u8, 1.22, soft, -0.22, 1.0).astype(np.float32)
        dst = frame[y0:y1, x0:x1].astype(np.float32)
        out = src * alpha[..., None] + dst * (1.0 - alpha[..., None])
        frame[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
        return frame
    except Exception as exc:
        print(f'[MOUTH MASK] inner-mouth restore failed: {exc}', flush=True)
        return frame

def create_lower_mouth_mask(
    face: Face, frame: Frame
) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    """Create a narrow inner-mouth holdout.

    The previous versions used the outer lip polygon, which allowed the
    original upper/lower lip skin to leak into the holdout.  This version
    deliberately creates a smaller ellipse inside the lip boundary so the
    swapped lips remain untouched while the central teeth/braces area can be
    preserved from the target camera frame.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    mouth_polygon = None
    mouth_box = (0, 0, 0, 0)

    if face is None or not hasattr(face, 'landmark_2d_106'):
        return mask, mouth_cutout, mouth_box, mouth_polygon

    landmarks = getattr(
        face,
        '_syrex_original_landmark_2d_106',
        getattr(face, 'landmark_2d_106', None),
    )
    if landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.shape[0] < 64:
        return mask, mouth_cutout, mouth_box, mouth_polygon

    try:
        outer = landmarks[52:64].astype(np.float32)
        if not np.all(np.isfinite(outer)):
            return mask, mouth_cutout, mouth_box, mouth_polygon

        x0, y0 = np.min(outer, axis=0)
        x1, y1 = np.max(outer, axis=0)
        bw = max(2.0, float(x1 - x0))
        bh = max(2.0, float(y1 - y0))
        cx = float((x0 + x1) * 0.5)

        size = max(0.0, min(100.0, float(
            getattr(modules.globals, 'mouth_mask_size', 0.0)
        )))
        s = size / 100.0

        # Keep the holdout well inside the lips.  The vertical centre is
        # shifted slightly downward because upper lip pixels otherwise enter
        # the protected area first.
        cy = float(y0 + bh * (0.56 + 0.04 * s))

        # At 100% this reaches a useful teeth/braces region, but still stays
        # away from the outer lip border.  At 0% the holdout is intentionally
        # tiny, effectively disabled.
        rx = max(2, int(round(bw * (0.08 + 0.25 * s))))
        ry = max(2, int(round(bh * (0.05 + 0.20 * s))))

        if rx <= 2 or ry <= 2:
            return mask, mouth_cutout, mouth_box, mouth_polygon

        min_x = max(0, int(round(cx - rx)))
        max_x = min(frame.shape[1], int(round(cx + rx + 1)))
        min_y = max(0, int(round(cy - ry)))
        max_y = min(frame.shape[0], int(round(cy + ry + 1)))
        if max_x <= min_x or max_y <= min_y:
            return mask, mouth_cutout, mouth_box, mouth_polygon

        roi_h = max_y - min_y
        roi_w = max_x - min_x
        mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
        center_local = (int(round(cx)) - min_x, int(round(cy)) - min_y)
        axes_local = (min(rx, roi_w // 2), min(ry, roi_h // 2))
        cv2.ellipse(mask_roi, center_local, axes_local, 0, 0, 360, 255, -1)

        # Very small feathering, so the lip edge stays crisp rather than being
        # visibly averaged with the original target lip.
        blur_k = 5
        mask_roi = cv2.GaussianBlur(mask_roi, (blur_k, blur_k), 0)
        mask[min_y:max_y, min_x:max_x] = mask_roi

        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
        mouth_polygon = np.array([
            [int(cx - rx), int(cy)],
            [int(cx), int(cy - ry)],
            [int(cx + rx), int(cy)],
            [int(cx), int(cy + ry)],
        ], dtype=np.int32)
        mouth_box = (min_x, min_y, max_x, max_y)
        return mask, mouth_cutout, mouth_box, mouth_polygon

    except Exception as e:
        print(f"Error in create_lower_mouth_mask: {e}", flush=True)
        return mask, mouth_cutout, mouth_box, mouth_polygon


def draw_mouth_mask_visualization(
    frame: Frame, face: Face, mouth_mask_data: tuple
) -> Frame:

    # Validate inputs
    if frame is None or face is None or mouth_mask_data is None or len(mouth_mask_data) != 4:
        return frame # Return original frame if inputs are invalid

    mask, mouth_cutout, box, lower_lip_polygon = mouth_mask_data
    (min_x, min_y, max_x, max_y) = box

    # Check if polygon is valid for drawing
    if lower_lip_polygon is None or not isinstance(lower_lip_polygon, np.ndarray) or len(lower_lip_polygon) < 3:
        return frame # Cannot draw without a valid polygon

    vis_frame = frame.copy()
    height, width = vis_frame.shape[:2]

    # Ensure box coordinates are valid integers within frame bounds
    try:
        min_x, min_y = max(0, int(min_x)), max(0, int(min_y))
        max_x, max_y = min(width, int(max_x)), min(height, int(max_y))
    except ValueError:
        # print("Warning: Invalid coordinates for mask visualization box.")
        return frame

    if max_x <= min_x or max_y <= min_y:
        return frame # Invalid box

    # Draw the lower lip polygon (green outline)
    try:
         # Ensure polygon points are within frame boundaries before drawing
         safe_polygon = lower_lip_polygon.copy()
         safe_polygon[:, 0] = np.clip(safe_polygon[:, 0], 0, width - 1)
         safe_polygon[:, 1] = np.clip(safe_polygon[:, 1], 0, height - 1)
         cv2.polylines(vis_frame, [safe_polygon.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    except Exception as e:
        print(f"Error drawing polygon for visualization: {e}") # Optional debug
        pass

    # Draw bounding box (red rectangle)
    cv2.rectangle(vis_frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

    # Optional: Add labels
    label_pos_y = min_y - 10 if min_y > 20 else max_y + 15 # Adjust position based on box location
    label_pos_x = min_x
    try:
        cv2.putText(vis_frame, "Mouth Mask", (label_pos_x, label_pos_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    except Exception as e:
        # print(f"Error drawing text for visualization: {e}") # Optional debug
        pass


    return vis_frame


def apply_mouth_area(
    frame: np.ndarray,
    mouth_cutout: np.ndarray,
    mouth_box: tuple,
    mouth_polygon: np.ndarray, # Specific polygon for the mouth area itself
) -> np.ndarray:

    # Basic validation
    if (frame is None or mouth_cutout is None or mouth_box is None or
        mouth_polygon is None):
        # print("Warning: Invalid input (None value) to apply_mouth_area") # Optional debug
        return frame
    if mouth_cutout.size == 0 or len(mouth_polygon) < 3:
        # print("Warning: Invalid input (empty array/polygon) to apply_mouth_area") # Optional debug
        return frame

    try: # Wrap main logic in try-except
        min_x, min_y, max_x, max_y = map(int, mouth_box) # Ensure integer coords
        box_width = max_x - min_x
        box_height = max_y - min_y

        # Check box validity
        if box_width <= 0 or box_height <= 0:
            # print("Warning: Invalid mouth box dimensions in apply_mouth_area.")
            return frame

        # Define the Region of Interest (ROI) on the target frame (swapped frame)
        frame_h, frame_w = frame.shape[:2]
        # Clamp coordinates strictly within frame boundaries
        min_y, max_y = max(0, min_y), min(frame_h, max_y)
        min_x, max_x = max(0, min_x), min(frame_w, max_x)

        # Recalculate box dimensions based on clamped coords
        box_width = max_x - min_x
        box_height = max_y - min_y
        if box_width <= 0 or box_height <= 0:
            # print("Warning: ROI became invalid after clamping in apply_mouth_area.")
            return frame # ROI is invalid

        roi = frame[min_y:max_y, min_x:max_x]

        # Ensure ROI extraction was successful
        if roi.size == 0:
            # print("Warning: Extracted ROI is empty in apply_mouth_area.")
            return frame

        # Resize mouth cutout from original frame to fit the ROI size
        resized_mouth_cutout = None
        if roi.shape[:2] != mouth_cutout.shape[:2]:
             # Check if mouth_cutout has valid dimensions before resizing
             if mouth_cutout.shape[0] > 0 and mouth_cutout.shape[1] > 0:
                  resized_mouth_cutout = gpu_resize(mouth_cutout, (box_width, box_height), interpolation=cv2.INTER_LINEAR)
             else:
                 # print("Warning: mouth_cutout has invalid dimensions, cannot resize.")
                 return frame # Cannot proceed without valid cutout
        else:
             resized_mouth_cutout = mouth_cutout

        # If resize failed or original was invalid
        if resized_mouth_cutout is None or resized_mouth_cutout.size == 0:
            # print("Warning: Mouth cutout is invalid after resize attempt.")
            return frame

        # --- Mask Creation ---
        # Create a mask based on the mouth_polygon, relative to the ROI
        polygon_mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
        adjusted_polygon = mouth_polygon - [min_x, min_y]
        cv2.fillPoly(polygon_mask_roi, [adjusted_polygon.astype(np.int32)], 255)

        # Feather the edges with Gaussian blur for smooth blending
        feather_amount = max(1, min(30, min(box_width, box_height) // 8))
        kernel_size = 2 * feather_amount + 1
        feathered_mask = cv2.GaussianBlur(polygon_mask_roi.astype(np.float32), (kernel_size, kernel_size), 0)

        # Normalize to [0.0, 1.0]
        max_val = feathered_mask.max()
        if max_val > 1e-6:
            feathered_mask = feathered_mask / max_val
        else:
            feathered_mask.fill(0.0)

        # The slider controls mask size. Once enabled, keep full local restoration
        # strength so small teeth/bracket/wire details are not diluted.
        feathered_mask *= 1.0

        # --- Blending: paste original mouth onto swapped face ---
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            mask_3ch = feathered_mask[:, :, np.newaxis].astype(np.float32)
            inv_mask = 1.0 - mask_3ch

            # Blend: (original_mouth * mask) + (swapped_face * (1 - mask))
            blended_roi = (resized_mouth_cutout.astype(np.float32) * mask_3ch +
                           roi.astype(np.float32) * inv_mask)

            frame[min_y:max_y, min_x:max_x] = np.clip(blended_roi, 0, 255).astype(np.uint8)

    except Exception as e:
        print(f"Error applying mouth area: {e}") # Optional debug
        # import traceback
        # traceback.print_exc()
        pass # Don't crash, just return the frame as is

    return frame


def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    """Creates a feathered mask covering the whole face area based on landmarks."""
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return np.zeros((0, 0), dtype=np.uint8)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8) # Start with uint8

    # Validate inputs
    if face is None or not hasattr(face, 'landmark_2d_106'):
        # print("Warning: Invalid face or frame for create_face_mask.")
        return mask # Return empty mask

    landmarks = face.landmark_2d_106
    if landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.shape[0] < 106:
        # print("Warning: Invalid or insufficient landmarks for face mask.")
        return mask # Return empty mask

    try: # Wrap main logic in try-except
        # Filter out non-finite landmark values
        if not np.all(np.isfinite(landmarks)):
            # print("Warning: Non-finite values detected in landmarks for face mask.")
            return mask

        landmarks_int = landmarks.astype(np.int32)

        # Use standard face outline landmarks (0-32)
        # Use standard face outline (0-32)
        face_outline = landmarks_int[0:33]

        # Estimate forehead points to ensure mask covers the whole face (including forehead)
        # This is critical for Poisson blending to work correctly on the forehead
        eyebrows = landmarks_int[33:43]
        if eyebrows.shape[0] > 0:
            chin = landmarks_int[16]
            eyebrow_center = np.mean(eyebrows, axis=0)
            
            # Vector from chin to eyebrows (upwards)
            up_vector = eyebrow_center - chin
            norm = np.linalg.norm(up_vector)
            if norm > 0:
                up_vector /= norm
                
                # Extend upwards by 1.0 of the chin-to-eyebrow distance (aggressive coverage)
                # This ensures the mask covers the entire forehead for proper blending
                forehead_offset = up_vector * (norm * 1.0)
                
                # Shift eyebrows up to create forehead points
                forehead_points = eyebrows + forehead_offset
                
                # Expand the top points slightly outwards to cover forehead corners
                # Calculate the center of the new top points
                top_center = np.mean(forehead_points, axis=0)
                
                # Expand outwards by 20%
                forehead_points = (forehead_points - top_center) * 1.2 + top_center
                
                # Combine outline and forehead points
                face_outline = np.concatenate((face_outline, forehead_points.astype(np.int32)), axis=0)

        # Calculate convex hull of these points
        # Use try-except as convexHull can fail on degenerate input
        try:
             hull = cv2.convexHull(face_outline.astype(np.float32)) # Use float for accuracy
             if hull is None or len(hull) < 3:
                 # print("Warning: Convex hull calculation failed or returned too few points.")
                 # Fallback: use bounding box of landmarks? Or just return empty mask?
                 return mask

             # Draw the filled convex hull on the mask
             cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        except Exception as hull_e:
             print(f"Error creating convex hull for face mask: {hull_e}")
             return mask # Return empty mask on error


        # Apply Gaussian blur to feather the mask edges (GPU-accelerated when available)
        blur_k_size = getattr(modules.globals, "face_mask_blur", 31) # Default 31
        blur_k_size = max(1, blur_k_size // 2 * 2 + 1) # Ensure odd and positive
        mask = gpu_gaussian_blur(mask, (blur_k_size, blur_k_size), 0)

        # --- Optional: Return float mask for apply_mouth_area ---
        # mask = mask.astype(float) / 255.0
        # ---

    except IndexError:
        # print("Warning: Landmark index out of bounds for face mask.") # Optional debug
        pass
    except Exception as e:
        print(f"Error creating face mask: {e}") # Print unexpected errors
        # import traceback
        # traceback.print_exc()
        pass

    return mask # Return uint8 mask


def apply_color_transfer(source, target):
    """
    Apply color transfer using LAB color space. Handles potential division by zero and ensures output is uint8.
    """
    # Input validation
    if source is None or target is None or source.size == 0 or target.size == 0:
        # print("Warning: Invalid input to apply_color_transfer.")
        return source # Return original source if invalid input

    # Ensure images are 3-channel BGR uint8
    if len(source.shape) != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        # print("Warning: Source image for color transfer is not uint8 BGR.")
        # Attempt conversion if possible, otherwise return original
        try:
            if len(source.shape) == 2: # Grayscale
                source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            source = np.clip(source, 0, 255).astype(np.uint8)
            if len(source.shape) != 3 or source.shape[2] != 3:
                raise ValueError("Conversion failed")
        except Exception:
            return source
    if len(target.shape) != 3 or target.shape[2] != 3 or target.dtype != np.uint8:
        # print("Warning: Target image for color transfer is not uint8 BGR.")
        try:
            if len(target.shape) == 2: # Grayscale
                target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
            target = np.clip(target, 0, 255).astype(np.uint8)
            if len(target.shape) != 3 or target.shape[2] != 3:
                raise ValueError("Conversion failed")
        except Exception:
             return source # Return original source if target invalid

    result_bgr = source # Default to original source in case of errors

    try:
        # Convert to float32 [0, 1] range for LAB conversion
        source_float = source.astype(np.float32) / 255.0
        target_float = target.astype(np.float32) / 255.0

        source_lab = cv2.cvtColor(source_float, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_float, cv2.COLOR_BGR2LAB)

        # Compute statistics
        source_mean, source_std = cv2.meanStdDev(source_lab)
        target_mean, target_std = cv2.meanStdDev(target_lab)

        # Reshape for broadcasting
        source_mean = source_mean.reshape((1, 1, 3))
        source_std = source_std.reshape((1, 1, 3))
        target_mean = target_mean.reshape((1, 1, 3))
        target_std = target_std.reshape((1, 1, 3))

        # Avoid division by zero or very small std deviations (add epsilon)
        epsilon = 1e-6
        source_std = np.maximum(source_std, epsilon)
        # target_std = np.maximum(target_std, epsilon) # Target std can be small

        # Perform color transfer in LAB space
        result_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean

        # --- No explicit clipping needed in LAB space typically ---
        # Clipping is handled implicitly by the conversion back to BGR and then to uint8

        # Convert back to BGR float [0, 1]
        result_bgr_float = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

        # Clip final BGR values to [0, 1] range before scaling to [0, 255]
        result_bgr_float = np.clip(result_bgr_float, 0.0, 1.0)

        # Convert back to uint8 [0, 255]
        result_bgr = (result_bgr_float * 255.0).astype("uint8")

    except cv2.error as e:
         # print(f"OpenCV error during color transfer: {e}. Returning original source.") # Optional debug
         return source # Return original source if conversion fails
    except Exception as e:
         # print(f"Unexpected color transfer error: {e}. Returning original source.") # Optional debug
         # import traceback
         # traceback.print_exc()
         return source

    return result_bgr
