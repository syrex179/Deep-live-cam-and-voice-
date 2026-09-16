"""Shared ONNX-based face enhancement utilities for GPEN-BFR models.

Provides session creation, pre/post processing, and the core
enhance-face-via-ONNX pipeline.
"""

import os
import platform
import threading
import time
from typing import Any

import cv2
import numpy as np
import onnxruntime

import modules.globals
from modules.platform_info import OPENVINO_PROVIDER_CONFIG

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Limit concurrent ONNX calls to avoid VRAM exhaustion on multi-face frames
THREAD_SEMAPHORE = threading.Semaphore(min(max(1, (os.cpu_count() or 1)), 8))

# Live GPEN detailed profiler. Diagnostics only; processing math is unchanged.
_GPEN_PROFILE_LOCK = threading.Lock()
_GPEN_PROFILE = {
    "n": 0,
    "affine": 0.0,
    "crop": 0.0,
    "pre": 0.0,
    "infer": 0.0,
    "post": 0.0,
    "warp_blend": 0.0,
}

# One-frame temporal cache used by the live UI when GPEN inference is skipped.
# We cache the aligned/restored 256x256 face, then reproject that cached result
# into the current frame geometry. This avoids the "GPEN on / GPEN off" flicker
# while keeping inference cost to every 2nd frame.
_cached_enhanced = None
_cached_face_bbox = None
_CACHE_MAX_CENTER_SHIFT = 0.55


def build_provider_config(providers=None):
    """Wrap raw provider name strings with optimised CUDA / CoreML options.

    Providers that are already ``(name, options_dict)`` tuples are passed
    through unchanged.  Non-CUDA providers are left as bare strings.
    """
    if providers is None:
        providers = modules.globals.execution_providers

    config = []
    for p in providers:
        if isinstance(p, tuple):
            # Already configured – pass through
            config.append(p)
        elif p == "CUDAExecutionProvider":
            # Use bare provider — ONNX Runtime's defaults are fastest on
            # modern GPUs (Blackwell/sm_120).  Custom options like
            # EXHAUSTIVE cudnn_conv_algo_search hurt performance on these
            # architectures.
            config.append(p)
        elif p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
            config.append((
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "AllowLowPrecisionAccumulationOnGPU": 1,
                },
            ))
        elif p == "OpenVINOExecutionProvider":
            # AUTO lets OpenVINO select the best device
            config.append(OPENVINO_PROVIDER_CONFIG)
        else:
            config.append(p)
    return config


def run_inference(session: onnxruntime.InferenceSession,
                  input_name: str,
                  input_tensor: "np.ndarray") -> "np.ndarray":
    """Run ONNX inference, using IO binding when a CUDA session is active.

    IO binding avoids redundant host↔device copies by transferring the
    input tensor directly to GPU memory and letting ONNX Runtime allocate
    the output on the device.  Falls back to the standard ``session.run``
    path for non-CUDA providers or if binding fails.
    """
    if "CUDAExecutionProvider" in session.get_providers():
        try:
            io_binding = session.io_binding()

            # Input: numpy → GPU
            ort_input = onnxruntime.OrtValue.ortvalue_from_numpy(
                input_tensor, "cuda", 0,
            )
            io_binding.bind_ortvalue_input(input_name, ort_input)

            # Output: allocate on GPU (avoids a CPU-side allocation)
            output_name = session.get_outputs()[0].name
            io_binding.bind_output(output_name, "cuda", 0)

            session.run_with_iobinding(io_binding)

            return io_binding.get_outputs()[0].numpy()
        except Exception:
            # Fall back to standard path (e.g. ORT version mismatch,
            # unsupported op, or VRAM pressure)
            pass

    return session.run(None, {input_name: input_tensor})[0]


def create_onnx_session(model_path: str) -> onnxruntime.InferenceSession:
    """Create an optimized ONNX Runtime session.

    On NVIDIA/Windows, prefer TensorRT FP16 for GPEN.  Fall back to CUDA and
    then CPU if TensorRT cannot build the engine.  Apple Silicon keeps the
    existing CoreML path.
    """
    if IS_APPLE_SILICON:
        from modules.onnx_optimize import optimize_for_coreml
        try:
            import onnx
            m = onnx.load(model_path)
            inp = m.graph.input[0]
            dims = inp.type.tensor_type.shape.dim
            shape = tuple(d.dim_value for d in dims if d.dim_value > 0)
            input_shape = shape if len(shape) == 4 else None
        except Exception:
            input_shape = None
        model_path = optimize_for_coreml(model_path, input_shape=input_shape)

    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    base_providers = build_provider_config()

    # TensorRT is intentionally limited to enhancer sessions. This does not
    # change the Face Swap TensorRT session or its geometry.
    is_windows = platform.system() == "Windows"
    has_cuda = any(
        (p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider"
        for p in base_providers
    )

    if is_windows and has_cuda:
        try:
            trt_cache = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "trt_cache_gpen",
            )
            os.makedirs(trt_cache, exist_ok=True)

            trt_providers = [
                ("TensorrtExecutionProvider", {
                    "device_id": 0,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": trt_cache,
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": trt_cache,
                    "trt_max_workspace_size": 4294967296,
                }),
                ("CUDAExecutionProvider", {"device_id": 0}),
                ("CPUExecutionProvider", {}),
            ]

            session = onnxruntime.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=trt_providers,
            )
            actual = session.get_providers()
            if actual and actual[0] == "TensorrtExecutionProvider":
                print(f"[DLC.GPEN] TensorRT FP16 enabled | providers={actual}")
                print(f"[DLC.GPEN] TRT cache={trt_cache}")
                return session
            print(f"[DLC.GPEN] TensorRT not primary, falling back: {actual}")
        except Exception as e:
            print(f"[DLC.GPEN] TensorRT init failed, falling back to CUDA: {e}")

    session = onnxruntime.InferenceSession(
        model_path, sess_options=session_options, providers=base_providers,
    )
    print(f"[DLC.GPEN] providers={session.get_providers()}")
    return session


def warmup_session(session: onnxruntime.InferenceSession) -> None:
    """Run a dummy inference pass to trigger JIT / compile caching."""
    try:
        input_feed = {
            inp.name: np.zeros(
                [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape],
                dtype=np.float32,
            )
            for inp in session.get_inputs()
        }
        session.run(None, input_feed)
    except Exception as e:
        print(f"ONNX enhancer warmup skipped (non-fatal): {e}")


def preprocess_face(face_img: np.ndarray, input_size: int) -> np.ndarray:
    """Resize, normalize, and convert a BGR face crop to ONNX input blob.

    GPEN-BFR expects [1, 3, H, W] float32 in RGB, normalized to [-1, 1].
    """
    resized = cv2.resize(face_img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob


def postprocess_face(output: np.ndarray) -> np.ndarray:
    """Convert ONNX output [1, 3, H, W] float32 back to BGR uint8 image."""
    img = output[0].transpose(1, 2, 0)
    img = ((img + 1.0) / 2.0 * 255.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _get_face_affine(face: Any, input_size: int):
    """Compute affine transform to align a face to GPEN input space.

    Returns (M, inv_M) — forward and inverse affine matrices.
    """
    template = np.array([
        [0.31556875, 0.4615741],
        [0.68262291, 0.4615741],
        [0.50009375, 0.6405054],
        [0.34947187, 0.8246919],
        [0.65343645, 0.8246919],
    ], dtype=np.float32) * input_size

    landmarks = None
    if hasattr(face, "kps") and face.kps is not None:
        landmarks = face.kps.astype(np.float32)
    elif hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        lm106 = face.landmark_2d_106
        landmarks = np.array([
            lm106[38],  # left eye
            lm106[88],  # right eye
            lm106[86],  # nose tip
            lm106[52],  # left mouth
            lm106[61],  # right mouth
        ], dtype=np.float32)

    if landmarks is None or len(landmarks) < 5:
        return None, None

    M = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)[0]
    if M is None:
        return None, None
    inv_M = cv2.invertAffineTransform(M)
    return M, inv_M



def reuse_last_enhancement(
    frame: np.ndarray,
    face: Any,
    input_size: int,
) -> np.ndarray:
    """Reuse the most recent GPEN result with current face geometry.

    This is intended for live mode only. It avoids GPEN inference on alternate
    frames, while still producing a GPEN-restored face every displayed frame.
    """
    global _cached_enhanced, _cached_face_bbox

    if _cached_enhanced is None:
        return frame
    if _cached_face_bbox is not None and hasattr(face, "bbox") and face.bbox is not None:
        bbox = np.asarray(face.bbox, dtype=np.float32)
        old = _cached_face_bbox
        old_w = max(1.0, old[2] - old[0])
        old_h = max(1.0, old[3] - old[1])
        new_w = max(1.0, bbox[2] - bbox[0])
        new_h = max(1.0, bbox[3] - bbox[1])
        old_c = np.array([(old[0] + old[2]) * 0.5, (old[1] + old[3]) * 0.5], dtype=np.float32)
        new_c = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)
        max_dim = max(old_w, old_h, new_w, new_h)
        if float(np.linalg.norm(new_c - old_c)) > _CACHE_MAX_CENTER_SHIFT * max_dim:
            return frame
        if new_w < 0.55 * old_w or new_w > 1.8 * old_w or new_h < 0.55 * old_h or new_h > 1.8 * old_h:
            return frame

    M, inv_M = _get_face_affine(face, input_size)
    if M is None:
        return frame

    h, w = frame.shape[:2]
    corners = np.array(
        [[0, 0], [input_size - 1, 0],
         [input_size - 1, input_size - 1], [0, input_size - 1]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    full_corners = cv2.transform(corners, inv_M).reshape(-1, 2)

    x0 = max(0, int(np.floor(full_corners[:, 0].min())) - 4)
    y0 = max(0, int(np.floor(full_corners[:, 1].min())) - 4)
    x1 = min(w, int(np.ceil(full_corners[:, 0].max())) + 5)
    y1 = min(h, int(np.ceil(full_corners[:, 1].max())) + 5)
    if x1 <= x0 or y1 <= y0:
        return frame

    roi_w = x1 - x0
    roi_h = y1 - y0
    roi_inv_M = inv_M.copy()
    roi_inv_M[0, 2] -= x0
    roi_inv_M[1, 2] -= y0

    mask = np.ones((input_size, input_size), dtype=np.float32)
    border = max(1, input_size // 16)
    mask[:border, :] = np.linspace(0, 1, border)[:, np.newaxis]
    mask[-border:, :] = np.linspace(1, 0, border)[:, np.newaxis]
    mask[:, :border] = np.minimum(mask[:, :border], np.linspace(0, 1, border)[np.newaxis, :])
    mask[:, -border:] = np.minimum(mask[:, -border:], np.linspace(1, 0, border)[np.newaxis, :])

    warped_enhanced = cv2.warpAffine(
        _cached_enhanced, roi_inv_M, (roi_w, roi_h),
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask, roi_inv_M, (roi_w, roi_h),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )

    roi_frame = frame[y0:y1, x0:x1]
    mask_3ch = warped_mask[:, :, np.newaxis]
    blended = (
        warped_enhanced.astype(np.float32) * mask_3ch +
        roi_frame.astype(np.float32) * (1.0 - mask_3ch)
    )
    frame[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return frame


def enhance_face_onnx(
    frame: np.ndarray,
    face: Any,
    session: onnxruntime.InferenceSession,
    input_size: int,
) -> np.ndarray:
    """Enhance a single face in the frame using an ONNX face restoration model."""
    t0 = time.perf_counter()
    M, inv_M = _get_face_affine(face, input_size)
    t1 = time.perf_counter()
    if M is None:
        return frame

    face_crop = cv2.warpAffine(
        frame, M, (input_size, input_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    t2 = time.perf_counter()

    blob = preprocess_face(face_crop, input_size)
    t3 = time.perf_counter()

    with THREAD_SEMAPHORE:
        input_name = session.get_inputs()[0].name
        output = run_inference(session, input_name, blob)
    t4 = time.perf_counter()

    enhanced = postprocess_face(output)
    t5 = time.perf_counter()

    # Moderate restoration strength: reduces the over-sharp GPEN look while
    # keeping most of the restoration benefit.
    GPEN_STRENGTH = 0.85 if input_size >= 512 else 0.65
    enhanced = cv2.addWeighted(
        enhanced, GPEN_STRENGTH,
        face_crop, 1.0 - GPEN_STRENGTH, 0.0,
    )

    # Cache the aligned restored face so skipped live frames can reuse it.
    global _cached_enhanced, _cached_face_bbox
    _cached_enhanced = enhanced.copy()
    if hasattr(face, "bbox") and face.bbox is not None:
        _cached_face_bbox = np.asarray(face.bbox, dtype=np.float32).copy()
    else:
        _cached_face_bbox = None

    # Create mask for blending (feathered edges)
    mask = np.ones((input_size, input_size), dtype=np.float32)
    border = max(1, input_size // 16)
    mask[:border, :] = np.linspace(0, 1, border)[:, np.newaxis]
    mask[-border:, :] = np.linspace(1, 0, border)[:, np.newaxis]
    mask[:, :border] = np.minimum(mask[:, :border], np.linspace(0, 1, border)[np.newaxis, :])
    mask[:, -border:] = np.minimum(mask[:, -border:], np.linspace(1, 0, border)[np.newaxis, :])

    # ROI optimization: preserve the exact same affine geometry, but avoid
    # warping/blending the entire 1280x960 frame. We first transform the
    # enhanced 256x256 corners into full-frame coordinates and process only
    # the bounding ROI (with a small safety margin).
    h, w = frame.shape[:2]

    corners = np.array(
        [[0, 0], [input_size - 1, 0],
         [input_size - 1, input_size - 1], [0, input_size - 1]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    full_corners = cv2.transform(corners, inv_M).reshape(-1, 2)

    x0 = max(0, int(np.floor(full_corners[:, 0].min())) - 4)
    y0 = max(0, int(np.floor(full_corners[:, 1].min())) - 4)
    x1 = min(w, int(np.ceil(full_corners[:, 0].max())) + 5)
    y1 = min(h, int(np.ceil(full_corners[:, 1].max())) + 5)

    if x1 <= x0 or y1 <= y0:
        return frame

    roi_w = x1 - x0
    roi_h = y1 - y0

    # Translate destination coordinates so warpAffine writes directly into ROI.
    roi_inv_M = inv_M.copy()
    roi_inv_M[0, 2] -= x0
    roi_inv_M[1, 2] -= y0

    warped_enhanced = cv2.warpAffine(
        enhanced, roi_inv_M, (roi_w, roi_h),
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask, roi_inv_M, (roi_w, roi_h),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )

    roi_frame = frame[y0:y1, x0:x1]
    mask_3ch = warped_mask[:, :, np.newaxis]

    # Blend only the ROI instead of the complete 1280x960 frame.
    blended = (
        warped_enhanced.astype(np.float32) * mask_3ch +
        roi_frame.astype(np.float32) * (1.0 - mask_3ch)
    )
    frame[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    result = frame
    t6 = time.perf_counter()

    # Lightweight profiling: same processing math, less console I/O.
    with _GPEN_PROFILE_LOCK:
        _GPEN_PROFILE["n"] += 1
        _GPEN_PROFILE["affine"] += (t1 - t0) * 1000
        _GPEN_PROFILE["crop"] += (t2 - t1) * 1000
        _GPEN_PROFILE["pre"] += (t3 - t2) * 1000
        _GPEN_PROFILE["infer"] += (t4 - t3) * 1000
        _GPEN_PROFILE["post"] += (t5 - t4) * 1000
        _GPEN_PROFILE["warp_blend"] += (t6 - t5) * 1000
        n = _GPEN_PROFILE["n"]
        if n % 100 == 0:
            print(
                "[GPEN DETAIL] "
                f"n={n} | infer={_GPEN_PROFILE['infer']/n:.2f}ms | "
                f"warp+blend={_GPEN_PROFILE['warp_blend']/n:.2f}ms | "
                f"total={(t6-t0)*1000:.2f}ms"
            )

    return result
