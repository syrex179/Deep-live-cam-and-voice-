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
    """Run enhancement inference through ONNX Runtime's stable memory arena.

    A new CUDA I/O binding and OrtValue used to be allocated for every webcam
    frame.  That path was not warmed by ``warmup_session`` and periodically
    forced CUDA allocation/synchronisation spikes (visible as 30–110 ms GPEN
    frames).  ``session.run`` reuses ONNX Runtime's allocation arena and has
    the same model output, with steadier frame timing for LIVE.
    """
    return session.run(None, {input_name: input_tensor})[0]


def create_onnx_session(model_path: str) -> onnxruntime.InferenceSession:
    """Create an ONNX Runtime session with optimised provider config.

    On Apple Silicon, applies CoreML graph optimizations (Pad decomposition,
    Shape/Gather folding, Split decomposition) to reduce CPU↔ANE partition
    boundaries.
    """
    if IS_APPLE_SILICON:
        from modules.onnx_optimize import optimize_for_coreml
        # Infer input shape from the model for Shape/Gather folding
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

    providers = build_provider_config()
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session = onnxruntime.InferenceSession(
        model_path, sess_options=session_options, providers=providers,
    )
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

    if (_cached_enhanced is None or
            _cached_enhanced.shape[:2] != (input_size, input_size)):
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
    # Fast uint16 blend: avoids large float32 temporary arrays for live ROI.
    alpha16 = np.clip(warped_mask * 255.0, 0, 255).astype(np.uint16)
    inv16 = 255 - alpha16
    enhanced16 = warped_enhanced.astype(np.uint16)
    roi16 = roi_frame.astype(np.uint16)
    blended16 = (
        enhanced16 * alpha16[:, :, None] +
        roi16 * inv16[:, :, None] +
        127
    ) // 255
    frame[y0:y1, x0:x1] = blended16.astype(np.uint8)
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

    try:
        scale=max(0.60,min(1.40,float(getattr(modules.globals,'face_scale',1.0))))
        angle=max(-30.0,min(30.0,float(getattr(modules.globals,'face_rotation',0.0))))
        fx=max(-100.0,min(100.0,float(getattr(modules.globals,'face_offset_x',0.0))))
        fy=max(-100.0,min(100.0,float(getattr(modules.globals,'face_offset_y',0.0))))
        if scale!=1.0 or abs(angle)>1e-6 or fx!=0.0 or fy!=0.0:
            bbox=np.asarray(getattr(face,'bbox',None),dtype=np.float32).reshape(-1)
            if bbox.size>=4 and np.all(np.isfinite(bbox)):
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
                inv_M=(T@H)[:2]
                M=cv2.invertAffineTransform(inv_M)
    except Exception:
        pass

    face_crop = cv2.warpAffine(
        frame, M, (input_size, input_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    t2 = time.perf_counter()

    blob = preprocess_face(face_crop, input_size)
    t3 = time.perf_counter()

    with THREAD_SEMAPHORE:
        model_inputs = session.get_inputs()
        input_name = model_inputs[0].name
        # Some high-resolution FaceFusion GPEN exports expose an optional
        # restoration-weight input.  Feed it explicitly; ordinary GPEN 256/512
        # exports remain on the zero-copy single-input path.
        if len(model_inputs) == 1:
            output = run_inference(session, input_name, blob)
        else:
            feeds = {input_name: blob}
            for model_input in model_inputs[1:]:
                if model_input.name.lower() == "weight":
                    feeds[model_input.name] = np.array([1.0], dtype=np.float64)
                else:
                    raise RuntimeError(
                        f"Unsupported GPEN model input: {model_input.name}"
                    )
            output = session.run(None, feeds)[0]
    t4 = time.perf_counter()

    enhanced = postprocess_face(output)
    t5 = time.perf_counter()

    # Performance Mode is a real processing control.
    # Keep more of the swapped face's fine identity texture in Quality mode.
    # A very high restoration mix acts like beauty smoothing and removes
    # dimples and small facial relief from the supplied source.
    perf_mode = getattr(modules.globals, "performance_mode", "Balanced")
    GPEN_STRENGTH = {
        # The previous Quality value acted like a beauty filter and erased
        # source-identity relief (dimples, beard texture and fine wrinkles).
        # These values still repair swap artifacts but retain the face itself.
        "Quality": 0.70,
        "Balanced": 0.60,
        "Performance": 0.45,
    }.get(str(perf_mode), 0.65)
    enhanced = cv2.addWeighted(
        enhanced, GPEN_STRENGTH,
        face_crop, 1.0 - GPEN_STRENGTH, 0.0,
    )

    # GPEN is good at removing artifacts, but it also removes identity cues
    # such as dimples and fine expression texture.  Restore only the current
    # pre-GPEN high-frequency residual in aligned face space, so it follows
    # the face and mouth on every frame rather than replaying an old image.
    texture = max(0.0, min(100.0, float(getattr(modules.globals, "texture_preservation", 0.0))))
    if texture > 0.0:
        base = cv2.GaussianBlur(face_crop, (0, 0), 2.0)
        residual = face_crop.astype(np.int16) - base.astype(np.int16)
        amount = (texture / 100.0) * 0.60
        enhanced = np.clip(
            enhanced.astype(np.float32) + residual.astype(np.float32) * amount,
            0, 255,
        ).astype(np.uint8)


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

    # Integer ROI blend avoids two large float32 3-channel allocations.  It
    # is visually equivalent for the 8-bit output and is noticeably cheaper
    # when the webcam face occupies a large part of a 1080p frame.
    alpha16 = np.clip(warped_mask * 255.0, 0, 255).astype(np.uint16)
    inv16 = 255 - alpha16
    blended = (
        warped_enhanced.astype(np.uint16) * alpha16[:, :, None]
        + roi_frame.astype(np.uint16) * inv16[:, :, None]
    ) // 255
    frame[y0:y1, x0:x1] = blended.astype(np.uint8)
    result = frame
    t6 = time.perf_counter()

    # Accumulate and print every 20 GPEN calls. No processing behavior changes.
    with _GPEN_PROFILE_LOCK:
        _GPEN_PROFILE["n"] += 1
        _GPEN_PROFILE["affine"] += (t1 - t0) * 1000
        _GPEN_PROFILE["crop"] += (t2 - t1) * 1000
        _GPEN_PROFILE["pre"] += (t3 - t2) * 1000
        _GPEN_PROFILE["infer"] += (t4 - t3) * 1000
        _GPEN_PROFILE["post"] += (t5 - t4) * 1000
        _GPEN_PROFILE["warp_blend"] += (t6 - t5) * 1000
        n = _GPEN_PROFILE["n"]
        if n % 20 == 0:
            print(
                "[GPEN DETAIL] "
                f"n={n} | affine={_GPEN_PROFILE['affine']/n:.2f}ms | "
                f"crop={_GPEN_PROFILE['crop']/n:.2f}ms | "
                f"pre={_GPEN_PROFILE['pre']/n:.2f}ms | "
                f"infer={_GPEN_PROFILE['infer']/n:.2f}ms | "
                f"post={_GPEN_PROFILE['post']/n:.2f}ms | "
                f"warp+blend={_GPEN_PROFILE['warp_blend']/n:.2f}ms | "
                f"ROI={roi_w}x{roi_h} | total={(t6-t0)*1000:.2f}ms"
            )

    return result
