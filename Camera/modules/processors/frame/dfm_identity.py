"""Runtime for the locally trained DeepFaceLab AMP identity model.

The model is a DFM/ONNX graph trained on the user's selected source identity.
It is deliberately separate from INSwapper: the normal swapper remains the
default and this runtime is loaded only when the trained-identity toggle is on.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

import modules.globals


_SESSION: Optional[ort.InferenceSession] = None
_SESSION_LOCK = threading.Lock()
_MASK_CACHE: dict[int, np.ndarray] = {}

# DeepFaceLab's whole-face alignment template.  It matches the ``wf`` model
# trained for this project and deliberately includes the lower jaw/beard.
_DFL_WHOLE_FACE = np.array(
    [[0.35342266, 0.39285716], [0.62797622, 0.39285716],
     [0.48660713, 0.54017860], [0.38839287, 0.68750011],
     [0.59821427, 0.68750011]], dtype=np.float32)


def model_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(root, "models", "sourceface_AMP_model.dfm")


def is_available() -> bool:
    return os.path.isfile(model_path())


def _providers():
    configured = list(getattr(modules.globals, "execution_providers", []) or [])
    providers = []
    for provider in configured:
        if provider in ort.get_available_providers():
            providers.append(provider)
    if "CUDAExecutionProvider" in ort.get_available_providers() and "CUDAExecutionProvider" not in providers:
        providers.insert(0, "CUDAExecutionProvider")
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return providers


def get_session() -> ort.InferenceSession:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            path = model_path()
            if not os.path.isfile(path):
                raise FileNotFoundError("Обученная DFM-модель не найдена")
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = 1
            _SESSION = ort.InferenceSession(path, sess_options=options, providers=_providers())
            print(f"[DLC.DFM] identity model loaded | providers={_SESSION.get_providers()}")
    return _SESSION


def clear_session() -> None:
    global _SESSION
    with _SESSION_LOCK:
        _SESSION = None


def _edge_mask(size: int) -> np.ndarray:
    cached = _MASK_CACHE.get(size)
    if cached is not None:
        return cached
    mask = np.zeros((size, size), np.float32)
    # Wide, lower oval: retains beard/chin but avoids a rectangular paste.
    cv2.ellipse(mask, (size // 2, int(size * .51)),
                (int(size * .49), int(size * .60)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(2.0, size / 28.0))
    _MASK_CACHE[size] = mask
    return mask


def _paste_roi(frame: np.ndarray, aligned: np.ndarray, alpha: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Paste only the affine ROI, avoiding a full-frame warp every live frame."""
    h, w = frame.shape[:2]
    size = aligned.shape[0]
    inverse = cv2.invertAffineTransform(matrix)
    corners = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    mapped = cv2.transform(corners[None, ...], inverse)[0]
    x1 = max(0, int(np.floor(mapped[:, 0].min())))
    y1 = max(0, int(np.floor(mapped[:, 1].min())))
    x2 = min(w, int(np.ceil(mapped[:, 0].max())) + 1)
    y2 = min(h, int(np.ceil(mapped[:, 1].max())) + 1)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return frame
    roi_matrix = inverse.copy()
    roi_matrix[0, 2] -= x1
    roi_matrix[1, 2] -= y1
    rw, rh = x2 - x1, y2 - y1
    aligned_roi = cv2.warpAffine(aligned, roi_matrix, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    alpha_roi = cv2.warpAffine(alpha, roi_matrix, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=0).clip(0, 1)[..., None]
    out = frame.copy()
    target = out[y1:y2, x1:x2].astype(np.float32)
    out[y1:y2, x1:x2] = np.clip(aligned_roi.astype(np.float32) * alpha_roi + target * (1.0 - alpha_roi), 0, 255).astype(np.uint8)
    return out


def swap(frame: np.ndarray, target_face) -> np.ndarray:
    """Apply trained identity to one detected face; return original on failure."""
    if target_face is None or getattr(target_face, "kps", None) is None:
        return frame
    try:
        session = get_session()
        size = int(session.get_inputs()[0].shape[1])
        points = np.asarray(target_face.kps, dtype=np.float32)
        if points.shape != (5, 2):
            return frame
        matrix, _ = cv2.estimateAffinePartial2D(points, _DFL_WHOLE_FACE * size, method=cv2.RANSAC, ransacReprojThreshold=100)
        if matrix is None:
            return frame
        crop = cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        inputs = {
            "in_face:0": crop.astype(np.float32)[None] / 255.0,
            "morph_value:0": np.array([float(getattr(modules.globals, "trained_identity_morph", 1.0))], np.float32),
        }
        target_mask, generated, source_mask = session.run(None, inputs)
        generated = np.clip(generated[0] * 255.0, 0, 255).astype(np.uint8)
        # Both masks are meaningful: their intersection prevents regions that
        # the model cannot reconstruct from replacing the live camera.
        alpha = np.minimum(target_mask[0, ..., 0], source_mask[0, ..., 0]).clip(0, 1)
        alpha *= _edge_mask(size)
        opacity = max(0.0, min(1.0, float(getattr(modules.globals, "opacity", 1.0))))
        alpha *= opacity
        return _paste_roi(frame, generated, alpha, matrix)
    except Exception as exc:
        print(f"[DLC.DFM] trained identity frame fallback: {exc}")
        return frame
