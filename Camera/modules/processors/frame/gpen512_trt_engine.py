"""Direct TensorRT runtime for a prebuilt GPEN-BFR-512 engine.

Visual blend v23: no rectangular/black boundary; full-quality center with soft face-shaped feather.

This module intentionally does not use ONNX Runtime to load the .engine file.
The engine is deserialized directly by TensorRT Runtime, so no engine build is
performed when Deep-Live-Cam starts.
"""

import os
import threading
import time
from typing import Any

import cv2
import numpy as np

import modules.globals
from modules.processors.frame._onnx_enhancer import (
    _get_face_affine,
    preprocess_face,
    postprocess_face,
)

TRT_ENGINE = None
TRT_CONTEXT = None
TRT_INPUT_NAME = None
TRT_OUTPUT_NAME = None
TRT_OUTPUT_SHAPE = None
TRT_OUTPUT_DTYPE = None
TRT_STREAM = None
TRT_INPUT_BUFFER = None
TRT_OUTPUT_BUFFER = None
TRT_LOCK = threading.Lock()


def _torch_dtype_from_numpy(np_dtype):
    import torch

    if np_dtype == np.float32:
        return torch.float32
    if np_dtype == np.float16:
        return torch.float16
    if np.int32 == np_dtype:
        return torch.int32
    if np.int64 == np_dtype:
        return torch.int64
    if np.uint8 == np_dtype:
        return torch.uint8
    raise TypeError(f"Unsupported TensorRT output dtype: {np_dtype}")


def load_engine(engine_path: str):
    global TRT_ENGINE, TRT_CONTEXT, TRT_INPUT_NAME
    global TRT_OUTPUT_NAME, TRT_OUTPUT_SHAPE, TRT_OUTPUT_DTYPE, TRT_STREAM
    global TRT_INPUT_BUFFER, TRT_OUTPUT_BUFFER

    with TRT_LOCK:
        if TRT_CONTEXT is not None:
            return TRT_CONTEXT

        import tensorrt as trt
        import torch

        # Dedicated non-default stream avoids TensorRT's default-stream
        # synchronization overhead warned about by enqueueV3().
        TRT_STREAM = torch.cuda.Stream()

        logger = trt.Logger(trt.Logger.WARNING)
        print(f"[GPEN-TRT] Loading prebuilt engine: {engine_path}")

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError("TensorRT could not deserialize GPEN-512 engine.")

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT could not create GPEN-512 execution context.")

        input_names = []
        output_names = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)

        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                f"Unexpected GPEN engine IO: inputs={input_names}, outputs={output_names}"
            )

        input_name = input_names[0]
        output_name = output_names[0]
        input_shape = tuple(engine.get_tensor_shape(input_name))

        if any(int(x) <= 0 for x in input_shape):
            input_shape = (1, 3, 512, 512)
            context.set_input_shape(input_name, input_shape)

        output_shape = tuple(context.get_tensor_shape(output_name))
        if any(int(x) <= 0 for x in output_shape):
            raise RuntimeError(f"Unresolved GPEN output shape: {output_shape}")

        np_dtype = trt.nptype(engine.get_tensor_dtype(output_name))

        TRT_ENGINE = engine
        TRT_CONTEXT = context
        TRT_INPUT_NAME = input_name
        TRT_OUTPUT_NAME = output_name
        TRT_OUTPUT_SHAPE = tuple(int(x) for x in output_shape)
        TRT_OUTPUT_DTYPE = _torch_dtype_from_numpy(np_dtype)

        # Persistent GPU buffers eliminate per-frame CUDA allocator churn.
        TRT_INPUT_BUFFER = torch.empty(
            input_shape, dtype=torch.float32, device="cuda"
        )
        TRT_OUTPUT_BUFFER = torch.empty(
            TRT_OUTPUT_SHAPE, dtype=TRT_OUTPUT_DTYPE, device="cuda"
        )

        print(
            "[GPEN-TRT] Ready | "
            f"input={input_name}{input_shape} | "
            f"output={output_name}{TRT_OUTPUT_SHAPE} | "
            f"dtype={np_dtype}"
        )
        return TRT_CONTEXT


def warmup_engine():
    import torch

    if TRT_CONTEXT is None:
        raise RuntimeError("GPEN TensorRT engine is not loaded.")

    with TRT_LOCK:
        inp = TRT_INPUT_BUFFER
        out = TRT_OUTPUT_BUFFER
        stream = TRT_STREAM

        if inp is None or out is None or stream is None:
            raise RuntimeError("GPEN TensorRT buffers/stream are not initialized.")

        TRT_CONTEXT.set_tensor_address(TRT_INPUT_NAME, int(inp.data_ptr()))
        TRT_CONTEXT.set_tensor_address(TRT_OUTPUT_NAME, int(out.data_ptr()))

        for _ in range(3):
            ok = TRT_CONTEXT.execute_async_v3(
                stream_handle=int(stream.cuda_stream)
            )
            if not ok:
                raise RuntimeError("TensorRT warmup execution failed.")

        stream.synchronize()
        print("[GPEN-TRT] Warmup complete.")


def infer(input_tensor: np.ndarray) -> np.ndarray:
    import torch

    if TRT_CONTEXT is None:
        raise RuntimeError("GPEN TensorRT engine is not loaded.")

    stream = TRT_STREAM
    inp = TRT_INPUT_BUFFER
    out = TRT_OUTPUT_BUFFER

    if stream is None or inp is None or out is None:
        raise RuntimeError("GPEN TensorRT buffers/stream are not initialized.")

    # Live pipeline is single-face/single-frame. Keep execution on the
    # dedicated non-default CUDA stream without an extra Python lock on the
    # hot path.
    src = torch.from_numpy(np.ascontiguousarray(input_tensor))

    with torch.cuda.stream(stream):
        inp.copy_(src, non_blocking=True)
        TRT_CONTEXT.set_tensor_address(TRT_INPUT_NAME, int(inp.data_ptr()))
        TRT_CONTEXT.set_tensor_address(TRT_OUTPUT_NAME, int(out.data_ptr()))

        ok = TRT_CONTEXT.execute_async_v3(
            stream_handle=int(stream.cuda_stream)
        )
        if not ok:
            raise RuntimeError("TensorRT GPEN execution failed.")

    # CPU/OpenCV postprocess needs the completed output.
    stream.synchronize()
    result = out.detach().cpu().numpy().copy()

    del src
    return result


_LAST_ENHANCED = None
_LAST_FACE_BBOX = None

def _cache_enhanced(enhanced, face):
    global _LAST_ENHANCED, _LAST_FACE_BBOX
    _LAST_ENHANCED = enhanced.copy()
    bbox = getattr(face, "bbox", None)
    _LAST_FACE_BBOX = None if bbox is None else np.asarray(bbox, dtype=np.float32).copy()

def reuse_last_enhancement(frame: np.ndarray, face: Any, input_size: int = 512):
    """Reuse the last 512px restoration only when the detected face barely moved.
    Uses the CURRENT affine, so the cached aligned face follows the current face
    without replaying an old full frame. Returns (frame, True) on safe reuse.
    """
    global _LAST_ENHANCED, _LAST_FACE_BBOX
    if _LAST_ENHANCED is None or _LAST_FACE_BBOX is None:
        return frame, False
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return frame, False
    bbox = np.asarray(bbox, dtype=np.float32)
    old = _LAST_FACE_BBOX
    oc = (old[:2] + old[2:4]) * 0.5
    nc = (bbox[:2] + bbox[2:4]) * 0.5
    old_size = max(1.0, float(np.mean(old[2:4] - old[:2])))
    new_size = max(1.0, float(np.mean(bbox[2:4] - bbox[:2])))
    shift = float(np.linalg.norm(nc - oc))
    scale_delta = abs(new_size / old_size - 1.0)
    # At 1280x960 this is deliberately conservative: if the head moves quickly,
    # force a fresh GPEN inference instead of letting the old face slide.
    if shift > max(14.0, new_size * 0.045) or scale_delta > 0.055:
        return frame, False
    M, inv_M = _get_face_affine(face, input_size)
    if M is None:
        return frame, False
    result = _blend_result(frame, _LAST_ENHANCED, None, inv_M, input_size)
    return result, True

def _blend_result(frame, enhanced, face_crop, inv_M, input_size):
    h, w = frame.shape[:2]

    # Match the regular GPEN-256 blend exactly: a 16px soft border in aligned
    # space.  This keeps a TensorRT switch visually neutral for live users.
    mask = np.ones((input_size, input_size), dtype=np.float32)
    border = max(1, input_size // 16)
    mask[:border, :] = np.linspace(0, 1, border)[:, np.newaxis]
    mask[-border:, :] = np.linspace(1, 0, border)[:, np.newaxis]
    mask[:, :border] = np.minimum(mask[:, :border], np.linspace(0, 1, border)[np.newaxis, :])
    mask[:, -border:] = np.minimum(mask[:, -border:], np.linspace(1, 0, border)[np.newaxis, :])

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

    warped_enhanced = cv2.warpAffine(
        enhanced, roi_inv_M, (roi_w, roi_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    warped_mask = cv2.warpAffine(
        mask, roi_inv_M, (roi_w, roi_h),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )

    roi_frame = frame[y0:y1, x0:x1]
    alpha16 = np.clip(warped_mask * 255.0, 0, 255).astype(np.uint16)
    inv16 = 255 - alpha16
    blended = (
        warped_enhanced.astype(np.uint16) * alpha16[:, :, None]
        + roi_frame.astype(np.uint16) * inv16[:, :, None]
    ) // 255
    frame[y0:y1, x0:x1] = blended.astype(np.uint8)
    return frame


def enhance_face_trt(frame: np.ndarray, face: Any, input_size: int = 512) -> np.ndarray:
    t0 = time.perf_counter()
    M, inv_M = _get_face_affine(face, input_size)
    t1 = time.perf_counter()

    if M is None:
        return frame

    face_crop = cv2.warpAffine(
        frame, M, (input_size, input_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    t2 = time.perf_counter()

    blob = preprocess_face(face_crop, input_size)
    t3 = time.perf_counter()

    output = infer(blob)
    t4 = time.perf_counter()

    enhanced = postprocess_face(output)
    t5 = time.perf_counter()

    # Preserve the same profile strength used by the regular CUDA GPEN path.
    perf_mode = getattr(modules.globals, "performance_mode", "Balanced")
    strength = {"Quality": 0.85, "Balanced": 0.65, "Performance": 0.45}.get(
        str(perf_mode), 0.65
    )
    enhanced = cv2.addWeighted(
        enhanced, strength,
        face_crop, 1.0 - strength, 0.0,
    )
    # Restore moving microtexture from the current swapped face after GPEN.
    # This preserves dimples and small expression relief without transferring
    # a frozen texture from an earlier video frame.
    texture = max(0.0, min(100.0, float(getattr(modules.globals, "texture_preservation", 0.0))))
    if texture > 0.0:
        base = cv2.GaussianBlur(face_crop, (0, 0), 2.0)
        residual = face_crop.astype(np.int16) - base.astype(np.int16)
        amount = (texture / 100.0) * 0.60
        enhanced = np.clip(
            enhanced.astype(np.float32) + residual.astype(np.float32) * amount,
            0, 255,
        ).astype(np.uint8)
    _cache_enhanced(enhanced, face)

    result = _blend_result(frame, enhanced, face_crop, inv_M, input_size)
    t6 = time.perf_counter()

    return result, {
        "affine": (t1 - t0) * 1000,
        "crop": (t2 - t1) * 1000,
        "pre": (t3 - t2) * 1000,
        "infer": (t4 - t3) * 1000,
        "post": (t5 - t4) * 1000,
        "warp_blend": (t6 - t5) * 1000,
        "total": (t6 - t0) * 1000,
    }
