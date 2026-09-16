"""GPEN-BFR-256 face enhancer — ONNX-based face restoration at 256x256."""

from typing import Any, List
import os
import threading

import modules.globals
import modules.processors.frame.core
from modules import imread_unicode, imwrite_unicode
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
from modules.utilities import (
    is_image,
    is_video,
)
from modules.processors.frame._onnx_enhancer import (
    create_onnx_session,
    warmup_session,
    enhance_face_onnx,
    reuse_last_enhancement,
)
from modules.processors.frame.gpen512_trt_engine import (
    load_engine as load_trt_engine,
    warmup_engine as warmup_trt_engine,
    enhance_face_trt,
)

NAME = "DLC.FACE-ENHANCER-GPEN256"
INPUT_SIZE = 256
MODEL_MIRROR_URL = "https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/GPEN-BFR/GPEN-BFR-256.onnx"
MODEL_FILE = "GPEN-BFR-256.onnx"
ENGINE_FILE = "GPEN-BFR-256_fp16.engine"

ENHANCER = None
THREAD_LOCK = threading.Lock()

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)


def _obtain_model():
    from modules.model_downloader import ensure_model

    model_path = ensure_model(MODEL_FILE)
    if model_path is not None:
        return model_path

    update_status(f"Retrying {MODEL_FILE} from the mirror...", NAME)
    from modules.utilities import conditional_download

    try:
        conditional_download(models_dir, [MODEL_MIRROR_URL])
    except Exception as error:
        update_status(f"Mirror download failed: {error}", NAME)
        return None
    fallback = os.path.join(models_dir, MODEL_FILE)
    return fallback if os.path.exists(fallback) else None


def pre_check() -> bool:
    if _obtain_model() is None:
        update_status(
            f"Could not obtain {MODEL_FILE}. Place it in the models folder "
            "manually or check your internet connection.",
            NAME,
        )
        return False
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_enhancer() -> Any:
    global ENHANCER
    with THREAD_LOCK:
        if ENHANCER is None:
            model_path = _obtain_model()
            if model_path is None:
                raise FileNotFoundError(
                    f"Model file not found: {os.path.join(models_dir, MODEL_FILE)}"
                )
            engine_path = os.path.join(models_dir, ENGINE_FILE)
            if (
                bool(getattr(modules.globals, "use_gpen256_tensorrt", False))
                and os.path.isfile(engine_path)
            ):
                try:
                    print(f"{NAME}: Loading TensorRT FP16 engine from {engine_path}")
                    load_trt_engine(engine_path)
                    warmup_trt_engine()
                    ENHANCER = "DIRECT_TRT_ENGINE"
                    print(f"{NAME}: TensorRT FP16 engine loaded successfully.")
                except Exception as exc:
                    print(f"{NAME}: TensorRT engine failed ({exc}); using ONNX/CUDA.")
                    ENHANCER = create_onnx_session(model_path)
                    warmup_session(ENHANCER)
            else:
                print(f"{NAME}: Loading validated ONNX/CUDA model from {model_path}")
                ENHANCER = create_onnx_session(model_path)
                warmup_session(ENHANCER)
                print(f"{NAME}: Model loaded successfully.")
    return ENHANCER


def enhance_face(temp_frame: Frame, face: Face) -> Frame:
    try:
        session = get_enhancer()
    except Exception as e:
        print(f"{NAME}: {e}")
        return temp_frame
    try:
        if session == "DIRECT_TRT_ENGINE":
            result, stats = enhance_face_trt(temp_frame, face, INPUT_SIZE)
            count = getattr(enhance_face, "_trt_count", 0) + 1
            enhance_face._trt_count = count
            if count % 100 == 0:
                print(
                    f"[GPEN256 TRT] infer={stats['infer']:.2f}ms | "
                    f"blend={stats['warp_blend']:.2f}ms | total={stats['total']:.2f}ms",
                    flush=True,
                )
            return result
        return enhance_face_onnx(temp_frame, face, session, INPUT_SIZE)
    except Exception as e:
        print(f"{NAME}: Error during face enhancement: {e}")
        return temp_frame


def reuse_face(temp_frame: Frame, face: Face) -> Frame:
    try:
        return reuse_last_enhancement(temp_frame, face, INPUT_SIZE)
    except Exception as e:
        print(f"{NAME}: Error during cached face reuse: {e}")
        return temp_frame


def process_frame(source_face: Face | None, temp_frame: Frame, detected_faces=None) -> Frame:
    if detected_faces:
        target_face = detected_faces[0]
    else:
        target_face = get_one_face(temp_frame)
    if target_face is None:
        return temp_frame
    return enhance_face(temp_frame, target_face)


def process_frame_v2(temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame, target_face)
    return temp_frame


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = imread_unicode(temp_frame_path)
        if temp_frame is None:
            if progress:
                progress.update(1)
            continue
        result = process_frame(None, temp_frame)
        imwrite_unicode(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str | None, target_path: str, output_path: str) -> None:
    target_frame = imread_unicode(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: Failed to read target image {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    imwrite_unicode(output_path, result_frame)
    print(f"{NAME}: Enhanced image saved to {output_path}")


def process_video(source_path: str | None, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
