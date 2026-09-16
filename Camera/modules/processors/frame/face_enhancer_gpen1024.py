"""Optional GPEN-BFR-1024 face enhancer for high-detail, slower live output."""

from typing import Any, List
import os
import threading

import modules.globals
import modules.processors.frame.core
from modules import imread_unicode, imwrite_unicode
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
from modules.utilities import is_image, is_video
from modules.processors.frame._onnx_enhancer import (
    create_onnx_session,
    warmup_session,
    enhance_face_onnx,
    reuse_last_enhancement,
)

NAME = "DLC.FACE-ENHANCER-GPEN1024"
INPUT_SIZE = 1024
# FaceFusion publishes this ONNX export with the exact hash listed in its model
# registry.  It is downloaded only when the user chooses the slow 1024 mode.
MODEL_URL = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_1024.onnx?download=true"
MODEL_FILE = "gpen_bfr_1024.onnx"

ENHANCER = None
THREAD_LOCK = threading.Lock()
_abs_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_abs_dir))), "models")


def _model_path() -> str:
    return os.path.join(MODELS_DIR, MODEL_FILE)


def pre_check() -> bool:
    if not os.path.exists(_model_path()):
        update_status("GPEN-1024 is not installed yet. Choose it once to download the model.", NAME)
    return True


def pre_start() -> bool:
    return bool(is_image(modules.globals.target_path) or is_video(modules.globals.target_path))


def get_enhancer() -> Any:
    global ENHANCER
    with THREAD_LOCK:
        if ENHANCER is None:
            model_path = _model_path()
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"{MODEL_FILE} is missing. Download it from the optional GPEN-1024 installer."
                )
            print(f"{NAME}: Loading GPEN-BFR-1024 via ONNX Runtime (slow mode).")
            ENHANCER = create_onnx_session(model_path)
            warmup_session(ENHANCER)
    return ENHANCER


def enhance_face(temp_frame: Frame, face: Face) -> Frame:
    try:
        return enhance_face_onnx(temp_frame, face, get_enhancer(), INPUT_SIZE)
    except Exception as error:
        print(f"{NAME}: Error during face enhancement: {error}")
        return temp_frame


def reuse_face(temp_frame: Frame, face: Face) -> Frame:
    return reuse_last_enhancement(temp_frame, face, INPUT_SIZE)


def process_frame(source_face: Face | None, temp_frame: Frame, detected_faces=None) -> Frame:
    face = detected_faces[0] if detected_faces else get_one_face(temp_frame)
    return enhance_face(temp_frame, face) if face is not None else temp_frame


def process_frame_v2(temp_frame: Frame) -> Frame:
    return process_frame(None, temp_frame)


def process_frames(source_path: str | None, temp_frame_paths: List[str], progress: Any = None) -> None:
    for path in temp_frame_paths:
        frame = imread_unicode(path)
        if frame is not None:
            imwrite_unicode(path, process_frame(None, frame))
        if progress:
            progress.update(1)


def process_image(source_path: str | None, target_path: str, output_path: str) -> None:
    frame = imread_unicode(target_path)
    if frame is not None:
        imwrite_unicode(output_path, process_frame(None, frame))


def process_video(source_path: str | None, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
