"""Portrait-aware virtual background compositing for the live camera."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import modules.globals as dlc_globals


_ROOT = Path(__file__).resolve().parents[1]
# U2NetP is the compact variant that remains practical beside the face swap
# and GPEN models on an 8 GB RTX 4060.  MODNet was measured at about 100 ms
# per matte here, which reduced LIVE to 8–11 FPS.
_MODEL_PATH = _ROOT / "facefusion" / ".assets" / "models" / "u2netp.onnx"


def _read_image(path: str) -> Optional[np.ndarray]:
    """Read a Unicode Windows path without relying on OpenCV's ANSI API."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


class VirtualBackground:
    """Keeps one MODNet session and composites a cover-cropped still image.

    U2NetP is a compact foreground matte: it retains the person rather than
    making a face-only cutout.  All heavy resources are lazy-loaded, so
    regular live camera use pays no extra startup or frame-processing cost.
    """

    def __init__(self) -> None:
        self._session = None
        self._input_name: Optional[str] = None
        self._background_key: Optional[Tuple[str, int, int]] = None
        self._background: Optional[np.ndarray] = None
        self._matte: Optional[np.ndarray] = None
        self._matte_size: Optional[Tuple[int, int]] = None
        self._frame_number = 0
        self.last_error: Optional[str] = None

    def _ensure_session(self) -> bool:
        if self._session is not None:
            return True
        if not _MODEL_PATH.is_file():
            self.last_error = "U2NetP model is missing"
            return False
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            providers = [
                provider
                for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if provider in available
            ]
            self._session = ort.InferenceSession(str(_MODEL_PATH), providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self._session = None
            return False

    def _get_background(self, path: str, width: int, height: int) -> Optional[np.ndarray]:
        key = (path, width, height)
        if self._background_key == key and self._background is not None:
            return self._background
        image = _read_image(path)
        if image is None:
            self.last_error = "Background image could not be read"
            return None
        src_h, src_w = image.shape[:2]
        scale = max(width / max(1, src_w), height / max(1, src_h))
        resized = cv2.resize(
            image,
            (max(width, int(round(src_w * scale))), max(height, int(round(src_h * scale)))),
            interpolation=cv2.INTER_LANCZOS4,
        )
        x = (resized.shape[1] - width) // 2
        y = (resized.shape[0] - height) // 2
        self._background = np.ascontiguousarray(resized[y:y + height, x:x + width])
        self._background_key = key
        return self._background

    def apply(self, frame: np.ndarray, background_path: str) -> np.ndarray:
        """Return ``frame`` with its room replaced, or the original on error."""
        if not self._ensure_session() or not background_path:
            return frame
        height, width = frame.shape[:2]
        background = self._get_background(background_path, width, height)
        if background is None:
            return frame
        try:
            self._frame_number += 1
            # The compact U2NetP model is fast enough to refresh every second
            # frame.  This keeps the subject attached during normal webcam
            # motion while leaving enough GPU time for face swap and GPEN.
            refresh = (
                self._matte is None
                or self._matte_size != (width, height)
                or self._frame_number % max(
                    1, int(getattr(dlc_globals, "virtual_background_interval", 2))
                ) == 1
            )
            if refresh:
                # U2NetP uses ImageNet normalisation at its fixed 320 px input.
                matte_input = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_AREA)
                matte_input = matte_input[:, :, ::-1].astype(np.float32) / 255.0
                matte_input = (matte_input - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
                matte_input = np.ascontiguousarray(matte_input.transpose(2, 0, 1)[None])
                matte = self._session.run(None, {self._input_name: matte_input})[0][0, 0]
                matte = np.clip(matte, 0.0, 1.0)
                matte = cv2.resize(matte, (width, height), interpolation=cv2.INTER_CUBIC)
                # A very small blur removes pixel stair-stepping without erasing
                # hair detail or creating an artificial glow around the person.
                self._matte = cv2.GaussianBlur(matte, (0, 0), 1.0)
                self._matte_size = (width, height)
            matte = self._matte
            alpha = matte[:, :, None]
            return np.clip(frame.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)
        except Exception as exc:
            self.last_error = str(exc)
            return frame
