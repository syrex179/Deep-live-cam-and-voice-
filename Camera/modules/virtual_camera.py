"""Low-overhead virtual-camera output for the live processing pipeline.

The OBS virtual-camera driver exposes a standard DirectShow camera that can
be opened by common consumers such as Telegram and Discord.  ``pyvirtualcam``
is imported lazily, so ordinary preview mode keeps working even when the
optional output package or driver is unavailable.
"""

from __future__ import annotations

from typing import Optional, Tuple
import threading

import cv2
import numpy as np


class VirtualCameraOutput:
    """Send BGR OpenCV frames to the OBS virtual camera.

    A camera instance is recreated only when the output dimensions or target
    frame rate change.  This leaves the face-swap pipeline frame-local and
    avoids an unnecessary BGR-to-RGB copy on every frame.
    """

    DEVICE_NAME = "OBS Virtual Camera"
    BACKEND = "obs"
    # Keep one mode for the producer and every consumer.  This is the native
    # mode exposed by the installed OBS driver and was verified with DirectShow
    # itself, which is the same Windows API Telegram and Discord use.
    WIDTH = 1280
    HEIGHT = 720
    FPS = 30

    def __init__(self) -> None:
        self._camera = None
        self._spec: Optional[Tuple[int, int, int]] = None

    @classmethod
    def _to_output_frame(cls, bgr_frame: np.ndarray) -> np.ndarray:
        """Letterbox a frame into the fixed virtual-camera video mode."""
        height, width = bgr_frame.shape[:2]
        if (width, height) == (cls.WIDTH, cls.HEIGHT):
            return np.ascontiguousarray(bgr_frame)

        scale = min(cls.WIDTH / float(width), cls.HEIGHT / float(height))
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(bgr_frame, (resized_w, resized_h), interpolation=interpolation)
        output = np.zeros((cls.HEIGHT, cls.WIDTH, 3), dtype=np.uint8)
        left = (cls.WIDTH - resized_w) // 2
        top = (cls.HEIGHT - resized_h) // 2
        output[top:top + resized_h, left:left + resized_w] = resized
        return output

    def send(self, bgr_frame: np.ndarray, fps: int) -> str:
        """Publish one stable 1280x720/30 BGR frame to the virtual camera."""
        if bgr_frame is None or bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
            raise ValueError("Virtual camera requires a BGR frame with 3 channels")
        if bgr_frame.dtype != np.uint8:
            raise TypeError("Virtual camera requires uint8 frames")

        # ``fps`` intentionally is not used as the camera's advertised mode:
        # inference FPS may fluctuate, while consumers need a stable stream.
        spec = (self.WIDTH, self.HEIGHT, self.FPS)
        if self._camera is None or self._spec != spec:
            self.close()
            try:
                import pyvirtualcam
            except ImportError as exc:
                raise RuntimeError(
                    "pyvirtualcam is not installed in Deep Live Cam's environment"
                ) from exc
            self._camera = pyvirtualcam.Camera(
                width=spec[0], height=spec[1], fps=spec[2],
                fmt=pyvirtualcam.PixelFormat.BGR,
                device=self.DEVICE_NAME,
                backend=self.BACKEND,
            )
            self._spec = spec

        # pyvirtualcam accepts a C-contiguous BGR frame directly.
        self._camera.send(self._to_output_frame(bgr_frame))
        return str(self._camera.device)

    def close(self) -> None:
        if self._camera is not None:
            try:
                # Clear the last frame before closing so consumers do not
                # keep showing a stale swapped face after LIVE is stopped.
                if self._spec is not None:
                    width, height, _fps = self._spec
                    try:
                        self._camera.send(np.zeros((height, width, 3), dtype=np.uint8))
                    except Exception:
                        pass
                self._camera.close()
            finally:
                self._camera = None
                self._spec = None


class VirtualCameraPublisher:
    """Publish the newest processed frame without stalling face processing.

    Virtual-camera drivers may block ``send`` while a consumer (OBS, Unity
    Capture, TikTok Live Studio) renegotiates or is briefly busy.  The live
    processor must never wait for that I/O: it owns the preview's frame rate.
    This publisher keeps exactly one latest frame and writes it from a small
    background thread, dropping obsolete frames instead of building latency.
    """

    def __init__(self) -> None:
        self._output = VirtualCameraOutput()
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._closing = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_fps = 30
        self._thread: Optional[threading.Thread] = None
        self._device_name: Optional[str] = None
        self._error: Optional[Exception] = None

    @property
    def device_name(self) -> Optional[str]:
        return self._device_name

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    def submit(self, bgr_frame: np.ndarray, fps: int) -> None:
        """Replace a pending frame and return immediately to the processor."""
        if self._closing.is_set():
            return
        if self._error is not None:
            raise RuntimeError(str(self._error)) from self._error
        with self._lock:
            # The processor makes a fresh output frame for every iteration;
            # retaining that immutable reference avoids an extra full-frame
            # copy on its hot path.
            self._latest_frame = bgr_frame
            self._latest_fps = max(1, int(fps))
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run,
                name="virtual-camera-output",
                daemon=True,
            )
            self._thread.start()
        self._ready.set()

    def _run(self) -> None:
        try:
            while not self._closing.is_set():
                self._ready.wait(0.10)
                self._ready.clear()
                if self._closing.is_set():
                    break
                with self._lock:
                    frame = self._latest_frame
                    fps = self._latest_fps
                    self._latest_frame = None
                if frame is not None:
                    self._device_name = self._output.send(frame, fps)
        except Exception as exc:
            self._error = exc
        finally:
            self._output.close()

    def close(self) -> None:
        """Request shutdown without blocking the live-processing thread."""
        self._closing.set()
        self._ready.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.25)
