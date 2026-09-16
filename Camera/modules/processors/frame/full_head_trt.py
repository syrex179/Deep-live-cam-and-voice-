"""Experimental full-head LivePortrait compositor accelerated by TensorRT."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import modules.globals


ROOT = Path(__file__).resolve().parents[3]
LIVEPORTRAIT = ROOT / "liveportrait"
if str(LIVEPORTRAIT) not in sys.path:
    sys.path.insert(0, str(LIVEPORTRAIT))


def _square(frame: np.ndarray, bbox: np.ndarray, factor: float = 2.35) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    side = max(32, int(round(max(x1 - x0, y1 - y0) * factor)))
    center = ((x0 + x1) * .5, (y0 + y1) * .5)
    return cv2.getRectSubPix(frame, (side, side), center)


def _aligned_256(frame: np.ndarray, face) -> np.ndarray:
    """LivePortrait expects a five-landmark aligned crop, not a bbox resize."""
    from insightface.utils import face_align
    kps = getattr(face, "kps", None)
    if kps is None:
        raise RuntimeError("Face landmarks are unavailable for full-head alignment")
    return face_align.norm_crop(frame, kps.astype(np.float32), image_size=256)


class FullHeadTRT:
    def __init__(self, source_video: str) -> None:
        import onnxruntime as ort
        import tensorrt as trt
        from modules.face_analyser import get_one_face
        from src.config.inference_config import InferenceConfig
        from src.config.crop_config import CropConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.camera import get_rotation_matrix, headpose_pred_to_degree
        from src.utils.cropper import Cropper

        self._trt = trt
        self._rotation = get_rotation_matrix
        self._headpose = headpose_pred_to_degree
        self._cfg = InferenceConfig()
        self._crop_cfg = CropConfig()
        self._cropper = Cropper(crop_cfg=self._crop_cfg)
        self._wrapper = LivePortraitWrapper(self._cfg)
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engines = LIVEPORTRAIT / "pretrained_weights" / "liveportrait" / "engines"
        # The original bundled motion engine produces NaNs with TensorRT 10.9.
        # This local engine is rebuilt from the same ONNX file on this RTX 4060.
        motion_path = engines / "liveportrait_motion_fp16_compat.engine"
        if not motion_path.is_file():
            raise RuntimeError("Compatible full-head motion engine is missing")
        self._motion_engine = self._runtime.deserialize_cuda_engine(motion_path.read_bytes())
        if not self._motion_engine:
            raise RuntimeError("Compatible full-head motion engine could not load")
        self._motion = self._motion_engine.create_execution_context()
        # The locally rebuilt decoder matches the active TensorRT runtime.
        # It is substantially faster than copying the 512px feature volume
        # through ONNX Runtime for every webcam frame.
        decode_path = engines / "liveportrait_warp_decode_fp16_compat.engine"
        self._decode_engine = (
            self._runtime.deserialize_cuda_engine(decode_path.read_bytes())
            if decode_path.is_file() else None
        )
        self._decode = (
            self._decode_engine.create_execution_context()
            if self._decode_engine is not None else None
        )
        # Decoding the 512px portrait via CUDA ONNX Runtime is reliable on this
        # system. Rebuilding the matching TensorRT decoder requires >25 GB RAM
        # and gets killed by Windows, while the old prebuilt decoder is invalid.
        self._decode_session = ort.InferenceSession(
            str(engines / "liveportrait_warp_decode_fp16.onnx"),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        source = self._load_reference_frame(source_video, get_one_face)
        if source is None:
            raise RuntimeError("Could not read full-head source video")
        source_face = get_one_face(source)
        source_info = self._cropper.crop_source_image(
            cv2.cvtColor(source, cv2.COLOR_BGR2RGB), self._crop_cfg
        )
        if source_face is None or source_info is None:
            raise RuntimeError("No face found in full-head source video")
        self._source_crop = source_info["img_crop_256x256"]
        self._source_visual = cv2.cvtColor(
            cv2.resize(self._source_crop, (512, 512), interpolation=cv2.INTER_CUBIC),
            cv2.COLOR_RGB2BGR,
        )
        self._head_alpha = self._create_head_alpha(self._source_visual)
        self._hair_alpha = self._create_hair_alpha(self._source_visual)
        source_tensor = self._wrapper.prepare_source(self._source_crop)
        self._source_info = self._wrapper.get_kp_info(source_tensor)
        self._source_canonical = self._source_info["kp"]
        self._source_rotation = self._rotation(self._source_info["pitch"], self._source_info["yaw"], self._source_info["roll"])
        self._source_keypoints = self._wrapper.transform_keypoint(self._source_info).half()
        self._feature = self._wrapper.extract_feature_3d(source_tensor).half()
        self._feature_np = self._feature.detach().cpu().numpy()
        self._source_keypoints_np = self._source_keypoints.detach().cpu().numpy()
        self._driver_zero = None
        self._target_M_c2o = None
        self._smoothed_M_c2o = None
        self._warm_frames = 0
        self._out = torch.empty((1, 3, 512, 512), device="cuda", dtype=torch.float16)
        # These bundled engines deserialize on this TensorRT runtime but emit
        # NaNs here. Do not even probe them per frame: a failed CUDA launch can
        # poison the stream and leave the native fallback waiting indefinitely.
        # The verified native CUDA path remains fully GPU-accelerated.
        # This engine was rebuilt locally from the bundled ONNX and verified
        # against the current TensorRT/CUDA versions.  Using it saves the
        # considerably slower PyTorch motion pass on every webcam frame.
        self._trt_motion_ok = True
        self._trt_decode_ok = self._decode is not None

    @classmethod
    def _load_reference_frame(cls, source_video: str, get_one_face) -> np.ndarray | None:
        """Read a cached best source frame, or select and cache it once.

        Full-head source preparation must not make the first live preview look
        broken while dozens of frames from a long source video are inspected.
        A per-video cache makes every later LIVE start immediate.
        """
        source_path = Path(source_video)
        cache_dir = ROOT / "source_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.blake2b(str(source_path.resolve()).encode("utf-8"), digest_size=8).hexdigest()
        cache_path = cache_dir / f"full_head_reference_{key}.jpg"
        if cache_path.is_file():
            cached = cv2.imread(str(cache_path))
            if cached is not None and get_one_face(cached) is not None:
                return cached
        source = cls._select_reference_frame(source_video, get_one_face)
        if source is not None:
            try:
                cv2.imwrite(str(cache_path), source)
            except cv2.error:
                pass
        return source

    @staticmethod
    def _select_reference_frame(source_video: str, get_one_face) -> np.ndarray | None:
        """Choose a clear, large source face rather than blindly using frame 0."""
        cap = cv2.VideoCapture(source_video)
        if not cap.isOpened():
            return None
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        stride = max(1, total // 70)
        best_frame = None
        best_score = -1.0
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride:
                index += 1
                continue
            face = get_one_face(frame)
            index += 1
            if face is None or getattr(face, "bbox", None) is None:
                continue
            x0, y0, x1, y1 = [float(v) for v in face.bbox]
            width, height = max(1.0, x1 - x0), max(1.0, y1 - y0)
            area = (width * height) / float(max(1, frame.shape[0] * frame.shape[1]))
            crop = frame[max(0, int(y0)):min(frame.shape[0], int(y1)), max(0, int(x0)):min(frame.shape[1], int(x1))]
            sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()) if crop.size else 0.0
            exposure = float(crop.mean()) if crop.size else 0.0
            # LivePortrait needs some space above the hairline. A frame where
            # the head touches the top edge creates a black crown after crop.
            top_margin = max(0.0, y0 / float(max(1, frame.shape[0])))
            score = (
                min(1.0, area / .12) * .42
                + min(1.0, sharpness / 180.0) * .27
                + max(0.0, 1.0 - abs(exposure - 128.0) / 128.0) * .08
                + min(1.0, top_margin / .08) * .23
            )
            if score > best_score:
                best_score, best_frame = score, frame.copy()
        cap.release()
        return best_frame

    @staticmethod
    def _create_head_alpha(source_bgr: np.ndarray) -> np.ndarray:
        """Segment head/hair/neck once; never leak source-video background."""
        try:
            import onnxruntime as ort
            model = ROOT / "facefusion" / ".assets" / "models" / "bisenet_resnet_34.onnx"
            # This runs once at source selection. Keep it on CPU so the parser
            # cannot reserve VRAM needed by the two live TensorRT engines.
            session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
            rgb = source_bgr[:, :, ::-1].astype(np.float32) / 255.0
            rgb = (rgb - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
            tensor = rgb.transpose(2, 0, 1)[None]
            labels = session.run(None, {session.get_inputs()[0].name: tensor})[0][0].argmax(0)
            # CelebAMask labels: face parts, ears, neck and hair. Background,
            # clothing and the room are deliberately excluded.
            mask = np.isin(labels, (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17)).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            return cv2.GaussianBlur(mask, (0, 0), 3).astype(np.float32) / 255.0
        except Exception:
            # Strict fallback: narrow head-only oval, never the wide source box.
            mask = np.zeros((512, 512), np.uint8)
            cv2.ellipse(mask, (256, 245), (190, 235), 0, 0, 360, 255, -1)
            return cv2.GaussianBlur(mask, (0, 0), 3).astype(np.float32) / 255.0

    @staticmethod
    def _create_hair_alpha(source_bgr: np.ndarray) -> np.ndarray:
        """Keep original source hair instead of synthesizing a clipped hairline."""
        try:
            import onnxruntime as ort
            model = ROOT / "facefusion" / ".assets" / "models" / "bisenet_resnet_34.onnx"
            session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
            rgb = source_bgr[:, :, ::-1].astype(np.float32) / 255.0
            rgb = (rgb - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
            tensor = rgb.transpose(2, 0, 1)[None]
            labels = session.run(None, {session.get_inputs()[0].name: tensor})[0][0].argmax(0)
            mask = (labels == 17).astype(np.uint8) * 255  # CelebAMask-HQ hair
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            return cv2.GaussianBlur(mask, (0, 0), 2.5).astype(np.float32) / 255.0
        except Exception:
            return np.zeros((512, 512), np.float32)

    def _motion_info(self, bgr_frame: np.ndarray, face) -> dict:
        if face is None or getattr(face, "bbox", None) is None:
            raise RuntimeError("Live driver face was not detected")
        crop_info = self._cropper.crop_source_image(
            cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB), self._crop_cfg
        )
        if crop_info is None:
            raise RuntimeError("Live driver face was not detected")
        # Paste the generated portrait back with LivePortrait's face-aligned
        # transform rather than a loose bounding-box square.  Smooth just the
        # transform between neighbouring camera frames: this removes detector
        # jitter while keeping the head physically attached to the neck.
        current_M = crop_info["M_c2o"].astype(np.float32)
        if self._smoothed_M_c2o is None:
            self._smoothed_M_c2o = current_M.copy()
        else:
            jump = float(np.linalg.norm(current_M[:2, 2] - self._smoothed_M_c2o[:2, 2]))
            if jump > 90.0:
                self._smoothed_M_c2o = current_M.copy()
            else:
                self._smoothed_M_c2o = self._smoothed_M_c2o * .62 + current_M * .38
        self._target_M_c2o = self._smoothed_M_c2o
        rgb = crop_info["img_crop_256x256"]
        portrait = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16).div_(255)
        outputs = {
            "pitch": torch.empty((1, 66), device="cuda", dtype=torch.float16),
            "yaw": torch.empty((1, 66), device="cuda", dtype=torch.float16),
            "roll": torch.empty((1, 66), device="cuda", dtype=torch.float16),
            "translation": torch.empty((1, 3), device="cuda", dtype=torch.float16),
            "expression": torch.empty((1, 63), device="cuda", dtype=torch.float16),
            "scale": torch.empty((1, 1), device="cuda", dtype=torch.float16),
            "keypoints": torch.empty((1, 63), device="cuda", dtype=torch.float16),
        }
        if self._trt_motion_ok:
            self._motion.set_tensor_address("portrait", portrait.data_ptr())
            for name, value in outputs.items():
                self._motion.set_tensor_address(name, value.data_ptr())
            ran = self._motion.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.current_stream().synchronize()
            if ran and all(torch.isfinite(value).all() for value in outputs.values()):
                return {"pitch": self._headpose(outputs["pitch"].float()).reshape(-1, 1), "yaw": self._headpose(outputs["yaw"].float()).reshape(-1, 1), "roll": self._headpose(outputs["roll"].float()).reshape(-1, 1), "t": outputs["translation"].float(), "exp": outputs["expression"].float().reshape(1, 21, 3), "scale": outputs["scale"].float(), "kp": outputs["keypoints"].float().reshape(1, 21, 3)}
            self._trt_motion_ok = False

        # Native PyTorch CUDA path. ``get_kp_info`` already returns pose in
        # degrees by default, so do not pass those scalars through the 66-bin
        # head-pose decoder a second time.
        native = self._wrapper.get_kp_info(self._wrapper.prepare_source(rgb))
        return {
            "pitch": native["pitch"].float().reshape(-1, 1),
            "yaw": native["yaw"].float().reshape(-1, 1),
            "roll": native["roll"].float().reshape(-1, 1),
            "t": native["t"].float(),
            "exp": native["exp"].float().reshape(1, 21, 3),
            "scale": native["scale"].float(),
            "kp": native["kp"].float().reshape(1, 21, 3),
        }

    def process(self, frame: np.ndarray, face) -> np.ndarray:
        driver = self._motion_info(frame, face)
        rotation = self._rotation(driver["pitch"], driver["yaw"], driver["roll"])
        if self._driver_zero is None:
            self._driver_zero = {k: v.clone() for k, v in driver.items()}
        zero = self._driver_zero
        # Webcam crops are less tightly normalised than LivePortrait's offline
        # driver videos. Clamp their first-order motion so one bad detection
        # can never fold the portrait into a black/empty generator output.
        motion = (driver["exp"] - zero["exp"]).clamp(-0.12, 0.12)
        delta = self._source_info["exp"] + motion
        pitch = (driver["pitch"] - zero["pitch"]).clamp(-18.0, 18.0) + zero["pitch"]
        yaw = (driver["yaw"] - zero["yaw"]).clamp(-18.0, 18.0) + zero["yaw"]
        roll = (driver["roll"] - zero["roll"]).clamp(-18.0, 18.0) + zero["roll"]
        rotation = self._rotation(pitch, yaw, roll)
        new_rotation = (rotation @ self._rotation(zero["pitch"], zero["yaw"], zero["roll"]).permute(0, 2, 1)) @ self._source_rotation
        scale = self._source_info["scale"] * (driver["scale"] / zero["scale"]).clamp(.90, 1.10)
        translation = self._source_info["t"] + (driver["t"] - zero["t"]).clamp(-.05, .05)
        translation[..., 2].fill_(0)
        driving = scale * (self._source_canonical @ new_rotation + delta) + translation
        driving = self._wrapper.stitching(self._source_keypoints.float(), driving).half()
        portrait = None
        if self._trt_decode_ok:
            self._decode.set_tensor_address("feature_3d", self._feature.data_ptr())
            self._decode.set_tensor_address("kp_source", self._source_keypoints.data_ptr())
            self._decode.set_tensor_address("kp_driving", driving.data_ptr())
            self._decode.set_tensor_address("portrait", self._out.data_ptr())
            ran = self._decode.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.current_stream().synchronize()
            candidate = self._out[0].float().permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
            if ran and np.isfinite(candidate).all() and float(candidate.mean()) >= 8.0:
                portrait = candidate
            else:
                self._trt_decode_ok = False
        if portrait is None:
            decoded = self._decode_session.run(
                None,
                {
                    "feature_3d": self._feature_np,
                    "kp_source": self._source_keypoints_np,
                    "kp_driving": driving.detach().cpu().numpy(),
                },
            )[0]
            portrait = np.clip(decoded[0].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        portrait = cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR)
        # Keep the generator output until a hair-only overlay is geometrically
        # aligned. A static source-hair paste must never cover the live face.
        if self._target_M_c2o is None:
            return frame
        # The generated portrait is a 512x512 LivePortrait crop, so use the
        # corresponding crop-to-original transform.  This is the same
        # alignment path used by LivePortrait itself and eliminates the
        # detached-head effect of a generic rectangle.
        alpha = self._head_alpha.copy()
        # Fade only the lower neck into the user's body. Hair, jaw and beard
        # remain fully generated, while bare shoulders or clothing cannot form
        # a hard artificial cut at the bottom of the head.
        fade_start = int(alpha.shape[0] * .72)
        alpha[fade_start:] *= np.linspace(1.0, 0.0, alpha.shape[0] - fade_start, dtype=np.float32)[:, None]
        # Warp only the bounding ROI of the 512px portrait, not the whole
        # 1080p webcam frame. It keeps the same landmark-accurate transform
        # but removes tens of milliseconds of needless pixel work per frame.
        crop_corners = np.array([[[0., 0.], [511., 0.], [511., 511.], [0., 511.]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(crop_corners, self._target_M_c2o)[0]
        left = max(0, int(np.floor(projected[:, 0].min())) - 3)
        top = max(0, int(np.floor(projected[:, 1].min())) - 3)
        right = min(frame.shape[1], int(np.ceil(projected[:, 0].max())) + 4)
        bottom = min(frame.shape[0], int(np.ceil(projected[:, 1].max())) + 4)
        if right <= left or bottom <= top:
            return frame
        shift = np.array([[1., 0., -left], [0., 1., -top], [0., 0., 1.]], dtype=np.float32)
        roi_M = shift @ self._target_M_c2o
        roi_size = (right - left, bottom - top)
        generated = cv2.warpPerspective(portrait, roi_M, roi_size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective((alpha * 255.0).astype(np.uint8), roi_M, roi_size, flags=cv2.INTER_LINEAR)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.4).astype(np.float32)[..., None] / 255.0
        roi = frame[top:bottom, left:right].astype(np.float32)
        frame[top:bottom, left:right] = np.clip(generated * mask + roi * (1.0 - mask), 0, 255).astype(np.uint8)
        return frame
