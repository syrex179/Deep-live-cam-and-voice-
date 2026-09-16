# Deep-Live-Cam — Current Codex Context

## Purpose

This document records the currently tested live-camera configuration and the
changes made during the September 2026 optimisation session. Use it as context
before making any future changes.

## Hardware and runtime

- GPU: NVIDIA RTX 4060, 8 GB VRAM
- CPU: Intel i7-12700F
- Camera: Iriun Webcam
- Camera input: 1280×960 at approximately 30 FPS
- Python: 3.12.10
- ONNX Runtime GPU: 1.26.0
- TensorRT: 10.9.0.34
- PyTorch CUDA: 12.8
- Live launch command:

```powershell
cd "C:\Users\danik\Desktop\Deep-Live-Cam-main"
.\venv\Scripts\python.exe run.py --execution-provider cuda
```

## Current user-facing configuration

- Face enhancer: GPEN-256
- GPEN strength: 65%
- Face Swap: TensorRT FP16
- Poisson Blend switch: ON
- Mouth Mask: ON at the user's chosen saved value
- Sharpness: restored at the user's chosen saved value
- Face detection: every frame
- GPEN: every frame
- Required processing order: **Face Swap → GPEN**

## Current measured result

The user reports approximately **20 FPS** with the configuration above.

Before the final eye-preservation change, live measurements were typically:

- detection: 12–13 ms
- Face Swap block: 19–21 ms
- GPEN-256: 19–22 ms
- Mouth Mask: about 0.8–1.1 ms
- fast Poisson Blend: about 2.5–4.1 ms
- total processing: about 53–59 ms (roughly 17–19 FPS in the internal log)

## Changes made

### `modules/processors/frame/_onnx_enhancer.py`

- GPEN strength remains `0.65`.
- No affine geometry, ROI semantics, mask geometry, blend weight, or GPEN
  input size was changed.
- GPEN diagnostic logging was reduced: a compact line is printed every 100
  calls instead of detailed output every 20 calls.

### `modules/processors/frame/face_swapper.py`

- The normal live Poisson Blend path was replaced with a real-time,
  affine-locked feathered blend over only the current face ROI.
- The public **Poisson Blend** switch remains ON and functional. The normal
  path no longer calls the expensive `cv2.seamlessClone`; the latter remains
  only as a defensive fallback if current affine blending cannot be used.
- The blend mask is derived from the swap's current affine transform every
  frame. There is no face-position cache, reprojection, affine smoothing, or
  temporal mask smoothing.
- An unused full-face mask calculation was removed from the Mouth Mask path.
  The visible Mouth Mask geometry and behaviour were kept unchanged.
- Target-eye preservation was added after face blending. Two small feathered
  eye regions are copied from the current camera frame at 85% strength using
  current 5-point detection landmarks. This makes gaze direction and blinking
  follow the real camera face instead of retaining the source portrait's
  forward-looking eyes.
- Teeth and braces are preserved after GPEN from the current target-camera
  frame only. The mask uses the actual InsightFace inner-mouth points
  `64:72`; points `52:64` are the outer lip and must not be restored, while
  points `96:104` are an eyebrow and must never be used for mouth recovery.
- The teeth holdout blends only its small mouth ROI. It is pixel-equivalent to
  the former full-frame blend but avoids full-frame float allocations.
- Diagnostics print `[BLEND DETAIL]` with timings for mouth, blend, and eye
  processing. These logs are diagnostic only.

### `modules/ui.py`

- `Sharpness` is now saved to and restored from `switch_states.json`.
- `Mouth Mask` size is now saved and restored instead of being reset to zero
  on every application launch.
- Moving either slider saves the new value immediately.

## Visual constraints that must stay unchanged

1. Keep **Face Swap → GPEN** order.
2. Keep GPEN-256 and strength 65% unless the user explicitly requests a test.
3. Keep current face affine math and Face Swap geometry.
4. Do not reintroduce stale-frame reprojection, temporal affine smoothing,
   locked affine, or temporal mask smoothing: previous tests caused the face
   to slide during fast head movement.
5. Do not casually replace the active TensorRT FP16 face swapper.
6. Preserve eye restoration from the current frame unless the user reports an
   unwanted identity/eye-colour trade-off.

## Backup files created in this session

### `backup_ui`

- `ui_before_persist_sharpness_mouth_v14.py`

### `backup_face_swapper`

- `face_swapper_before_poisson_roi_mouth_opt_v12.py`
- `face_swapper_before_fast_poisson_v13.py`
- `face_swapper_before_blend_profile_v15.py`
- `face_swapper_before_fast_poisson_cpu_opt_v16.py`
- `face_swapper_before_target_eye_preservation_v17.py`
- `face_swapper_before_inner_mouth_roi_opt_v18.py`
- `face_swapper_before_inner_mouth_landmark_fix_v19.py`
- `face_swapper_before_full_face_coverage_v20.py`

## Latest adjustment: fuller face replacement

- The affine-locked oval mask for both the initial face paste and Poisson
  blend was widened from `0.44 x 0.44` to `0.47 x 0.49` of the aligned face.
- This covers substantially more of the forehead, cheeks, and chin while
  retaining a narrow soft oval safety rim around the crop, so the background
  and hair corners are not pasted as a rectangle.
- The final, post-GPEN inner-mouth holdout remains in place: current-frame
  teeth and braces are retained to avoid black or artificial-looking teeth.

## Virtual camera output for OBS

- Installed `pyvirtualcam 0.15.0` in the project's `venv` and added a
  persistent `Virtual Camera` toggle to the Options card.
- When enabled during LIVE mode, the final processed BGR frame is sent to the
  Windows DirectShow device `Unity Video Capture`; OBS can add this as a
  `Video Capture Device` source.
- Installed and verified Unity Capture. The driver DLLs are registered from
  `C:\\Users\\danik\\Desktop\\Deep-Live-Cam-main\\UnityCapture-master\\Install`.
  Do not move or delete this directory while the camera is in use. To remove
  it later, first run its `Uninstall.bat` as Administrator.
- Backups: `backup_ui\\ui_before_virtual_camera_v21.py` and
  `backup_ui\\globals_before_virtual_camera_v21.py`.

## Natural teeth reference

- A fixed natural-teeth donor was tested but rejected: its shape did not match
  the user's moving mouth and looked artificial.
- The active path has been restored to the user's own current-frame inner
  mouth after GPEN. This preserves real teeth, braces and natural speech
  movement without a black mouth cavity.
- The existing `Mouth Mask` slider now controls own-mouth restoration:
  `0` turns it off; `100` applies the full inner-mouth restoration. The saved
  value remains `100`.
- `assets\\teeth_reference_smile.jpg` remains as an unused downloaded file;
  it is not read by the active live pipeline.
- Backups: `backup_face_swapper\\face_swapper_before_reference_teeth_v22.py`
  and `backup_ui\\ui_before_reference_teeth_v23.py`.

## Restore user's own teeth

- The current-camera mouth restore was a stable fallback, but it also copied
  real braces over the swapped face.
- The active path now snapshots the current Face Swap result *before* GPEN and
  restores only that current frame's inner mouth after GPEN. This keeps the
  face swapper's open/closed expression but avoids GPEN's dark mouth cavity
  and avoids copying the user's braces from the camera.
- It is not a temporal cache: the snapshot is made and used within the same
  video frame.
- Backups: `backup_face_swapper\\face_swapper_before_restore_own_teeth_v25.py`
  and `backup_ui\\ui_before_restore_own_teeth_v25.py`.

## Preserve swapped mouth before GPEN

- Moved the mouth snapshot in the live worker to the point after Face Swap and
  before GPEN, then restored it after GPEN.
- Backup: `backup_ui\\ui_before_swapped_mouth_restore_v27.py`.

## Own-teeth clarity

- Added one restrained local unsharp pass to the tiny current-camera
  inner-mouth ROI before it is restored after GPEN. It increases captured
  tooth edge contrast without generating or replacing teeth, lips, or skin.
- Backup: `backup_face_swapper\\face_swapper_before_own_teeth_clarity_v26.py`.

## Virtual camera control layout fix

- Fixed an Options-grid row collision: the `Virtual Camera` switch is now on
  its own row above the Face Enhancer selector and is clickable.
- Backup: `backup_ui\\ui_before_virtual_camera_layout_fix_v24.py`.

## LivePortrait mouth-model evaluation (2026-09-03)

- Installed the official LivePortrait project separately in `liveportrait\\`;
  the existing live camera pipeline and its settings were not changed.
- Created an isolated Python environment, installed the official weights, and
  verified that PyTorch detects the NVIDIA RTX 4060.
- Tested a neutral-to-smile driving sequence using the selected closed-mouth
  source portrait. In full-face mode the model generated teeth but visibly
  distorted head pose; in lip-only mode it opened the mouth without producing
  usable teeth. Both modes took about 16 seconds for 26 frames in this setup.
- Result: LivePortrait is installed for future experiments, but is deliberately
  not connected to the live camera. It does not meet the required natural-teeth
  quality or real-time performance for the active pipeline.

## HyperSwap quality camera mode (2026-09-04)

- Evaluated FaceFusion separately in `facefusion\\`; the active Deep Live Cam
  pipeline was left unchanged.
- `hyperswap_1a_256` was the only tested swap model that produced individual,
  natural-looking teeth without the user's braces or a black mouth cavity.
  `ghost_1_256` and `hififace_unofficial_256` were rejected because they kept
  the muddy/braced-mouth artefacts.
- A verified still-image result is `test_hyperswap_result.png`; this is the
  quality reference for the new mode.
- Added a `unity` webcam output mode to FaceFusion. It sends processed RGB
  frames through `pyvirtualcam` to the existing `Unity Video Capture` device,
  which OBS can select as a normal Video Capture Device.
- The FaceFusion webcam UI defaults to Unity output, 640x480, 25 FPS. Camera
  device `3` is the Iriun Webcam and is now selected by default; detected
  device IDs were `0, 2, 3`. Devices `0` and `2` are virtual/invalid sources
  and produce a near-black or noisy image.
- Measured video processing speed after startup was about 8.5 FPS at the
  natural-teeth quality setting with the original eight-worker configuration.
  Benchmarks on the same 25-frame local clip showed that two workers are the
  fastest stable setting: warm processing reached 23.6 FPS (one worker:
  18.4 FPS). The FaceFusion camera launcher now uses
  `--execution-thread-count 2`. The prior fast Deep Live Cam mode stays near
  20 FPS but cannot create tooth detail from its smaller face representation.
- Backups are stored next to the modified files with the suffix
  `.before_unity_output` and `webcam_options.py.before_quality_defaults`.

## Live controls and quality-preserving performance (2026-09-04)

- The main Deep Live Cam profile remains at `Quality`, `1080p`, and GPEN-256
  on every frame. The experimental low-resolution turbo profile was not kept,
  because the user prioritised image quality.
- The former red `Close` action in the live controls was changed to `Stop
  video`. It stops only the webcam workers, closes the Unity virtual-camera
  writer and releases Iriun Webcam; the main application stays open and can
  be resumed with `LIVE`.
- The OS window close button still exits the application normally.
- The Performance-only GPEN alternate-frame reproject path is available in
  code but is inactive in the Quality profile. It does not change the current
  image.
- Backups: `backup_ui\\ui_before_turbo_gpen_interval_v28.py` and
  `switch_states.before_turbo_profile_v28.json`.

## Expanded head coverage and source preview (2026-09-04)

- The live face-swap alpha was extended upward to the hairline and downward
  through the lower jaw/upper-neck part of the aligned crop. It stays a soft,
  affine-locked oval so it follows the head and does not create a rectangular
  pasted-photo edge.
- This is controlled internally by `full_head_coverage` and is enabled by
  default. INSwapper remains a face-swap model, so it cannot reliably invent
  a new hairstyle or a complete neck at arbitrary head angles; its expanded
  edge blends the model output where available and retains a safe soft rim.
- The source and target cards are now 280x180. Source photos use a
  proportion-preserving letterboxed preview rather than centre cropping, so
  the complete selected portrait remains visible.
- Backups: `backup_face_swapper\\face_swapper_before_head_neck_coverage_v29.py`,
  `backup_ui\\ui_before_full_source_preview_v29.py`, and
  `backup_ui\\globals_before_head_neck_coverage_v29.py`.

## Source hair, beard and mouth-mask alignment (2026-09-04)

- Added a cached aligned-source edge blend. It applies selected-source texture
  to the hair cap plus the jaw/chin/upper-neck rim while leaving the centre of
  the mouth to the live swap. A beardless source therefore overlays its clean
  jaw texture; a bearded source overlays its beard texture.
- The edge blend is saved as `source_edge_identity` and enabled by default.
  It is geometric, so an extreme profile turn or a source portrait with hair
  covering the face can still need a narrower blend adjustment.
- Fixed the mouth mask to use current prepared landmarks and to apply the
  same live X/Y/scale/rotation transform as the pasted face. It previously
  could remain at the untransformed mouth location. For a closed mouth it now
  uses a small outer-lip fallback instead of silently doing nothing.
- Backups: `backup_face_swapper\\face_swapper_before_source_hair_beard_v30.py`
  and `backup_ui\\globals_before_source_hair_beard_v30.py`.
- The source-edge texture overlay was immediately reverted after live visual
  testing produced visible patchwork artefacts. The expanded soft face mask
  and full-photo preview remain active; no source-photo texture overlay is
  active in the current pipeline.

## Mouth-mask geometry correction (2026-09-04)

- Reapplied only the safe mouth-mask fix, independently of the reverted
  hair/beard experiment: the mask now uses prepared landmarks and follows
  live X/Y/scale/rotation. A small outer-lip fallback makes the setting active
  with closed lips as well as an open mouth.
- The current fast INSwapper model is retained for stable real-time use. It
  cannot consistently replace hair or facial hair outside its learned facial
  region; doing so needs a different full-head generative pipeline and has
  materially different performance/quality trade-offs.
- Backup: `backup_face_swapper\\face_swapper_before_mouth_geometry_fix_v31.py`.

## Crisp mouth-mask blend (2026-09-04)

- Removed the 5px Gaussian blur from the inner-mouth contour when the Mouth
  Mask slider is at 100%. This stops the mask itself from softening teeth and
  the mouth opening.
- Removed the additional unsharp/blur cycle on the copied mouth ROI. The
  pre-enhancer swapped pixels are now copied without an extra filtering pass.
- The restored ROI is capped at 70% opacity at the highest slider value, so
  30% of the sharper GPEN result remains visible rather than being hidden by
  the low-resolution 128px swap crop.
- This is a blend-quality repair, not a new teeth generator: a one-photo
  face-swap model cannot add reliably detailed, moving teeth that are absent
  from its source representation.
- Backup: `backup_face_swapper\\face_swapper_before_crisp_mouth_mask_v32.py`.

## Mouth-mask rollback after live test (2026-09-04)

- The crisp-mask experiment was reverted after the live result showed a more
  artificial, dark mouth. The reliable landmark-alignment and closed-mouth
  fallback remain in place.
- The mask again restores the pre-GPEN swapped mouth at the selected slider
  strength, with its small 5px lip-edge feather. This is the prior stable
  behavior and avoids mixing a dark GPEN mouth into the teeth region.
- Backup of the rejected crisp-mask version:
  `backup_face_swapper\\face_swapper_before_revert_crisp_mouth_v33.py`.

## Full-head source-video feasibility test (2026-09-04)

- Supplied source video: 1276x718 H.264, 25 FPS, 311.6 seconds. It contains
  stable frontal speech, visible hair, beard and mouth, and is suitable as a
  high-quality full-head identity source.
- LivePortrait was tested on a one-second, 25-frame self-driven excerpt.
  It preserved the complete source head rather than producing the current
  INSwapper-style inner-face mask. The output confirms that this architecture
  can retain source hair, beard and teeth.
- The RTX 4060 and PyTorch CUDA path are healthy. The test nevertheless took
  17 seconds for 25 generated frames (about 1.5 FPS), so it must not be wired
  into the live virtual camera yet.
- `torch.compile` cannot optimize this Windows environment because no working
  Triton installation is present. LivePortrait's ONNX auxiliary modules also
  report a CUDA provider DLL load error and fall back from that provider.
- Next engineering step for a usable live full-head mode is an exported
  TensorRT/ONNX generator pipeline or a different real-time full-portrait
  model. Reusing the current 128px swap mask cannot achieve full replacement.

## TensorRT acceleration for the full-head prototype (2026-09-04)

- Built and verified two fixed-shape FP16 TensorRT engines for RTX 4060:
  `liveportrait_warp_decode_fp16.engine` (215 MB) and
  `liveportrait_motion_fp16.engine` (62 MB), stored under
  `liveportrait\\pretrained_weights\\liveportrait\\engines`.
- The generated full-head W+G stage improved from 91.6 ms (10.9 FPS) in eager
  PyTorch to 45.25 ms (22.1 FPS) in TensorRT. Motion extraction plus full-head
  generation together benchmarks at 47.73 ms (20.95 FPS), excluding webcam
  capture, crop/paste composition and virtual-camera output.
- TensorRT does not natively support LivePortrait's five-dimensional
  GridSample. `liveportrait\\tools\\build_full_head_tensorrt.py` contains a
  tested trilinear gather-based export-only equivalent. Its mean output
  difference from the original PyTorch model is 0.00025 on the 0..1 scale.
- `liveportrait\\tools\\build_motion_tensorrt.py` builds the matching
  expression/pose engine. Both engines were compared to the original models;
  motion output differences are consistent with expected FP16 rounding.
- `onnxscript` and `onnx_ir` were installed in the main project virtual
  environment solely to support PyTorch ONNX export.
- The engines are not yet attached to the Deep Live Cam UI. The remaining
  work is a full-head compositor that maps the generated 512px portrait onto
  the current webcam head while preserving the webcam background. It must be
  implemented before exposing a live toggle, otherwise a source-video
  rectangle would be visible.

## Safe future workflow

Before replacing a working Python file:

1. Create a uniquely named backup in the relevant backup folder.
2. Make one focused change only.
3. Syntax-check the generated Python file before testing.
4. Test live FPS, rapid left/right head movement, the face edge, blinking,
   and gaze direction.
5. Keep only changes that improve the requested result without face sliding
   or visible jitter.
