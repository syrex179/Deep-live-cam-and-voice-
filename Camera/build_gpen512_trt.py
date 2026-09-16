from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "GPEN-BFR-512.onnx"
ENGINE = ROOT / "models" / "GPEN-BFR-512_fp16.engine"


def fail(message: str) -> None:
    print(f"[GPEN-TRT] ERROR: {message}")
    raise SystemExit(1)


def main() -> None:
    print("[GPEN-TRT] GPEN-512 TensorRT FP16 builder")

    if not MODEL.exists():
        fail(f"Model not found: {MODEL}")

    try:
        import tensorrt as trt
    except Exception as exc:
        fail(f"Could not import TensorRT: {exc}")

    logger = trt.Logger(trt.Logger.INFO)

    print(f"[GPEN-TRT] TensorRT version: {trt.__version__}")
    print(f"[GPEN-TRT] ONNX: {MODEL}")
    print(f"[GPEN-TRT] ENGINE: {ENGINE}")

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with MODEL.open("rb") as f:
        data = f.read()

    if not parser.parse(data):
        errors = []
        for i in range(parser.num_errors):
            errors.append(str(parser.get_error(i)))
        fail("ONNX parse failed:\n" + "\n".join(errors))

    if network.num_inputs != 1:
        fail(f"Expected one input, found {network.num_inputs}")

    inp = network.get_input(0)
    shape = tuple(inp.shape)
    print(f"[GPEN-TRT] Input: {inp.name} {shape} dtype={inp.dtype}")

    # GPEN-BFR-512 live path is static 1x3x512x512.
    if shape != (1, 3, 512, 512):
        fail(f"Unexpected GPEN-512 input shape: {shape}")

    config = builder.create_builder_config()

    # 4 GiB workspace is enough for this model on the RTX 4060.
    workspace = 4 * (1 << 30)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    except Exception as exc:
        print(f"[GPEN-TRT] Workspace setting failed: {exc}")

    if hasattr(config, "set_flag"):
        config.set_flag(trt.BuilderFlag.FP16)

    # Prefer timing cache between rebuilds.
    timing_path = ROOT / "trt_cache_gpen"
    timing_path.mkdir(parents=True, exist_ok=True)
    timing_cache_file = timing_path / "gpen512_timing.cache"

    timing_cache = None
    if timing_cache_file.exists():
        try:
            cache_data = timing_cache_file.read_bytes()
            timing_cache = config.create_timing_cache(cache_data)
            config.set_timing_cache(timing_cache, False)
            print(f"[GPEN-TRT] Reusing timing cache: {timing_cache_file}")
        except Exception as exc:
            print(f"[GPEN-TRT] Timing cache load skipped: {exc}")

    print("[GPEN-TRT] Building FP16 engine. First build may take a while...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        fail("TensorRT failed to build the serialized engine.")

    ENGINE.parent.mkdir(parents=True, exist_ok=True)
    ENGINE.write_bytes(bytes(serialized))
    print(f"[GPEN-TRT] Engine written: {ENGINE}")
    print(f"[GPEN-TRT] Size: {ENGINE.stat().st_size / (1024 * 1024):.1f} MiB")

    # Save updated timing cache when the API provides it.
    try:
        cache = config.get_timing_cache()
        cache_data = cache.serialize()
        timing_cache_file.write_bytes(bytes(cache_data))
        print(f"[GPEN-TRT] Timing cache written: {timing_cache_file}")
    except Exception as exc:
        print(f"[GPEN-TRT] Timing cache save skipped: {exc}")

    print("[GPEN-TRT] BUILD COMPLETE")


if __name__ == "__main__":
    main()
