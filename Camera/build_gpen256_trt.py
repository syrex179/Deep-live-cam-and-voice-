"""Build a TensorRT FP16 engine for the existing GPEN-BFR-256 model."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "GPEN-BFR-256.onnx"
ENGINE = ROOT / "models" / "GPEN-BFR-256_fp16.engine"


def main() -> None:
    if not MODEL.exists():
        raise SystemExit(f"[GPEN-TRT] Model is missing: {MODEL}")

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(MODEL.read_bytes()):
        details = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise SystemExit(f"[GPEN-TRT] ONNX parse failed:\n{details}")

    inp = network.get_input(0)
    if tuple(inp.shape) != (1, 3, 256, 256):
        raise SystemExit(f"[GPEN-TRT] Unexpected input shape: {tuple(inp.shape)}")

    config = builder.create_builder_config()
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1 << 30))
    except Exception:
        pass
    config.set_flag(trt.BuilderFlag.FP16)

    cache_dir = ROOT / "trt_cache_gpen"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "gpen256_timing.cache"
    if cache_file.exists():
        try:
            config.set_timing_cache(config.create_timing_cache(cache_file.read_bytes()), False)
        except Exception:
            pass

    print("[GPEN-TRT] Building GPEN-256 FP16 engine...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("[GPEN-TRT] TensorRT could not build the engine")
    ENGINE.write_bytes(bytes(serialized))
    try:
        cache_file.write_bytes(bytes(config.get_timing_cache().serialize()))
    except Exception:
        pass
    print(f"[GPEN-TRT] Ready: {ENGINE} ({ENGINE.stat().st_size / 1048576:.1f} MiB)", flush=True)


if __name__ == "__main__":
    main()
