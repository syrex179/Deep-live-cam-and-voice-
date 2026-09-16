"""Build the RVC training manifest for the prepared single-speaker dataset."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "logs" / "source_voice"


def normalized_stem(path: Path) -> str:
    name = path.name
    for suffix in (".wav.npy", ".npy", ".wav"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def escaped(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\")


def main() -> None:
    gt = EXPERIMENT / "0_gt_wavs"
    features = EXPERIMENT / "3_feature768"
    f0 = EXPERIMENT / "2a_f0"
    f0_nsf = EXPERIMENT / "2b-f0nsf"

    names = (
        {normalized_stem(path) for path in gt.glob("*.wav")}
        & {normalized_stem(path) for path in features.glob("*.npy")}
        & {normalized_stem(path) for path in f0.glob("*.wav.npy")}
        & {normalized_stem(path) for path in f0_nsf.glob("*.wav.npy")}
    )
    if not names:
        raise RuntimeError("No complete RVC training records were found.")

    records = [
        "|".join(
            (
                escaped(gt / f"{name}.wav"),
                escaped(features / f"{name}.npy"),
                escaped(f0 / f"{name}.wav.npy"),
                escaped(f0_nsf / f"{name}.wav.npy"),
                "0",
            )
        )
        for name in sorted(names)
    ]

    mute_root = ROOT / "logs" / "mute"
    mute_record = "|".join(
        (
            escaped(mute_root / "0_gt_wavs" / "mute40k.wav"),
            escaped(mute_root / "3_feature768" / "mute.npy"),
            escaped(mute_root / "2a_f0" / "mute.wav.npy"),
            escaped(mute_root / "2b-f0nsf" / "mute.wav.npy"),
            "0",
        )
    )
    records.extend((mute_record, mute_record))
    random.Random(20260907).shuffle(records)
    (EXPERIMENT / "filelist.txt").write_text("\n".join(records), encoding="utf-8")

    config_path = EXPERIMENT / "config.json"
    if not config_path.exists():
        shutil.copyfile(ROOT / "configs" / "v1" / "40k.json", config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["train"]["batch_size"] = 4
    config.pop("speaker_info", None)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(names)} voice segments + 2 mute records.")


if __name__ == "__main__":
    main()
