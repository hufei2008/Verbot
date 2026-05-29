#!/usr/bin/env python3
import argparse
import math
import os
import struct
import wave
from pathlib import Path


def read_wav(path: Path):
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"{path} must be 16-bit mono PCM")
        sr = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    n = len(frames) // 2
    samples = struct.unpack("<" + "h" * n, frames) if n else []
    floats = [s / 32768.0 for s in samples]
    return sr, floats


def metrics(samples, sr):
    if not samples:
        return {
            "dur": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "zcr": 0.0,
            "clip": 0.0,
        }
    rms = math.sqrt(sum(x * x for x in samples) / len(samples))
    peak = max(abs(x) for x in samples)
    crossings = sum(1 for a, b in zip(samples, samples[1:]) if (a < 0) != (b < 0))
    clip = sum(1 for x in samples if abs(x) > 0.98) / len(samples)
    return {
        "dur": len(samples) / sr,
        "rms": rms,
        "peak": peak,
        "zcr": crossings / max(1, len(samples) - 1),
        "clip": clip,
    }


def key_for(path: Path):
    name = path.name
    return name.split("_", 1)[0] if "_" in name else path.stem


def main():
    parser = argparse.ArgumentParser(description="Compare Verbot ASR debug raw_capture and asr_input wav files.")
    parser.add_argument("debug_dir", nargs="?", default="asr_debug_audio")
    args = parser.parse_args()

    root = Path(args.debug_dir)
    raw_dir = root / "raw_capture"
    asr_dir = root / "asr_input"
    if not raw_dir.is_dir() or not asr_dir.is_dir():
        raise SystemExit(f"Missing {raw_dir} or {asr_dir}. Run ./build/Verbot debug first.")

    raw_files = {key_for(p): p for p in raw_dir.glob("*.wav")}
    asr_files = {key_for(p): p for p in asr_dir.glob("*.wav")}
    keys = sorted(set(raw_files) & set(asr_files))
    if not keys:
        raise SystemExit("No matching wav pairs found.")

    print("idx  file                                      dur_raw dur_asr rms_raw rms_asr peak_raw peak_asr zcr_raw zcr_asr")
    for key in keys:
        raw_path = raw_files[key]
        asr_path = asr_files[key]
        raw_sr, raw = read_wav(raw_path)
        asr_sr, asr = read_wav(asr_path)
        if raw_sr != asr_sr:
            raise ValueError(f"Sample-rate mismatch: {raw_path}={raw_sr}, {asr_path}={asr_sr}")
        rm = metrics(raw, raw_sr)
        am = metrics(asr, asr_sr)
        print(
            f"{key:>4} {os.path.basename(asr_path)[:40]:40} "
            f"{rm['dur']:7.2f} {am['dur']:7.2f} "
            f"{rm['rms']:7.4f} {am['rms']:7.4f} "
            f"{rm['peak']:8.4f} {am['peak']:8.4f} "
            f"{rm['zcr']:7.3f} {am['zcr']:7.3f}"
        )


if __name__ == "__main__":
    main()
