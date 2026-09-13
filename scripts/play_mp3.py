# /// script
# requires-python = "==3.10.*"
# dependencies = ["bbos", "numpy", "miniaudio"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Play an mp3 file on speaker.audio. usage: play_mp3.py [file.mp3] [--volume V]"""
import argparse
import sys
from pathlib import Path

import miniaudio
import numpy as np

from bbos import Config, Type, Writer

CFG = Config("speaker")
HERE = Path(__file__).resolve().parent
DEFAULT_VOLUME = 0.3  # 1.0 = original level, 0.5 = half, etc.


def find_default_mp3():
    mp3s = sorted(HERE.glob("*.mp3"))
    if len(mp3s) == 1:
        return mp3s[0]
    if not mp3s:
        sys.exit(f"no mp3 given and none found in {HERE}")
    sys.exit(f"no mp3 given and {len(mp3s)} found in {HERE}, pass one: "
              + ", ".join(p.name for p in mp3s))


def resolve(arg):
    path = Path(arg)
    if not path.is_absolute() and not path.exists():
        path = HERE / path
    if not path.exists():
        sys.exit(f"no such file: {arg}")
    return path


def load(path, volume):
    # decode_file resamples and remixes channels to match the speaker directly.
    decoded = miniaudio.decode_file(
        str(path),
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=CFG.channels,
        sample_rate=CFG.sample_rate,
    )
    samples = np.array(decoded.samples, dtype=np.float32) * volume
    frames = samples.clip(-32768, 32767).astype(np.int16).reshape(-1, CFG.channels)
    pad = -len(frames) % CFG.chunk_size
    if pad:
        frames = np.concatenate([frames, np.zeros((pad, CFG.channels), np.int16)])
    return frames.reshape(-1, CFG.chunk_size, CFG.channels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="mp3 file to play (default: the only *.mp3 next to this script)")
    ap.add_argument("--volume", type=float, default=DEFAULT_VOLUME,
                     help=f"1.0 = original level, 0.5 = half, etc. (default {DEFAULT_VOLUME})")
    args = ap.parse_args()
    path = resolve(args.file) if args.file else find_default_mp3()
    print(f"playing {path.name} at volume={args.volume}", flush=True)
    with Writer("speaker.audio", Type("speaker_audio")) as w:
        for chunk in load(path, args.volume):
            with w.buf() as b:
                b["audio"] = chunk


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
