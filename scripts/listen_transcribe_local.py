# /// script
# dependencies = ["faster-whisper", "numpy<2", "sounddevice"]
# ///
"""Same diagnostic as listen_transcribe.py, but using this laptop's own mic
and CPU instead of the robot's -- much faster to iterate on phrase-matching
and thresholds without an scp+ssh round trip for every change. The gating
and transcription logic is identical (voice_commands.py's _run_segment_loop
backs both), so whatever works here works on the robot too.

Run: uv run listen_transcribe_local.py
Stop with Ctrl+C.
"""
import time

import voice_commands as vc


def main():
    model = vc.get_model()
    print("\nListening on this laptop's mic -- speak normally. Ctrl+C to stop.\n")
    print(f"(live volume readout below -- need to cross {vc.START_RMS:.0f} to start capturing a phrase)\n")

    last_print = [0.0]

    def on_rms(rms):
        now = time.monotonic()
        if now - last_print[0] > 0.5:
            last_print[0] = now
            marker = "  <-- above threshold, capturing" if rms > vc.START_RMS else ""
            print(f"    volume={rms:.0f}{marker}")

    def on_segment(text):
        tags = vc.describe_tags(text)
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        print(f'[{time.strftime("%H:%M:%S")}] heard: "{text}"{tag_str}')

    try:
        vc.listen_continuous_local_mic(on_segment, model=model, on_rms=on_rms)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
