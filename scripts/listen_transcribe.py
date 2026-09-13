# /// script
# dependencies = ["bbos", "faster-whisper", "numpy<2"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Standalone mic-to-text diagnostic: prints every phrase it hears, live, so
you can check the mic + Whisper setup is actually understanding you clearly
before trusting it inside full_delivery.py. Also flags whether each segment
would trigger a stop/continue/thank-you command, using the same matching
voice_commands.py uses for real.

Run on the robot: uv run listen_transcribe.py
Stop with Ctrl+C.
"""
import time

import voice_commands as vc


def main():
    model = vc.get_model()
    print("\nListening -- speak near the mic. Ctrl+C to stop.\n")
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
        vc.listen_continuous(on_segment, model=model, on_rms=on_rms)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
