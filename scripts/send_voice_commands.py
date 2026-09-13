# /// script
# dependencies = ["faster-whisper", "numpy<2", "sounddevice"]
# ///
"""Runs entirely on this laptop: captures mic audio, transcribes locally
(same pipeline listen_transcribe_local.py already validated), and POSTs each
recognized phrase to full_delivery.py running on the robot. The robot's own
mic turned out to be too finicky about distance/volume for a reliable demo,
so this moves BOTH the mic and the Whisper inference here -- the robot just
receives already-recognized text and reacts (stop/continue/food-request/
thank-you), no audio capture or ML inference on that side at all.

Run: uv run send_voice_commands.py [robot-ip]
(defaults to the robot's Tailscale IP below if not given)
Stop with Ctrl+C.
"""
import sys
import time

import voice_commands as vc

DEFAULT_ROBOT_IP = "100.66.69.194"


def main():
    robot_ip = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROBOT_IP
    robot_url = f"http://{robot_ip}:{vc.VOICE_COMMAND_PORT}/voice"

    model = vc.get_model()
    print(f"\nListening on this laptop's mic -- sending recognized phrases to {robot_url}")
    print("Ctrl+C to stop.\n")

    def on_segment(text):
        tags = vc.describe_tags(text)
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        print(f'[{time.strftime("%H:%M:%S")}] heard: "{text}"{tag_str}')
        try:
            vc.send_text_to_robot(robot_url, text)
        except Exception as e:
            print(f"    (failed to reach robot at {robot_url}: {e!r})")

    try:
        vc.listen_continuous_local_mic(on_segment, model=model)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
