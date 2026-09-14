# BracketBot Waiter Delivery

## DEMO AND PROJECT: [BracketBot Devpost](https://devpost.com/software/bracketbutler)

Autonomous delivery loop for a two-armed BracketBot: hears a food request,
picks up a tray from a chair, drives it to a person, waits for a thank-you,
returns the tray, and lowers the arms back down.

## Scripts

- `scripts/full_delivery.py` — the main loop. Run on the robot.
- `scripts/voice_commands.py` — speech recognition (faster-whisper), shared
  by the robot's HTTP command server and the laptop-side mic capture.
- `scripts/send_voice_commands.py` — run on a laptop near the person.
  Captures mic audio, transcribes locally, POSTs recognized phrases to the
  robot.
- `scripts/listen_transcribe.py` / `listen_transcribe_local.py` — standalone
  checks for what's actually being transcribed (robot mic / laptop mic).
- `scripts/view_aruco.py` — live MJPEG camera feed with ArUco marker overlay.
- `scripts/play_mp3.py` — plays an mp3 over the robot's speaker, used for
  the voice cues and elevator music during the loop.
- `goto_tag.py` — drives the robot up to a specific ArUco tag and stops.

## Running it

On the robot:

    cd ~/bbapps
    uv run full_delivery.py

On a laptop near the person, in a separate terminal:

    uv run send_voice_commands.py [robot-ip]

Say "Hey Bracket, can I have my food?" to start the loop, and "thank you"
once the tray's dropped off.

Distance thresholds, drive speeds, and mp3 volumes are constants near the
top of `full_delivery.py`.

## Calibration

`calibration/` holds a copy of the robot's fisheye camera calibrations
(`fisheye_calib_left.json` — head cam, `fisheye_calib_wrist_left.json` —
both wrist cams, right falls back to left's). Live copies stay at
`/home/bracketbot/` on the robot; these are just a backup.

## Deploy layout

The robot's `~/bbapps` is a separate deploy target, not this checkout — files
are copied over by hand, e.g.:

    scp scripts/full_delivery.py bracketbot@<robot-ip>:~/bbapps/

`goto_tag.py` and `play_mp3.py` live under `~/bbapps/examples/` on the robot;
mp3s live under `~/bbapps/sounds/`.
