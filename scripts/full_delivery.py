# /// script
# dependencies = ["bbos", "numpy<2", "opencv-python"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Full delivery loop, building on orchestrate_pickup.py:
  1. Pick up the tray (same as orchestrate_pickup.py) -- approach pickup tag
     (id=5), arms to preset mid-approach, confirm both wrist markers, lift.
     Tray stays lifted from here on -- back up ~0.75m to clear the chair.
  2. Turn/search for the drop-off tag (id=0) and approach it, stopping
     500-750mm away.
  3. Wait to hear a recognized thank-you (real speech recognition via
     voice_commands.py, driven by send_voice_commands.py on a laptop).
  4. Drop the tray half-way back down (the lifted tray blocks the head
     camera's view of the chair's tag), then turn/search back to the pickup
     tag (id=5) and return to it.
  5. Lower the arms the rest of the way to the original pre-lift height,
     back up ~1m, and end.
"""
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.request
import numpy as np
import cv2
from bbos import Reader, Writer, Type, Config

import voice_commands as vc

WRIST_CALIB_PATH = {
    "left": "/home/bracketbot/fisheye_calib_wrist_left.json",
    "right": "/home/bracketbot/fisheye_calib_wrist_right.json",
}

GRIPPER_IDX = 7
# Single approach preset, from mimic/main.py's arm_config_6_new_chair_height.json.
TARGET_LEFT = np.array([1.9854, 0.0725, -0.0013, 0.1995, 0.3168, 0.2469, 0.0328, -0.2446], dtype=np.float32)
TARGET_RIGHT = np.array([-1.9881, -0.0958, -0.0026, -0.1828, -0.1104, -0.2372, 0.1600, 0.2471], dtype=np.float32)
ARM_NAME = {"left": "arm_left", "right": "arm_right"}
MARKER_ID = {"left": 7, "right": 6}

PICKUP_TAG_ID = 5
DROPOFF_TAG_ID = 0
# Physical printed size (side length, meters) of each tag -- goto_tag.py needs
# the real size to convert solvePnP's pose into an actual distance; getting
# this wrong scales every distance estimate by the size ratio. Pickup tag
# reverted back to 69mm; drop-off tag is still the 120mm reprint (under-tray
# wrist tags are untouched).
MARKER_SIZE_M = {PICKUP_TAG_ID: 0.069, DROPOFF_TAG_ID: 0.120}
TRIGGER_DISTANCE_MM = 2000.0
PICKUP_ARRIVE_MM = 815.0
DROPOFF_ARRIVE_MM = 650.0  # 300mm came in too close to the actual person; widened per feedback (target range 500-750mm)
RETURN_ARRIVE_MM = 820.0
SET_DOWN_TRIGGER_MM = 820.0  # start the full lower once the return leg gets this close, while still driving
POST_ARRIVE_SETTLE_S = 0.5

# goto_tag.py speed defaults are max-v=0.15 m/s, max-w=0.5 rad/s, search-w=0.3
# rad/s -- bumped rotation a bit across all legs, and forward speed a bit more
# just for the drop-off approach (Leg 2), per feedback.
MAX_W_RAD_S = 0.65
SEARCH_W_RAD_S = 0.4
DROPOFF_MAX_V_MPS = 0.22
PICKUP_MAX_V_MPS = 0.22  # sped up per feedback -- default is 0.15

SOUND_DIR = "/home/bracketbot/bbapps/sounds"
COMING_RIGHT_UP_MP3 = f"{SOUND_DIR}/coming_right_up.mp3"
ELEVATOR_MUSIC_MP3 = f"{SOUND_DIR}/elevator_music.mp3"
BON_APPETIT_MP3 = f"{SOUND_DIR}/bon_appetit.mp3"
COMING_RIGHT_UP_VOLUME = 1.0  # louder than play_mp3.py's own default (0.3)
BON_APPETIT_VOLUME = 1.6  # bumped further per feedback -- above 1.0 clips softly but reads louder
ELEVATOR_MUSIC_VOLUME = 0.5  # bumped up too, but less than the cues
GOTO_TAG_PORT = 8013
STATUS_POLL_S = 0.2
MAX_APPROACH_S = 120.0

LIFT_CM = 50.0
PARTIAL_LOWER_CM = 30.0  # how far down (of LIFT_CM) before the return drive; the rest happens once back
EMPIRICAL_SIGN = {"left": 1.0, "right": -1.0}
WRIST_CONFIRM_WINDOW_S = 20.0

BACKUP_DISTANCE_M = 0.75
BACKUP_SPEED_MPS = 0.12
BACKUP_MAX_DURATION_S = 30.0
RETURN_BACKUP_DISTANCE_M = 1.0

VOICE_TIMEOUT_S = 45.0

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# Loosened from defaults -- the right wrist camera views the tray's markers
# at a steep angle, which foreshortens the square into a thin quadrilateral
# and makes bit decoding noisier at the edges. The marker ID must still
# match exactly afterward, so this doesn't risk false-positive IDs, just
# accepts more perspective-distorted/noisier candidates as detections at all.
_ARUCO_PARAMS = cv2.aruco.DetectorParameters()
_ARUCO_PARAMS.polygonalApproxAccuracyRate = 0.045  # default 0.03 -- tolerate a less-square quad
_ARUCO_PARAMS.minMarkerPerimeterRate = 0.025  # default 0.03 -- accept a smaller apparent marker
_ARUCO_PARAMS.errorCorrectionRate = 0.7  # default 0.6 -- allow more bit-decode errors
_ARUCO_PARAMS.maxErroneousBitsInBorderRate = 0.425  # default 0.35 -- tolerate noisier border bits
_ARUCO_PARAMS.perspectiveRemoveIgnoredMarginPerCell = 0.165  # default 0.13 -- ignore more per-cell edge noise
_ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
DETECTOR = cv2.aruco.ArucoDetector(DICT, _ARUCO_PARAMS)


def get_status():
    try:
        with urllib.request.urlopen(f"http://localhost:{GOTO_TAG_PORT}/status", timeout=1.0) as r:
            return json.loads(r.read())
    except Exception:
        return None


def get_motor_pos(arm):
    with Reader(f"{arm}.state") as r:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if r.ready():
                return r.data["pos"].copy()
            time.sleep(0.01)
    raise RuntimeError(f"no state for {arm}")


def ramp_both(writers, starts, targets, duration):
    wait_deadline = time.monotonic() + 5.0
    while time.monotonic() < wait_deadline:
        if writers["left"].ready() and writers["right"].ready():
            break
        time.sleep(0.02)
    else:
        raise RuntimeError("ctrl writer(s) never became ready")
    write_counts = {"left": 0, "right": 0}
    t0 = time.monotonic()
    while True:
        a = min((time.monotonic() - t0) / duration, 1.0)
        for side in ("left", "right"):
            if writers[side].ready():
                cmd = starts[side] + a * (targets[side] - starts[side])
                writers[side]["pos"] = cmd.astype(np.float32)
                write_counts[side] += 1
        if a >= 1.0:
            break
        time.sleep(0.01)
    for side, count in write_counts.items():
        if count == 0:
            print(f"  WARNING: {side} ctrl writer never accepted a single write.")


def ramp_hold_target(hold_target, targets, duration):
    # Retargets the continuously-refreshing hold_target dict in place instead
    # of stopping hold_loop and running a separate ramp_both -- stopping the
    # refresh even briefly (stop_hold + join, however short) was observed to
    # make the arm stop responding to any further position commands for the
    # rest of that ramp, so a supposed 4s move silently did nothing and the
    # arm just stayed pinned at its last held position. hold_loop keeps
    # writing every 10ms throughout; only the value it writes changes here.
    starts = {s: hold_target[s].copy() for s in ("left", "right")}
    t0 = time.monotonic()
    while True:
        a = min((time.monotonic() - t0) / duration, 1.0)
        for s in ("left", "right"):
            hold_target[s] = (starts[s] + a * (targets[s] - starts[s])).astype(np.float32)
        if a >= 1.0:
            break
        time.sleep(0.01)


def ramp_hold_target_until_converged(hold_target, targets, duration, label, max_attempts=3, tol=0.05):
    """ramp_hold_target, but verified against the arm's actual measured
    position afterward and retried if it didn't get there -- observed
    repeatedly that a retarget issued after the arm has been holding one
    fixed position for a long stretch (e.g. through a whole drive leg) can
    silently do nothing even though hold_loop never stops writing; root
    cause not confirmed, so retrying with fresh writes is the pragmatic
    mitigation until it is. Returns True once both arms are within `tol`
    turns of target, False if still not converged after max_attempts."""
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            # Re-seed from the arm's REAL measured position, not hold_target --
            # by the time attempt 1's ramp finishes, hold_target already
            # equals targets exactly, so re-ramping from hold_target is a
            # no-op (starts == targets, no motion commanded) that just
            # re-asserts the same unreached setpoint. Seeding from the actual
            # position instead makes every retry a fresh corrective move.
            for s in ("left", "right"):
                hold_target[s] = get_motor_pos(ARM_NAME[s])
        ramp_hold_target(hold_target, targets, duration)
        time.sleep(0.3)
        errs = {}
        for s in ("left", "right"):
            actual = get_motor_pos(ARM_NAME[s])
            errs[s] = abs(actual[0] - targets[s][0])
            print(f"  {s}: at J0={actual[0]:.4f} ({label} target={targets[s][0]:.4f}) -- "
                  f"{'OK' if errs[s] < tol else 'MISMATCH'} (attempt {attempt}/{max_attempts})")
        if all(e < tol for e in errs.values()):
            return True
    return False


def load_wrist_undistort_maps(side):
    path = WRIST_CALIB_PATH[side]
    if not os.path.exists(path):
        other_side = "left" if side == "right" else "right"
        fallback = WRIST_CALIB_PATH[other_side]
        if os.path.exists(fallback):
            print(f"  no calibration at {path} -- falling back to {other_side}'s calibration ({fallback})")
            path = fallback
        else:
            print(f"  (no fisheye calibration at {path} -- detecting on raw distorted frame for {side})")
            return None
    data = json.loads(open(path).read())
    K = np.array(data["K"], dtype=np.float64)
    D = np.array(data["D"], dtype=np.float64).reshape(-1, 1)
    size = tuple(data["image_size"])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, size, cv2.CV_16SC2)
    print(f"  loaded wrist fisheye calibration for {side} from {path} (image_size={size})")
    return map1, map2


def check_wrist_marker(side, undistort_maps=None):
    with Reader(f"camera.{side}.jpeg") as r:
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            if r.ready():
                n = int(r.data["jpeg_len"])
                buf = r.data["jpeg"][:n].copy()
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if undistort_maps is not None:
                    map1, map2 = undistort_maps
                    img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = DETECTOR.detectMarkers(gray)
                return ids is not None and MARKER_ID[side] in ids.flatten().tolist()
            time.sleep(0.02)
    return False


def launch_goto_tag(side, marker_id, target_distance_mm, max_v=None):
    args = ["uv", "run", "examples/goto_tag.py", "--side", side, "--marker-id", str(marker_id),
            "--target-distance-mm", str(target_distance_mm),
            "--marker-size", str(MARKER_SIZE_M[marker_id]),
            "--max-w", str(MAX_W_RAD_S), "--search-w", str(SEARCH_W_RAD_S)]
    if max_v is not None:
        args += ["--max-v", str(max_v)]
    return subprocess.Popen(
        args,
        cwd="/home/bracketbot/bbapps",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def stop_goto_tag(proc):
    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()


def drive_to_tag_simple(side, marker_id, target_distance_mm, label, max_v=None):
    """Launch goto_tag targeting marker_id, wait for 'arrived', stop it. No
    concurrent arm action -- used for the drop-off and return legs, which are
    pure navigation."""
    print(f"\n=== {label}: driving to tag id={marker_id}, arrive={target_distance_mm:.0f}mm ===")
    proc = launch_goto_tag(side, marker_id, target_distance_mm, max_v=max_v)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < MAX_APPROACH_S:
            st = get_status()
            if st is not None:
                print(f"  status: {st}")
                if st.get("mode") == "arrived":
                    return True
            time.sleep(STATUS_POLL_S)
        print(f"  ABORT: never arrived at tag {marker_id} within {MAX_APPROACH_S:.0f}s.")
        return False
    finally:
        stop_goto_tag(proc)


def back_up_by_odometry(distance_m=BACKUP_DISTANCE_M, speed_mps=BACKUP_SPEED_MPS,
                         max_duration_s=BACKUP_MAX_DURATION_S):
    """Reverse in a straight line until wheel odometry says we've covered
    distance_m -- dead-reckoning via drive.state.vel (measured wheel turns/s,
    converted to m/s the same way teleop.py does), not a blind timed guess.
    drive.ctrl needs continuous refresh (its own daemon-side watchdog stops
    the robot if commands stop arriving for 100ms, per goto_tag.py's own
    docstring) -- same lesson as the arm hold, applied here too."""
    print(f"\n=== Backing up ~{distance_m:.1f}m before turning to find the drop-off tag ===")
    drive_cfg = Config("drive")
    wheel_diam = drive_cfg.wheel_diam
    with Reader("drive.state") as r_state, Writer("drive.ctrl", Type("drive_ctrl")) as w_ctrl:
        traveled = 0.0
        t_start = time.monotonic()
        last_t = t_start
        while traveled < distance_m and time.monotonic() - t_start < max_duration_s:
            now = time.monotonic()
            dt = now - last_t
            last_t = now
            if r_state.ready():
                vel = r_state.data["vel"]
                v_left_mps = float(vel[0]) * wheel_diam * math.pi
                v_right_mps = float(vel[1]) * wheel_diam * math.pi
                traveled += abs((v_left_mps + v_right_mps) / 2.0) * dt
            if w_ctrl.ready():
                w_ctrl["twist"] = np.array([-speed_mps, 0.0], dtype=np.float32)
            time.sleep(0.02)
        if w_ctrl.ready():
            w_ctrl["twist"] = np.array([0.0, 0.0], dtype=np.float32)
        print(f"  backed up ~{traveled:.2f}m (target {distance_m:.1f}m)"
              + ("" if traveled >= distance_m else " -- hit max duration before reaching target"))


class AudioPlayer:
    """Plays mp3s via examples/play_mp3.py subprocesses (that script opens
    bbos's speaker.audio Writer, which is exclusive -- only one play_mp3.py
    may run at a time, so every call here goes through this one object)."""

    def __init__(self):
        self._proc = None

    def play(self, path, blocking=True, volume=None):
        args = ["uv", "run", "examples/play_mp3.py", path]
        if volume is not None:
            args += ["--volume", str(volume)]
        self._proc = subprocess.Popen(
            args,
            cwd="/home/bracketbot/bbapps",
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if blocking:
            self._proc.wait()

    def stop(self):
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()


def start_play_then_loop(player, intro_path, loop_path, intro_volume=None, loop_volume=None):
    """Plays intro_path once (if given), then loops loop_path repeatedly
    until stopped -- all in a background thread so it never blocks driving
    or arm motion."""
    stop_event = threading.Event()

    def worker():
        if intro_path:
            player.play(intro_path, blocking=True, volume=intro_volume)
        while not stop_event.is_set():
            player.play(loop_path, blocking=False, volume=loop_volume)
            while player._proc.poll() is None:
                if stop_event.is_set():
                    player.stop()
                    return
                time.sleep(0.1)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return stop_event, thread


def stop_play_then_loop(player, stop_event, thread):
    stop_event.set()
    player.stop()
    thread.join(timeout=5.0)


def start_voice_thread(waiting_for_thanks, thanks_heard, food_requested):
    """Starts the single background voice-command receiver for the whole
    script's lifetime -- food-request and thank-you are dispatched from
    here.

    Listens over HTTP (run_command_server) instead of capturing/
    transcribing audio itself: run send_voice_commands.py on a laptop near
    whoever's speaking, and it does the mic capture + Whisper transcription
    there, POSTing each recognized phrase here.

    - food-request ("I want my food") is the starting gate: recognized at
      all times, but main() only ever waits on it once, right at the start,
      before Leg 1 begins.
    - thank-you is only recognized while waiting_for_thanks is set (Leg 3
      sets it right before waiting, clears it right after).
    """

    def on_segment(text):
        print(f'  [voice] heard: "{text}"')
        if not food_requested.is_set() and vc.matches_any(text, vc.FOOD_REQUEST_PHRASES):
            print("  >>> food request recognized -- starting delivery.")
            food_requested.set()
        if waiting_for_thanks.is_set() and vc.matches_any(text, vc.THANKS_PHRASES):
            print("  >>> thank-you recognized.")
            thanks_heard.set()

    thread = threading.Thread(target=vc.run_command_server, args=(on_segment,), daemon=True)
    thread.start()
    print(f"  [voice] listening for commands from send_voice_commands.py on port {vc.VOICE_COMMAND_PORT}")
    return thread


audio = None  # module-level so a Ctrl+C in __main__ can reach it and stop playback


def main():
    global audio
    side = sys.argv[1] if len(sys.argv) > 1 else "left"

    print("=== Loading wrist camera fisheye calibration (if available) ===")
    wrist_undistort = {s: load_wrist_undistort_maps(s) for s in ("left", "right")}

    # Voice commands run for the whole script, starting now so the food
    # request works even before Leg 1's initial approach begins.
    waiting_for_thanks = threading.Event()
    thanks_heard = threading.Event()
    food_requested = threading.Event()
    start_voice_thread(waiting_for_thanks, thanks_heard, food_requested)

    print("\n=== Waiting to hear a food request (e.g. \"I want my food\") before starting ===")
    food_requested.wait()
    t_food_recognized = time.monotonic()

    audio = AudioPlayer()
    audio_stop, audio_thread = start_play_then_loop(audio, COMING_RIGHT_UP_MP3, ELEVATOR_MUSIC_MP3,
                                                     intro_volume=COMING_RIGHT_UP_VOLUME, loop_volume=ELEVATOR_MUSIC_VOLUME)

    # ---- Leg 1: pick up the tray ----
    print(f"\n=== Leg 1: pick up tray at tag id={PICKUP_TAG_ID} ===")
    proc = launch_goto_tag(side, PICKUP_TAG_ID, PICKUP_ARRIVE_MM, max_v=PICKUP_MAX_V_MPS)
    print(f"  [timing] goto_tag subprocess launched {time.monotonic() - t_food_recognized:.2f}s after food request recognized")
    cfgs = {"left": Config("arm_left"), "right": Config("arm_right")}
    dof = cfgs["left"].dof
    with Writer("arm_left.torque", Type("arm_torque")) as wt_l, \
         Writer("arm_right.torque", Type("arm_torque")) as wt_r, \
         Writer("arm_left.ctrl", Type("arm_ctrl")) as wc_l, \
         Writer("arm_right.ctrl", Type("arm_ctrl")) as wc_r:

        wt_l["enable"] = np.ones(dof, dtype=np.bool_)
        wt_r["enable"] = np.ones(dof, dtype=np.bool_)
        writers = {"left": wc_l, "right": wc_r}

        # try/finally nested INSIDE the with block (not wrapping it) so
        # stop_goto_tag() runs while the arm writers are still open --
        # closing them is only ever safe here since nothing has lifted yet.
        try:
            t_start = time.monotonic()
            triggered = False
            first_status_logged = False
            while time.monotonic() - t_start < MAX_APPROACH_S:
                st = get_status()
                if st is not None:
                    if not first_status_logged:
                        first_status_logged = True
                        print(f"  [timing] first status from goto_tag {time.monotonic() - t_food_recognized:.2f}s "
                              f"after food request recognized (subprocess startup + camera init)")
                    print(f"  status: {st}")
                    if st.get("mode") == "arrived":
                        triggered = True
                        break
                    dist = st.get("target_distance_mm")
                    if dist is not None and dist <= TRIGGER_DISTANCE_MM:
                        triggered = True
                        break
                time.sleep(STATUS_POLL_S)
            if not triggered:
                print(f"ABORT: never got within {TRIGGER_DISTANCE_MM:.0f}mm within {MAX_APPROACH_S:.0f}s.")
                return

            print("\n=== Trigger hit: moving arms to preset WHILE still driving ===")
            starts = {s: get_motor_pos(ARM_NAME[s]) for s in ("left", "right")}
            targets = {"left": TARGET_LEFT, "right": TARGET_RIGHT}
            ramp_both(writers, starts, targets, 4.0)
            time.sleep(1.0)
            for s in ("left", "right"):
                final = get_motor_pos(ARM_NAME[s])
                err = np.abs(final - targets[s]).max()
                print(f"  {s}: max joint error vs preset = {err:.3f} turns")
            print("preset pose commanded (base may still be driving).")

            print(f"\n=== Waiting for goto_tag to report arrived (<= {PICKUP_ARRIVE_MM:.0f}mm) ===")
            t_arrive_wait = time.monotonic()
            arrived = False
            while time.monotonic() - t_arrive_wait < MAX_APPROACH_S:
                st = get_status()
                if st is not None:
                    print(f"  status: {st}")
                    if st.get("mode") == "arrived":
                        arrived = True
                        break
                time.sleep(STATUS_POLL_S)
            if not arrived:
                print("ABORT: never reached pickup arrival distance in time.")
                return
        finally:
            stop_goto_tag(proc)

        print(f"\nRobot reported arrived and stopped -- settling {POST_ARRIVE_SETTLE_S:.1f}s.")
        time.sleep(POST_ARRIVE_SETTLE_S)

        print(f"\n=== Confirm wrist markers (up to {WRIST_CONFIRM_WINDOW_S:.0f}s) ===")
        found = {"left": False, "right": False}
        t_confirm = time.monotonic()
        while time.monotonic() - t_confirm < WRIST_CONFIRM_WINDOW_S:
            if not found["left"]:
                found["left"] = check_wrist_marker("left", wrist_undistort["left"])
            if not found["right"]:
                found["right"] = check_wrist_marker("right", wrist_undistort["right"])
            print(f"    wrist status: left={found['left']} right={found['right']}")
            if found["left"] and found["right"]:
                print("Both wrist markers confirmed.")
                break
            time.sleep(0.2)
        if not (found["left"] and found["right"]):
            print(f"\nNOT lifting -- wrist confirmation incomplete: {found}")
            return

        print(f"\n=== Raise shoulders {LIFT_CM:.0f}cm (J0 only) -- tray stays lifted from here on ===")
        lift_starts = {s: get_motor_pos(ARM_NAME[s]) for s in ("left", "right")}
        lift_targets = {}
        lift_m = LIFT_CM / 100.0
        for s, cfg in cfgs.items():
            denom = float(cfg.ik_sign[0]) * cfg.wheel_radius * 2 * np.pi
            delta = EMPIRICAL_SIGN[s] * lift_m / denom
            t = lift_starts[s].copy()
            t[0] += delta
            lift_targets[s] = t.astype(np.float32)
        ramp_both(writers, lift_starts, lift_targets, 4.0)
        time.sleep(0.3)
        hold_target = {}
        for s in ("left", "right"):
            final = get_motor_pos(ARM_NAME[s])
            hold_target[s] = final.copy()
            commanded = lift_targets[s][0] - lift_starts[s][0]
            actual = final[0] - lift_starts[s][0]
            status = "OK" if abs(actual - commanded) < 0.05 else "MISMATCH"
            print(f"  {s}: commanded={commanded:.4f} actual={actual:.4f} -- {status}")

        # From here on the arm writers must NEVER go quiet while the tray is
        # lifted: closing/pausing them earlier (a bug in a prior version of
        # this script) let the arm sag and drop the tray, because holding
        # against full gravity apparently needs continuous refresh -- unlike
        # resting near home, which tolerated a disconnect fine (see
        # hold_test.py). A background thread keeps re-sending hold_target for
        # the rest of this process's life; the outer `with Writer(...)` block
        # (still open around this whole function) is never allowed to exit
        # normally while that's the case -- every path below ends in
        # hold_forever() instead of returning.
        #
        # Also re-asserts torque "enable" every cycle, not just position --
        # observed repeatedly that a retarget issued after a long hold (e.g.
        # after the ~1-2 minute Leg 4 drive) would silently do nothing, arm
        # frozen exactly at its last position, even though position writes
        # never stopped. Every other realtime channel in this system needs
        # continuous refresh or a daemon-side watchdog cuts it off; "enable"
        # was the one signal here we were only ever setting once, at the
        # very start, so it's the leading suspect for that silent freeze.
        torque_writers = {"left": wt_l, "right": wt_r}
        stop_hold = threading.Event()

        def hold_loop():
            while not stop_hold.is_set():
                for s in ("left", "right"):
                    if torque_writers[s].ready():
                        torque_writers[s]["enable"] = np.ones(dof, dtype=np.bool_)
                    if writers[s].ready():
                        writers[s]["pos"] = hold_target[s].astype(np.float32)
                time.sleep(0.01)

        hold_thread = threading.Thread(target=hold_loop, daemon=True)
        hold_thread.start()

        def hold_forever(message):
            print(f"\n{message}")
            print("Holding tray indefinitely at its current lifted position -- will NOT lower or "
                  "release on its own. Press Ctrl+C to stop this script (doing so WILL let the arm "
                  "drop, since active holding requires this process to keep running).")
            try:
                t0 = time.monotonic()
                while True:
                    time.sleep(1.0)
                    if int(time.monotonic() - t0) % 10 == 0:
                        print(f"  ... still holding ({time.monotonic() - t0:.0f}s elapsed) ...")
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
            finally:
                stop_hold.set()
                hold_thread.join(timeout=1.0)

        print("\n=== Pickup complete. Tray is lifted and being actively held. Proceeding to drop-off. ===")

        back_up_by_odometry()

        # ---- Leg 2: navigate to drop-off tag ----
        if not drive_to_tag_simple(side, DROPOFF_TAG_ID, DROPOFF_ARRIVE_MM, "Leg 2", max_v=DROPOFF_MAX_V_MPS):
            hold_forever("ABORT: could not reach drop-off tag.")
            return

        # ---- Leg 3: wait for a recognized thank-you ----
        print(f"\n=== Waiting up to {VOICE_TIMEOUT_S:.0f}s to hear a thank-you ===")
        thanks_heard.clear()
        waiting_for_thanks.set()
        got_thanks = thanks_heard.wait(timeout=VOICE_TIMEOUT_S)
        waiting_for_thanks.clear()
        if not got_thanks:
            print("  no thank-you heard within timeout -- proceeding anyway.")

        stop_play_then_loop(audio, audio_stop, audio_thread)
        audio_stop, audio_thread = start_play_then_loop(audio, BON_APPETIT_MP3, ELEVATOR_MUSIC_MP3,
                                                         intro_volume=BON_APPETIT_VOLUME, loop_volume=ELEVATOR_MUSIC_VOLUME)

        # Drop the tray part-way (PARTIAL_LOWER_CM of the LIFT_CM total)
        # BEFORE attempting the return drive -- not after, and not all the
        # way down yet. The fully-lifted tray sits right in front of the
        # head camera and can block its view of the chair's tag entirely,
        # which is almost certainly why the return leg sometimes got stuck
        # searching forever in earlier runs: goto_tag can't navigate to a
        # tag it can never see. Coming down most of the way is enough to
        # clear the camera's view while leaving only a small remainder for
        # the full lower once we've actually arrived back.
        print(f"\n=== Lowering tray {PARTIAL_LOWER_CM:.0f}cm of {LIFT_CM:.0f}cm before returning "
              "(tray was blocking the chair's tag) ===")
        full_up = {s: hold_target[s].copy() for s in ("left", "right")}
        partial_frac = PARTIAL_LOWER_CM / LIFT_CM
        partial_target = {s: (full_up[s] + partial_frac * (lift_starts[s] - full_up[s])).astype(np.float32)
                           for s in ("left", "right")}
        ramp_hold_target_until_converged(hold_target, partial_target, 4.0, "partial-lower")

        # ---- Leg 4: return to pickup tag ----
        # Tray is only half-lowered, still under active holding -- on
        # failure here we must still not let the writers close. Inlined
        # (like Leg 1) rather than using drive_to_tag_simple so the set-down
        # lower can trigger WHILE STILL DRIVING once within
        # SET_DOWN_TRIGGER_MM of the tag, the same "do arm stuff mid-
        # approach, don't wait for a full stop" pattern Leg 1 already uses.
        print(f"\n=== Leg 4 (return): driving to tag id={PICKUP_TAG_ID}, arrive={RETURN_ARRIVE_MM:.0f}mm ===")
        proc = launch_goto_tag(side, PICKUP_TAG_ID, RETURN_ARRIVE_MM)
        set_down_triggered = False
        try:
            t_return = time.monotonic()
            arrived = False
            while time.monotonic() - t_return < MAX_APPROACH_S:
                st = get_status()
                if st is not None:
                    print(f"  status: {st}")
                    if not set_down_triggered:
                        dist = st.get("target_distance_mm")
                        if dist is not None and dist <= SET_DOWN_TRIGGER_MM:
                            set_down_triggered = True
                            print(f"\n=== Within {SET_DOWN_TRIGGER_MM:.0f}mm of the chair tag -- "
                                  "lowering arms all the way WHILE still driving ===")
                            if not ramp_hold_target_until_converged(hold_target, lift_starts, 4.0, "full-lower"):
                                print("  WARNING: full-lower did not converge after retries -- "
                                      "tray may not be fully down.")
                    if st.get("mode") == "arrived":
                        arrived = True
                        break
                time.sleep(STATUS_POLL_S)
        finally:
            stop_goto_tag(proc)
        if not arrived:
            hold_forever("ABORT: could not return to pickup tag.")
            return

        # Defensive fallback: shouldn't happen since RETURN_ARRIVE_MM <=
        # SET_DOWN_TRIGGER_MM (arrival is always within the trigger
        # distance), but don't skip lowering the tray if it somehow does.
        if not set_down_triggered:
            print("\n=== Arrived without crossing the set-down trigger distance -- lowering arms now ===")
            ramp_hold_target_until_converged(hold_target, lift_starts, 4.0, "full-lower")

        # Tray is fully back at its original pre-lift height now -- the same
        # "safe to stop holding" point already validated in
        # orchestrate_pickup.py/full_pickup.py, so it's fine to stop the
        # refresh thread and let the writers close at the end of this
        # function.
        stop_hold.set()
        hold_thread.join(timeout=1.0)

        back_up_by_odometry(distance_m=RETURN_BACKUP_DISTANCE_M)

        stop_play_then_loop(audio, audio_stop, audio_thread)

        print("\n=== Delivery loop complete. Tray lowered back to original height. ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # A daemon audio thread just relaunches the next mp3 as soon as the
        # current one exits -- if the currently-playing play_mp3.py subprocess
        # doesn't reliably get Ctrl+C's SIGINT itself (e.g. it's a grandchild
        # via `uv run`, not a direct child of the foreground process group),
        # music keeps looping even after this script has exited. Killing it
        # by PID here doesn't depend on that signal propagation at all.
        if audio is not None:
            audio.stop()
        print("\nInterrupted -- audio playback stopped.")
