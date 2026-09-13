# /// script
# dependencies = ["bbos", "numpy<2", "opencv-python"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Orchestrates goto_tag.py (autonomous drive-to-tag, a separate process
owning drive.ctrl) with our pickup sequence (a separate process owning
arm_*.ctrl/torque) via goto_tag's HTTP /status endpoint. The two processes
never touch the same hardware channel, so there's no IPC needed beyond
polling status -- each just drives its own half of the robot.

Flow:
  1. Launch goto_tag.py as a subprocess (real drive, not --dry-run).
  2. Poll /status until target_distance_mm <= TRIGGER_DISTANCE_MM.
  3. At that point, move both arms to the arm_config_4 preset -- this runs
     while goto_tag.py's subprocess is still independently driving.
  4. Keep polling /status until mode == "arrived".
  5. Confirm both wrist markers (reuses full_pickup.py's logic).
  6. Lift, hold, lower back down.
  7. Stop the goto_tag.py subprocess.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import numpy as np
import cv2
from bbos import Reader, Writer, Type, Config

WRIST_CALIB_PATH = {
    "left": "/home/bracketbot/fisheye_calib_wrist_left.json",
    "right": "/home/bracketbot/fisheye_calib_wrist_right.json",
}

GRIPPER_IDX = 7
TARGET_LEFT = np.array([1.9854, 0.0725, -0.0013, 0.1995, 0.3168, 0.2469, 0.0328, -0.2446], dtype=np.float32)
TARGET_RIGHT = np.array([-1.9881, -0.0958, -0.0026, -0.1828, -0.1104, -0.2372, 0.1600, 0.2471], dtype=np.float32)
ARM_NAME = {"left": "arm_left", "right": "arm_right"}
MARKER_ID = {"left": 7, "right": 6}

TRIGGER_DISTANCE_MM = 2000.0
ARRIVE_DISTANCE_MM = 875.0
POST_ARRIVE_SETTLE_S = 0.5  # extra dwell after "arrived" before trusting wrist detections, on top of goto_tag's own thread having already exited (guaranteeing zero twist sent)
GOTO_TAG_PORT = 8013
STATUS_POLL_S = 0.2
MAX_APPROACH_S = 120.0

LIFT_CM = 50.0
HOLD_AT_TOP_S = 3.0
EMPIRICAL_SIGN = {"left": 1.0, "right": -1.0}
WRIST_CONFIRM_WINDOW_S = 20.0

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())


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


def load_wrist_undistort_maps(side):
    """(map1, map2) for cv2.remap, or None if that wrist camera hasn't been
    fisheye-calibrated yet (see calibrate_fisheye.py --camera wrist_<side>).
    Detection still works fine without this -- ArUco id decoding doesn't
    need undistortion, this is purely a robustness improvement for markers
    near the edge of a fisheye frame.

    Both wrist cameras are the same physical model, so if `side`'s own
    calibration is missing, fall back to the other side's file rather than
    skip undistortion entirely -- not per-unit accurate, but closer than raw."""
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
    size = tuple(data["image_size"])  # (w, h)
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


def main():
    side = sys.argv[1] if len(sys.argv) > 1 else "left"

    print("=== Loading wrist camera fisheye calibration (if available) ===")
    wrist_undistort = {s: load_wrist_undistort_maps(s) for s in ("left", "right")}

    print(f"\n=== Launching goto_tag.py (side={side}, arrive={ARRIVE_DISTANCE_MM:.0f}mm) ===")
    goto_proc = subprocess.Popen(
        ["uv", "run", "examples/goto_tag.py", "--side", side, "--marker-id", "5",
         "--target-distance-mm", str(ARRIVE_DISTANCE_MM)],
        cwd="/home/bracketbot/bbapps",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        print(f"=== Waiting for tag, then for distance <= {TRIGGER_DISTANCE_MM:.0f}mm ===")
        t_start = time.monotonic()
        triggered = False
        while time.monotonic() - t_start < MAX_APPROACH_S:
            st = get_status()
            if st is not None:
                print(f"  status: {st}")
                if st.get("mode") == "arrived":
                    print("  arrived before trigger point?? proceeding to pickup anyway.")
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
        cfgs = {"left": Config("arm_left"), "right": Config("arm_right")}
        dof = cfgs["left"].dof
        with Writer("arm_left.torque", Type("arm_torque")) as wt_l, \
             Writer("arm_right.torque", Type("arm_torque")) as wt_r, \
             Writer("arm_left.ctrl", Type("arm_ctrl")) as wc_l, \
             Writer("arm_right.ctrl", Type("arm_ctrl")) as wc_r:

            wt_l["enable"] = np.ones(dof, dtype=np.bool_)
            wt_r["enable"] = np.ones(dof, dtype=np.bool_)
            writers = {"left": wc_l, "right": wc_r}

            starts = {s: get_motor_pos(ARM_NAME[s]) for s in ("left", "right")}
            targets = {"left": TARGET_LEFT, "right": TARGET_RIGHT}
            ramp_both(writers, starts, targets, 5.0)
            time.sleep(0.5)
            for s in ("left", "right"):
                final = get_motor_pos(ARM_NAME[s])
                err = np.abs(final - targets[s]).max()
                print(f"  {s}: max joint error vs preset = {err:.3f} turns")
            print("preset pose commanded (base may still be driving).")

            print(f"\n=== Waiting for goto_tag to report arrived (<= {ARRIVE_DISTANCE_MM:.0f}mm) ===")
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
                print("ABORT: never reached arrival distance in time.")
                return

            # goto_tag's control thread exits its loop and sends a final zero
            # twist the moment it reports "arrived" -- but wait a beat longer
            # here explicitly, so wrist detections are never trusted while
            # there's any chance of residual motion, regardless of how this
            # code evolves later. Wrist-checking (and therefore lifting) only
            # starts after this point -- never earlier, even if a marker
            # happens to be visible mid-approach.
            print(f"\nRobot reported arrived and stopped -- settling {POST_ARRIVE_SETTLE_S:.1f}s before trusting any wrist detection.")
            time.sleep(POST_ARRIVE_SETTLE_S)

            print("\n=== Confirm wrist markers (robot now fully stopped) ===")
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

            print(f"\n=== Raise shoulders {LIFT_CM:.0f}cm (J0 only) ===")
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
            lifted_pos = {s: get_motor_pos(ARM_NAME[s]) for s in ("left", "right")}
            for s in ("left", "right"):
                commanded = lift_targets[s][0] - lift_starts[s][0]
                actual = lifted_pos[s][0] - lift_starts[s][0]
                status = "OK" if abs(actual - commanded) < 0.05 else "MISMATCH"
                print(f"  {s}: commanded={commanded:.4f} actual={actual:.4f} -- {status}")

            print(f"\n=== Hold {HOLD_AT_TOP_S:.0f}s, then lower back ===")
            time.sleep(HOLD_AT_TOP_S)
            ramp_both(writers, lifted_pos, lift_starts, 4.0)
            time.sleep(0.3)
            for s in ("left", "right"):
                final = get_motor_pos(ARM_NAME[s])
                err = abs(final[0] - lift_starts[s][0])
                print(f"  {s}: returned to J0={final[0]:.4f} -- {'OK' if err < 0.05 else 'MISMATCH'}")
        print("\ndone.")
    finally:
        print("\n=== Stopping goto_tag.py ===")
        goto_proc.terminate()
        try:
            goto_proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            goto_proc.kill()


if __name__ == "__main__":
    main()
