# /// script
# dependencies = ["bbos", "numpy<2", "opencv-python"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Full approach-and-pickup sequence:
  0. Both arms move ONCE to the arm_config_2 preset pose and stay frozen there
     for the entire rest of this script -- no further arm changes until the lift.
  1. Drive slowly toward the chair marker (head cam, id=5), steering with a
     simple proportional correction on its pixel x-offset from center.
  2. Every loop tick, also check both wrist cams for their markers (left=7,
     right=6). As soon as BOTH have been seen, stop driving immediately.
  3. Safety stops: if the head marker is lost, stop driving until reacquired;
     if the marker gets very large (very close), stop as a proximity backstop;
     if neither wrist marker is confirmed within MAX_DURATION_S, abort (stop
     driving, do not lift).
  4. Only if both wrist markers confirmed: raise both shoulders (J0 only,
     arm shape frozen) by LIFT_CM.
"""
import time
import numpy as np
import cv2
from bbos import Reader, Writer, Type, Config

GRIPPER_IDX = 7
TARGET_LEFT = np.array([0.5498, 0.0491, 0.0192, 0.2061, 0.0106, 0.2262, -0.2384, -0.2620], dtype=np.float32)
TARGET_RIGHT = np.array([-0.5769, -0.0714, 0.0311, -0.1765, -0.0830, -0.2426, 0.2457, 0.2515], dtype=np.float32)
ARM_NAME = {"left": "arm_left", "right": "arm_right"}
MARKER_ID = {"left": 7, "right": 6}
HEAD_MARKER_ID = 5

FORWARD_SPEED = 0.06       # m/s, slow and conservative
STEER_GAIN = 0.6           # rad/s at full normalized pixel error
MAX_ANGULAR = 0.4          # rad/s cap
# Calibrated against a manually-positioned reference shot: id=5 measured
# width=50px, height=47px at the exact distance we want the base to stop.
TARGET_WIDTH_PX = 50.0
TARGET_WIDTH_TOL_PX = 4.0
PROXIMITY_WIDTH_PX = 80.0  # hard backstop well past target, in case of overshoot
MAX_DURATION_S = 40.0
LOST_MARKER_GRACE_S = 3.0  # keep trying this long after losing the head marker before giving up
WRIST_CONFIRM_WINDOW_S = 6.0  # after stopping, how long to keep trying to confirm wrist markers

LIFT_CM = 30.0
EMPIRICAL_SIGN = {"left": 1.0, "right": -1.0}  # see mimic_and_lift.py

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())


def get_motor_pos(arm):
    with Reader(f"{arm}.state") as r:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if r.ready():
                return r.data["pos"].copy()
            time.sleep(0.01)
    raise RuntimeError(f"no state for {arm}")


def ramp_both(writers, starts, targets, duration):
    # Wait for both writers to actually be ready before starting the clock --
    # a writer that's not ready yet (e.g. right after a fresh daemon restart)
    # would otherwise make this loop silently do nothing for the whole
    # duration while still "succeeding" on the elapsed-time exit condition.
    wait_deadline = time.monotonic() + 5.0
    while time.monotonic() < wait_deadline:
        if writers["left"].ready() and writers["right"].ready():
            break
        time.sleep(0.02)
    else:
        raise RuntimeError("ctrl writer(s) never became ready -- refusing to silently no-op the move")

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
            print(f"  WARNING: {side} ctrl writer never accepted a single write during this ramp.")


def read_head_marker(r_head):
    if not r_head.ready():
        return None
    n = int(r_head.data["jpeg_len"])
    buf = r_head.data["jpeg"][:n].copy()
    stereo = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    half = stereo.shape[1] // 2
    gray = cv2.cvtColor(stereo[:, :half], cv2.COLOR_BGR2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(gray)
    if ids is None or HEAD_MARKER_ID not in ids.flatten().tolist():
        return None
    idx = ids.flatten().tolist().index(HEAD_MARKER_ID)
    c = corners[idx][0]
    cx = float(c[:, 0].mean())
    width_px = float(c[:, 0].max() - c[:, 0].min())
    frame_w = gray.shape[1]
    return cx, width_px, frame_w


def check_wrist_marker(side):
    with Reader(f"camera.{side}.jpeg") as r:
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            if r.ready():
                n = int(r.data["jpeg_len"])
                buf = r.data["jpeg"][:n].copy()
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = DETECTOR.detectMarkers(gray)
                return ids is not None and MARKER_ID[side] in ids.flatten().tolist()
            time.sleep(0.02)
    return False


def main():
    cfgs = {"left": Config("arm_left"), "right": Config("arm_right")}
    dof = cfgs["left"].dof

    with Writer("arm_left.torque", Type("arm_torque")) as wt_l, \
         Writer("arm_right.torque", Type("arm_torque")) as wt_r, \
         Writer("arm_left.ctrl", Type("arm_ctrl")) as wc_l, \
         Writer("arm_right.ctrl", Type("arm_ctrl")) as wc_r, \
         Writer("drive.ctrl", Type("drive_ctrl")) as w_drive, \
         Reader("camera.head.jpeg") as r_head:

        wt_l["enable"] = np.ones(dof, dtype=np.bool_)
        wt_r["enable"] = np.ones(dof, dtype=np.bool_)
        writers = {"left": wc_l, "right": wc_r}

        print("=== Step 0: arms to preset pose (frozen for rest of run) ===")
        starts = {s: get_motor_pos(ARM_NAME[s]) for s in ("left", "right")}
        targets = {"left": TARGET_LEFT, "right": TARGET_RIGHT}
        max_disp = max(np.abs(targets[s] - starts[s]).max() for s in ("left", "right"))
        print(f"  max joint displacement: {max_disp:.3f} turns")
        ramp_both(writers, starts, targets, 10.0)  # generous: this move happens once, no rush
        time.sleep(1.0)
        for side in ("left", "right"):
            final = get_motor_pos(ARM_NAME[side])
            err = np.abs(final - targets[side]).max()
            if err > 0.05:
                raise RuntimeError(f"{side} arm did not reach preset pose (max joint error {err:.3f} turns) -- aborting, will not drive.")
        print("preset pose reached and verified, arms frozen from here on.")

        print("\n=== Step 1: drive toward chair marker, watching wrist cams ===")

        def stop_drive():
            if w_drive.ready():
                w_drive["twist"] = np.array([0.0, 0.0], dtype=np.float32)

        t_start = time.monotonic()
        last_seen_head = t_start
        arrived = False
        try:
            while True:
                now = time.monotonic()
                if now - t_start > MAX_DURATION_S:
                    print(f"\nABORT: {MAX_DURATION_S}s elapsed without reaching target distance.")
                    stop_drive()
                    return

                head_result = read_head_marker(r_head)
                if head_result is None:
                    if now - last_seen_head > LOST_MARKER_GRACE_S:
                        print("  head marker lost too long -- stopping and aborting.")
                        stop_drive()
                        return
                    stop_drive()
                    print("  head marker not currently visible, holding...")
                else:
                    last_seen_head = now
                    cx, width_px, frame_w = head_result
                    if width_px >= TARGET_WIDTH_PX - TARGET_WIDTH_TOL_PX:
                        print(f"  target distance reached (width={width_px:.0f}px >= {TARGET_WIDTH_PX-TARGET_WIDTH_TOL_PX:.0f}px) -- stopping.")
                        stop_drive()
                        arrived = True
                        break
                    if width_px > PROXIMITY_WIDTH_PX:
                        print(f"  proximity backstop hit (marker width {width_px:.0f}px > {PROXIMITY_WIDTH_PX}) -- stopping.")
                        stop_drive()
                        arrived = True
                        break
                    err = (cx - frame_w / 2.0) / (frame_w / 2.0)  # [-1, 1]
                    w = float(np.clip(-STEER_GAIN * err, -MAX_ANGULAR, MAX_ANGULAR))
                    if w_drive.ready():
                        w_drive["twist"] = np.array([FORWARD_SPEED, w], dtype=np.float32)
                    print(f"  driving: marker cx={cx:.0f} w={width_px:.0f}px err={err:+.2f} -> v={FORWARD_SPEED} w={w:+.2f}")

                time.sleep(0.1)
        finally:
            stop_drive()

        if not arrived:
            print("did not arrive; not checking wrist markers.")
            return

        print(f"\n=== Step 1b: confirm wrist markers (up to {WRIST_CONFIRM_WINDOW_S:.0f}s) ===")
        found = {"left": False, "right": False}
        t_confirm = time.monotonic()
        while time.monotonic() - t_confirm < WRIST_CONFIRM_WINDOW_S:
            if not found["left"]:
                found["left"] = check_wrist_marker("left")
            if not found["right"]:
                found["right"] = check_wrist_marker("right")
            print(f"    wrist status: left={found['left']} right={found['right']}")
            if found["left"] and found["right"]:
                print("Both wrist markers confirmed.")
                break
            time.sleep(0.2)

        if not (found["left"] and found["right"]):
            print(f"\nNOT lifting -- wrist confirmation incomplete: {found}")
            return

        print(f"\n=== Step 2: raise shoulders {LIFT_CM:.0f}cm (J0 only, arm shape unchanged) ===")
        lift_starts = {s: get_motor_pos(ARM_NAME[s]) for s in ("left", "right")}
        lift_targets = {}
        lift_m = LIFT_CM / 100.0
        for side, cfg in cfgs.items():
            denom = float(cfg.ik_sign[0]) * cfg.wheel_radius * 2 * np.pi
            delta = EMPIRICAL_SIGN[side] * lift_m / denom
            t = lift_starts[side].copy()
            t[0] += delta
            lift_targets[side] = t.astype(np.float32)
            print(f"  {side}: J0 delta = {delta:.4f} turns")

        ramp_both(writers, lift_starts, lift_targets, 4.0)
        time.sleep(0.3)
        for side in ("left", "right"):
            final = get_motor_pos(ARM_NAME[side])
            commanded = lift_targets[side][0] - lift_starts[side][0]
            actual = final[0] - lift_starts[side][0]
            status = "OK" if abs(actual - commanded) < 0.05 else "MISMATCH"
            print(f"  {side}: commanded={commanded:.4f} actual={actual:.4f} -- {status}")
        print("\ndone.")


if __name__ == "__main__":
    main()
