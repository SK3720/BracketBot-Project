# /// script
# dependencies = ["bbos", "numpy<2", "opencv-python"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Full pickup motion, no driving/approach:
  1. Both arms move to the arm_config_2 preset pose (verified to actually
     arrive, not just commanded).
  2. Confirm both wrist cams see their marker (left=7, right=6) before
     proceeding -- retries for a few seconds, no arm movement during this.
  3. Only if both confirmed: raise both shoulders 30cm (J0 only, arm shape
     frozen the entire time).
"""
import time
import numpy as np
import cv2
from bbos import Reader, Writer, Type, Config

GRIPPER_IDX = 7
TARGET_LEFT = np.array([0.8867, 0.1421, 0.0004, 0.1114, 0.1022, 0.2333, -0.2094, -0.2429], dtype=np.float32)
TARGET_RIGHT = np.array([-0.8435, -0.1537, -0.0100, -0.0891, -0.0977, -0.2325, 0.1998, 0.2634], dtype=np.float32)
ARM_NAME = {"left": "arm_left", "right": "arm_right"}
MARKER_ID = {"left": 7, "right": 6}

LIFT_CM = 30.0  # retesting: preset height changed (arm_config_4), old 16cm ceiling was measured against arm_config_2
HOLD_AT_TOP_S = 3.0  # how long to hold at the top before lowering back down
EMPIRICAL_SIGN = {"left": 1.0, "right": -1.0}  # confirmed direction from earlier test
WRIST_CONFIRM_WINDOW_S = 8.0

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


def check_wrist_marker(side, save_tag=None):
    with Reader(f"camera.{side}.jpeg") as r:
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            if r.ready():
                n = int(r.data["jpeg_len"])
                buf = r.data["jpeg"][:n].copy()
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = DETECTOR.detectMarkers(gray)
                found = ids is not None and MARKER_ID[side] in ids.flatten().tolist()
                if save_tag is not None:
                    annotated = img.copy()
                    if ids is not None:
                        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
                    path = f"/tmp/live_{side}_{save_tag}.jpg"
                    cv2.imwrite(path, annotated)
                return found
            time.sleep(0.02)
    return False


def main():
    cfgs = {"left": Config("arm_left"), "right": Config("arm_right")}
    dof = cfgs["left"].dof

    with Writer("arm_left.torque", Type("arm_torque")) as wt_l, \
         Writer("arm_right.torque", Type("arm_torque")) as wt_r, \
         Writer("arm_left.ctrl", Type("arm_ctrl")) as wc_l, \
         Writer("arm_right.ctrl", Type("arm_ctrl")) as wc_r:

        wt_l["enable"] = np.ones(dof, dtype=np.bool_)
        wt_r["enable"] = np.ones(dof, dtype=np.bool_)
        writers = {"left": wc_l, "right": wc_r}

        print("=== Step 1: arms to arm_config_2 preset pose ===")
        starts = {s: get_motor_pos(ARM_NAME[s]) for s in ("left", "right")}
        targets = {"left": TARGET_LEFT, "right": TARGET_RIGHT}
        max_disp = max(np.abs(targets[s] - starts[s]).max() for s in ("left", "right"))
        print(f"  max joint displacement: {max_disp:.3f} turns")
        ramp_both(writers, starts, targets, 10.0)
        time.sleep(1.0)

        for side in ("left", "right"):
            final = get_motor_pos(ARM_NAME[side])
            err = np.abs(final - targets[side]).max()
            print(f"  {side}: max joint error vs preset = {err:.3f} turns")
            if err > 0.05:
                print(f"  WARNING: {side} arm did not reach preset within tolerance -- continuing anyway per instruction, but flagging this.")
        print("preset pose commanded.")

        print(f"\n=== Step 2: confirm wrist markers (up to {WRIST_CONFIRM_WINDOW_S:.0f}s) ===")
        found = {"left": False, "right": False}
        t_confirm = time.monotonic()
        SNAPSHOT_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        snapshots_taken = set()
        while time.monotonic() - t_confirm < WRIST_CONFIRM_WINDOW_S:
            elapsed = time.monotonic() - t_confirm
            due = [t for t in SNAPSHOT_TIMES if elapsed >= t and t not in snapshots_taken]
            tag = None
            if due:
                tag = f"{due[0]:.0f}s"
                snapshots_taken.add(due[0])
            # Always check+snapshot both sides when a snapshot is due, even if
            # already found, so we get a real time series for comparison.
            left_result = check_wrist_marker("left", save_tag=tag)
            right_result = check_wrist_marker("right", save_tag=tag)
            found["left"] = found["left"] or left_result
            found["right"] = found["right"] or right_result
            print(f"    wrist status: left={found['left']} right={found['right']}" + (f"  [snapshot {tag} saved: L={left_result} R={right_result}]" if tag else ""))
            if found["left"] and found["right"]:
                print("Both wrist markers confirmed.")
                break
            time.sleep(0.2)
        # Always leave a final-state snapshot too, success or failure.
        check_wrist_marker("left", save_tag="final")
        check_wrist_marker("right", save_tag="final")

        if not (found["left"] and found["right"]):
            print(f"\nNOT lifting -- wrist confirmation incomplete: {found}")
            return

        print(f"\n=== Step 3: raise shoulders {LIFT_CM:.0f}cm (J0 only, arm shape unchanged) ===")
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
        lifted_pos = {}
        for side in ("left", "right"):
            final = get_motor_pos(ARM_NAME[side])
            lifted_pos[side] = final
            commanded = lift_targets[side][0] - lift_starts[side][0]
            actual = final[0] - lift_starts[side][0]
            status = "OK" if abs(actual - commanded) < 0.05 else "MISMATCH"
            print(f"  {side}: commanded={commanded:.4f} actual={actual:.4f} -- {status}")

        print(f"\n=== Step 4: hold {HOLD_AT_TOP_S:.0f}s, then lower back to preset height ===")
        time.sleep(HOLD_AT_TOP_S)
        ramp_both(writers, lifted_pos, lift_starts, 4.0)
        time.sleep(0.3)
        for side in ("left", "right"):
            final = get_motor_pos(ARM_NAME[side])
            err = abs(final[0] - lift_starts[side][0])
            status = "OK" if err < 0.05 else "MISMATCH"
            print(f"  {side}: returned to J0={final[0]:.4f} (target={lift_starts[side][0]:.4f}) -- {status}")
        print("\ndone.")


if __name__ == "__main__":
    main()
