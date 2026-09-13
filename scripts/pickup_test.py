# /// script
# dependencies = [
#   "bbos",
#   "numpy<2",
#   "opencv-python",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Full two-arm pickup test:
  1. Both arms ramp simultaneously to their recorded preset ("under the tray") pose.
  2. Each wrist camera tries to detect its own marker (left=7, right=6).
  3. If not seen, blind micro-search: nudge that hand a few small amounts
     (+/-2cm in y and z) and recheck after each, until found or exhausted.
  4. Only if BOTH hands confirm their marker: lift both arms +15cm.

Preset joint values (0-6) come from the arm_config.json mimic recording;
gripper (index 7) is overridden with the separately-confirmed open value for
each arm, since the mimic demo's gripper position wasn't the calibrated open
target.
"""
import time
import numpy as np
import cv2
from bbos import Reader, Writer, Type, Config

GRIPPER_IDX = 7
LIFT_M = 0.05  # conservative first test; raise once direction/range is confirmed

TARGET_LEFT = np.array([0.5498, 0.0491, 0.0192, 0.2061, 0.0106, 0.2262, -0.2384, -0.1592], dtype=np.float32)
TARGET_RIGHT = np.array([-0.5769, -0.0714, 0.0311, -0.1765, -0.0830, -0.2426, 0.2457, 0.1592], dtype=np.float32)

MARKER_ID = {"left": 7, "right": 6}
ARM_NAME = {"left": "arm_left", "right": "arm_right"}

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(DICT, PARAMS)

# Search offsets in the hand's local (y, z) plane-ish -- reuses world-frame
# deltas via IK, same as move_arm_to_pose.py. Small and conservative.
SEARCH_DELTAS = [
    (0.0, 0.02), (0.0, -0.02),
    (0.02, 0.0), (-0.02, 0.0),
    (0.02, 0.02), (-0.02, -0.02),
]


def get_motor_pos(arm):
    with Reader(f"{arm}.state") as r:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if r.ready():
                return r.data["pos"].copy()
            time.sleep(0.01)
    raise RuntimeError(f"no state for {arm}")


def fk_pose(cfg, motor_pos):
    urdf8 = cfg.q2urdf(motor_pos.astype(np.float64).copy())
    result = cfg.ik.fk(urdf8[:7].tolist())
    return np.array(result[0], dtype=np.float64), np.array(result[1], dtype=np.float64)


def solve_iter(cfg, pos_xyz, quat_xyzw, max_calls=80, tol_mm=5.0):
    pos_xyz, quat_xyzw = list(pos_xyz), list(quat_xyzw)
    joints7, err_mm = None, float("inf")
    for _ in range(max_calls):
        sol = cfg.ik.solve(pos_xyz, quat_xyzw)
        if sol is None:
            continue
        joints7 = np.asarray(sol, dtype=np.float64)
        fk_pos = np.asarray(cfg.ik.fk(joints7.tolist())[0], dtype=np.float64)
        err_mm = 1000.0 * float(np.linalg.norm(fk_pos - np.asarray(pos_xyz)))
        if err_mm <= tol_mm:
            break
    return joints7, err_mm


def detect_marker(side, target_id):
    with Reader(f"camera.{side}.jpeg") as r:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if r.ready():
                n = int(r.data["jpeg_len"])
                buf = r.data["jpeg"][:n].copy()
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = DETECTOR.detectMarkers(gray)
                found = ids is not None and target_id in ids.flatten().tolist()
                annotated = img.copy()
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
                cv2.imwrite(f"/tmp/pickup_{side}_check.jpg", annotated)
                return found
            time.sleep(0.02)
    return False


def ramp_both(cfgs, starts, targets, duration, torque_writers, ctrl_writers):
    t0 = time.monotonic()
    while True:
        alpha = min((time.monotonic() - t0) / duration, 1.0)
        for side in ("left", "right"):
            if ctrl_writers[side].ready():
                cmd = starts[side] + alpha * (targets[side] - starts[side])
                ctrl_writers[side]["pos"] = cmd.astype(np.float32)
        if alpha >= 1.0:
            break
        time.sleep(0.01)


def main():
    cfgs = {"left": Config("arm_left"), "right": Config("arm_right")}
    for c in cfgs.values():
        c.ik.init()
    dof = cfgs["left"].dof

    print("=== Phase 1: move both arms to preset pose ===")
    starts = {s: get_motor_pos(ARM_NAME[s]) for s in ("left", "right")}
    targets = {"left": TARGET_LEFT, "right": TARGET_RIGHT}
    print("start left: ", np.round(starts["left"], 4))
    print("start right:", np.round(starts["right"], 4))

    with Writer("arm_left.torque", Type("arm_torque")) as wt_l, \
         Writer("arm_right.torque", Type("arm_torque")) as wt_r, \
         Writer("arm_left.ctrl", Type("arm_ctrl")) as wc_l, \
         Writer("arm_right.ctrl", Type("arm_ctrl")) as wc_r:

        wt_l["enable"] = np.ones(dof, dtype=np.bool_)
        wt_r["enable"] = np.ones(dof, dtype=np.bool_)
        torque_writers = {"left": wt_l, "right": wt_r}
        ctrl_writers = {"left": wc_l, "right": wc_r}

        ramp_both(cfgs, starts, targets, 4.0, torque_writers, ctrl_writers)
        time.sleep(0.5)
        print("preset pose reached.")

        print("\n=== Phase 2: detect markers ===")
        found = {}
        for side in ("left", "right"):
            found[side] = detect_marker(side, MARKER_ID[side])
            print(f"  {side} (id={MARKER_ID[side]}): {'FOUND' if found[side] else 'not found'}")

        def ramp_one(side, target_motor, duration):
            wc = ctrl_writers[side]
            one_start = get_motor_pos(ARM_NAME[side])
            t0 = time.monotonic()
            while True:
                a = min((time.monotonic() - t0) / duration, 1.0)
                if wc.ready():
                    wc["pos"] = (one_start + a * (target_motor - one_start)).astype(np.float32)
                if a >= 1.0:
                    break
                time.sleep(0.01)

        print("\n=== Phase 3: search for any not found ===")
        for side in ("left", "right"):
            if found[side]:
                continue
            print(f"  searching for {side} marker...")
            cur_motor = get_motor_pos(ARM_NAME[side])
            cur_pos, cur_quat = fk_pose(cfgs[side], cur_motor)
            for dy, dz in SEARCH_DELTAS:
                trial_pos = cur_pos + np.array([0.0, dy, dz])
                joints7, err = solve_iter(cfgs[side], trial_pos, cur_quat)
                if joints7 is None or err > 25.0:
                    print(f"    skip dy={dy} dz={dz}: IK error {err:.1f}mm")
                    continue
                urdf8 = np.zeros(8)
                urdf8[:7] = joints7
                motor8 = cfgs[side].urdf2q(urdf8).astype(np.float32)
                motor8[GRIPPER_IDX] = targets[side][GRIPPER_IDX]  # exact confirmed open value

                ramp_one(side, motor8, 1.0)
                time.sleep(0.3)
                if detect_marker(side, MARKER_ID[side]):
                    print(f"    FOUND at dy={dy} dz={dz}")
                    found[side] = True
                    break
                print(f"    not found at dy={dy} dz={dz}, reverting")
                ramp_one(side, cur_motor, 1.0)

        print(f"\nfinal detection status: {found}")
        if not (found.get("left") and found.get("right")):
            print("NOT lifting -- at least one hand never confirmed its marker.")
            return

        print(f"\n=== Phase 4: lift both arms +{LIFT_M*100:.0f}cm via J0 only (no other joint changes) ===")
        lift_starts = {s: get_motor_pos(ARM_NAME[s]) for s in ("left", "right")}
        lift_targets = {}
        for side in ("left", "right"):
            cfg = cfgs[side]
            # urdf_j0 is already in meters; motor_j0 = urdf_j0 / (ik_sign[0] * wheel_radius * 2*pi).
            # Solving for the delta directly avoids touching q2urdf/urdf2q (which would also
            # require valid values for joints 1-6, unnecessary since only J0 moves).
            denom = float(cfg.ik_sign[0]) * cfg.wheel_radius * 2 * np.pi
            motor_delta_j0 = LIFT_M / denom
            print(f"  {side}: J0 motor delta = {motor_delta_j0:.4f} turns (denom={denom:.4f})")
            target = lift_starts[side].copy()
            target[0] += motor_delta_j0
            lift_targets[side] = target.astype(np.float32)

        ramp_both(cfgs, lift_starts, lift_targets, 3.0, torque_writers, ctrl_writers)
        time.sleep(0.3)

        for side in ("left", "right"):
            final_motor = get_motor_pos(ARM_NAME[side])
            actual_delta_turns = final_motor[0] - lift_starts[side][0]
            commanded_delta_turns = lift_targets[side][0] - lift_starts[side][0]
            print(f"  {side}: commanded J0 delta={commanded_delta_turns:.4f} turns, "
                  f"actual={actual_delta_turns:.4f} turns "
                  f"({'OK' if abs(actual_delta_turns-commanded_delta_turns) < 0.05 else 'MISMATCH -- check for stall'})")
        print("lift complete.")


if __name__ == "__main__":
    main()
