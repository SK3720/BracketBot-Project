# /// script
# dependencies = [
#   "bbos",
#   "numpy<2",
#   "opencv-python",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Test tipping the hand from its current hanging-down orientation toward
horizontal (pitch about world Y) -- the pose we need to slide flat under a
tray. Solves IK iteratively (see move_arm_to_pose.py), refuses if error is
large, ramps slowly, and photographs the result from the head camera so we
can visually judge how close to horizontal it got.
"""
import argparse
import time
import numpy as np
import cv2
from bbos import Reader, Writer, Type, Config

GRIPPER_IDX = 7


def quat_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def quat_from_axis_angle(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle_rad / 2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle_rad/2.0)])


def get_motor_pos(arm):
    with Reader(f"{arm}.state") as r:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if r.ready():
                return r.data["pos"].copy()
            time.sleep(0.01)
    raise RuntimeError("no state")


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


def save_head_photo(tag):
    with Reader("camera.head.jpeg") as r:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if r.ready():
                n = int(r.data["jpeg_len"])
                buf = r.data["jpeg"][:n].copy()
                stereo = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                half = stereo.shape[1] // 2
                path = f"/tmp/orient_head_{tag}.jpg"
                cv2.imwrite(path, stereo[:, :half])
                print(f"saved {path}")
                return
            time.sleep(0.02)
    print("WARNING: head camera not ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["arm_left", "arm_right"], required=True)
    ap.add_argument("--pitch-deg", type=float, default=-90.0,
                     help="rotation about world Y to apply to current orientation")
    ap.add_argument("--gripper-urdf", type=float, default=1.0)
    args = ap.parse_args()

    cfg = Config(args.arm)
    dof = cfg.dof
    cfg.ik.init()

    start_motor = get_motor_pos(args.arm)
    cur_pos, cur_quat = fk_pose(cfg, start_motor)
    print(f"current EE pos={np.round(cur_pos,4)} quat={np.round(cur_quat,4)}")
    save_head_photo("before")

    delta_quat = quat_from_axis_angle([0, 1, 0], np.radians(args.pitch_deg))
    target_quat = quat_mult(delta_quat, cur_quat)
    target_quat = target_quat / np.linalg.norm(target_quat)
    print(f"target quat (pitch {args.pitch_deg} deg about world Y): {np.round(target_quat,4)}")

    joints7, err_mm = solve_iter(cfg, cur_pos, target_quat)
    if joints7 is None:
        print("IK never returned a solution. Aborting, no motion.")
        return
    print(f"IK solve error: {err_mm:.2f} mm")
    if err_mm > 30.0:
        print(f"REFUSING to move: error {err_mm:.1f}mm exceeds 30mm threshold.")
        return

    urdf8 = np.zeros(8, dtype=np.float64)
    urdf8[:7] = joints7[:7]
    urdf8[GRIPPER_IDX] = args.gripper_urdf
    target_motor = cfg.urdf2q(urdf8).astype(np.float32)
    print(f"target motor turns: {np.round(target_motor,4)}")

    with Writer(f"{args.arm}.torque", Type("arm_torque")) as w_t, \
         Writer(f"{args.arm}.ctrl", Type("arm_ctrl")) as w_c:
        w_t["enable"] = np.ones(dof, dtype=np.bool_)
        t0 = time.monotonic()
        duration = 4.0
        print(f"ramping over {duration}s...")
        while True:
            if w_c.ready():
                alpha = min((time.monotonic() - t0) / duration, 1.0)
                cmd = start_motor + alpha * (target_motor - start_motor)
                w_c["pos"] = cmd.astype(np.float32)
                if alpha >= 1.0:
                    break
            time.sleep(0.01)

    time.sleep(0.5)
    final_motor = get_motor_pos(args.arm)
    final_pos, final_quat = fk_pose(cfg, final_motor)
    print(f"final EE pos={np.round(final_pos,4)} quat={np.round(final_quat,4)}")
    save_head_photo("after")


if __name__ == "__main__":
    main()
