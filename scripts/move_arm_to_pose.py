# /// script
# dependencies = [
#   "bbos",
#   "numpy<2",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""
First hardware test: command one arm to a cartesian target via the onboard IK
solver, ramped smoothly from its current pose (same interp pattern as
leader_follower_teleop.py). No camera, no tray -- just answering "can this arm
reach a commanded pose reliably?" before anything else gets built on top of it.

Target is specified as a DELTA from the arm's current end-effector pose (not
an absolute xyz) so a first run can never send the arm somewhere wild -- you're
always nudging from wherever it already is.

Usage:
  uv run move_arm_to_pose.py --arm arm_left --dry-run
  uv run move_arm_to_pose.py --arm arm_left --dx 0.05
  uv run move_arm_to_pose.py --arm arm_left --dx 0.05 --dz -0.03 --duration 3
"""
import argparse
import time
import numpy as np
from bbos import Reader, Writer, Type, Config

GRIPPER_IDX = 7
FALLBACK_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # identity, xyzw


def get_current_motor_pos(arm_name):
    with Reader(f"{arm_name}.state") as r:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if r.ready():
                return r.data["pos"].copy()
            time.sleep(0.01)
        raise RuntimeError(f"no {arm_name}.state after 5s -- is the daemon running?")


def fk_pose(cfg, motor_pos):
    """Current end-effector (pos, quat) in the IK solver's world frame."""
    urdf8 = cfg.q2urdf(motor_pos.astype(np.float64).copy())
    joints7 = urdf8[:7].tolist()
    result = cfg.ik.fk(joints7)
    pos = np.array(result[0], dtype=np.float64)
    quat = np.array(result[1], dtype=np.float64) if len(result) > 1 else FALLBACK_QUAT.copy()
    return pos, quat


def solve_motor_target(cfg, pos_xyz, quat_xyzw, gripper_motor_turns,
                        max_calls=80, tol_mm=5.0):
    """ik.solve() is a warm-started QP that only takes a handful of internal
    steps per call (see constants.py rik_max_iterations) -- it's built to be
    called every tick of a control loop, not once. Call it repeatedly with the
    same target so it can walk from the cold-start config to convergence,
    exactly like view_quest.py does every frame."""
    pos_xyz = list(pos_xyz)
    quat_xyzw = list(quat_xyzw)
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
    if joints7 is None:
        raise RuntimeError(f"ik.solve() never returned a solution for target {pos_xyz}")
    urdf8 = np.zeros(8, dtype=np.float64)
    urdf8[:7] = joints7[:7]
    motor8 = cfg.urdf2q(urdf8)
    motor8[GRIPPER_IDX] = gripper_motor_turns  # gripper is not IK-driven, set directly
    return motor8.astype(np.float32), joints7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["arm_left", "arm_right"], required=True)
    ap.add_argument("--dx", type=float, default=0.0, help="meters, +x = forward")
    ap.add_argument("--dy", type=float, default=0.0, help="meters, +y = left")
    ap.add_argument("--dz", type=float, default=0.0, help="meters, +z = up")
    ap.add_argument("--gripper", type=float, default=None,
                     help="motor turns for gripper slot; default = keep current")
    ap.add_argument("--duration", type=float, default=2.5, help="ramp time, seconds")
    ap.add_argument("--dry-run", action="store_true",
                     help="print the computed target and FK error, touch no hardware")
    args = ap.parse_args()

    cfg = Config(args.arm)
    dof = cfg.dof

    start_motor = get_current_motor_pos(args.arm)
    cfg.ik.init()
    cur_pos, cur_quat = fk_pose(cfg, start_motor)
    print(f"[{args.arm}] current EE pose: pos={np.round(cur_pos, 4)} quat={np.round(cur_quat, 4)}")

    target_pos = cur_pos + np.array([args.dx, args.dy, args.dz], dtype=np.float64)
    target_quat = cur_quat  # keep current orientation for this first test
    gripper_turns = args.gripper if args.gripper is not None else start_motor[GRIPPER_IDX]

    target_motor, target_joints7 = solve_motor_target(cfg, target_pos, target_quat, gripper_turns)
    achieved_pos, _ = fk_pose(cfg, target_motor)
    err_mm = 1000.0 * float(np.linalg.norm(achieved_pos - target_pos))

    print(f"[{args.arm}] target EE pose:  pos={np.round(target_pos, 4)}")
    print(f"[{args.arm}] IK joints (URDF space, j0..j6): {np.round(target_joints7, 4)}")
    print(f"[{args.arm}] target motor turns (incl. gripper): {np.round(target_motor, 4)}")
    print(f"[{args.arm}] IK solve error vs target: {err_mm:.2f} mm")

    if args.dry_run:
        print("[dry-run] not touching hardware.")
        return

    if err_mm > 25.0:
        print(f"[{args.arm}] REFUSING to move: IK error {err_mm:.1f}mm exceeds 25mm safety threshold.")
        return

    with Writer(f"{args.arm}.torque", Type("arm_torque")) as w_torque, \
         Writer(f"{args.arm}.ctrl", Type("arm_ctrl")) as w_ctrl:

        w_torque["enable"] = np.ones(dof, dtype=np.bool_)

        t0 = time.monotonic()
        print(f"[{args.arm}] ramping over {args.duration:.1f}s...")
        while True:
            if w_ctrl.ready():
                alpha = min((time.monotonic() - t0) / args.duration, 1.0)
                cmd = start_motor + alpha * (target_motor - start_motor)
                w_ctrl["pos"] = cmd.astype(np.float32)
                if alpha >= 1.0:
                    break
            time.sleep(0.01)

    time.sleep(0.2)
    final_motor = get_current_motor_pos(args.arm)
    final_pos, _ = fk_pose(cfg, final_motor)
    reach_err_mm = 1000.0 * float(np.linalg.norm(final_pos - target_pos))
    print(f"[{args.arm}] done. final EE pos={np.round(final_pos, 4)}, error vs target: {reach_err_mm:.2f} mm")


if __name__ == "__main__":
    main()
