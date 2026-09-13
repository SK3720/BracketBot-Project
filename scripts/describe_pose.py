# /// script
# dependencies = ["bbos", "numpy<2"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Given a raw 8-value motor position array, print the URDF-space joints and
the cartesian (pos, quat) end-effector pose via FK. Used to turn a mimic
recording's held frame into an actual target pose we can hardcode."""
import argparse
import json
import numpy as np
from bbos import Config

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["arm_left", "arm_right"], required=True)
ap.add_argument("--motor", type=str, required=True, help="JSON list of 8 motor values")
args = ap.parse_args()

motor = np.array(json.loads(args.motor), dtype=np.float64)
cfg = Config(args.arm)
cfg.ik.init()

urdf8 = cfg.q2urdf(motor.copy())
print(f"motor pos:  {np.round(motor, 4)}")
print(f"urdf pos:   {np.round(urdf8, 4)}  (j0..j6 + gripper)")

pos, quat = cfg.ik.fk(urdf8[:7].tolist())
pos = np.array(pos, dtype=np.float64)
quat = np.array(quat, dtype=np.float64)
print(f"EE position (m): {np.round(pos, 4)}")
print(f"EE quat (xyzw):   {np.round(quat, 4)}")

# round-trip sanity check: does solving IK for this pose reproduce these joints?
sol = None
for _ in range(80):
    sol = cfg.ik.solve(pos.tolist(), quat.tolist())
    if sol is None:
        continue
    fk_pos = np.array(cfg.ik.fk(list(sol))[0])
    err = 1000.0 * np.linalg.norm(fk_pos - pos)
    if err <= 5.0:
        break
if sol is not None:
    err = 1000.0 * np.linalg.norm(np.array(cfg.ik.fk(list(sol))[0]) - pos)
    print(f"round-trip IK solve error: {err:.2f} mm")
    print(f"round-trip joints: {np.round(sol, 4)}")
else:
    print("round-trip IK solve FAILED")
