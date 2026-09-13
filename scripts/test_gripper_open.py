# /// script
# dependencies = [
#   "bbos",
#   "numpy<2",
#   "opencv-python",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Try both extremes of the gripper's URDF range (0 and 1 -- its <limit> is
exactly that range, firmware-enforced) and photograph the result each time so
we can see which one is the flat, 180-degree-open hand -- without touching any
other joint, and without needing anyone at the slider.
"""
import argparse
import time
import numpy as np
import cv2
from bbos import Reader, Writer, Type, Config

GRIPPER_IDX = 7


def get_motor_pos(arm):
    with Reader(f"{arm}.state") as r:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if r.ready():
                return r.data["pos"].copy()
            time.sleep(0.01)
    raise RuntimeError("no state")


def save_wrist_photo(side, tag):
    with Reader(f"camera.{side}.jpeg") as r:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if r.ready():
                n = int(r.data["jpeg_len"])
                buf = r.data["jpeg"][:n].copy()
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                path = f"/tmp/gripper_{side}_{tag}.jpg"
                cv2.imwrite(path, img)
                print(f"saved {path}")
                return
            time.sleep(0.02)
    print(f"WARNING: camera.{side}.jpeg not ready, no photo for {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["arm_left", "arm_right"], required=True)
    args = ap.parse_args()
    side = "left" if args.arm == "arm_left" else "right"

    cfg = Config(args.arm)
    dof = cfg.dof
    start_motor = get_motor_pos(args.arm)
    urdf8 = cfg.q2urdf(start_motor.astype(np.float64).copy())
    print(f"current gripper urdf value: {urdf8[GRIPPER_IDX]:.4f}")

    save_wrist_photo(side, "start")

    with Writer(f"{args.arm}.torque", Type("arm_torque")) as w_t, \
         Writer(f"{args.arm}.ctrl", Type("arm_ctrl")) as w_c:
        w_t["enable"] = np.ones(dof, dtype=np.bool_)

        for candidate in [0.0, 1.0]:
            target_urdf8 = urdf8.copy()
            target_urdf8[GRIPPER_IDX] = candidate
            target_motor = cfg.urdf2q(target_urdf8.copy()).astype(np.float32)

            print(f"\nmoving gripper to urdf={candidate} (motor={target_motor[GRIPPER_IDX]:.4f})...")
            t0 = time.monotonic()
            duration = 1.5
            cur = get_motor_pos(args.arm)
            while True:
                if w_c.ready():
                    alpha = min((time.monotonic() - t0) / duration, 1.0)
                    cmd = cur + alpha * (target_motor - cur)
                    w_c["pos"] = cmd.astype(np.float32)
                    if alpha >= 1.0:
                        break
                time.sleep(0.01)
            time.sleep(0.5)
            save_wrist_photo(side, f"urdf_{candidate}")

    print("\ndone. Leaving gripper at urdf=1.0 (last candidate tested).")


if __name__ == "__main__":
    main()
