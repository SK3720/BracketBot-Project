# /// script
# dependencies = ["bbos", "numpy<2", "opencv-python"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Ramp directly to a known 8-value motor target (no IK) -- for replaying a
recorded/known-good pose exactly, rather than re-solving IK and risking a
different elbow configuration."""
import argparse
import json
import time
import numpy as np
import cv2
from bbos import Reader, Writer, Type, Config


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
                path = f"/tmp/playback_{side}_{tag}.jpg"
                cv2.imwrite(path, img)
                print(f"saved {path}")
                return
            time.sleep(0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["arm_left", "arm_right"], required=True)
    ap.add_argument("--motor", type=str, required=True)
    ap.add_argument("--duration", type=float, default=3.0)
    args = ap.parse_args()
    side = "left" if args.arm == "arm_left" else "right"

    target = np.array(json.loads(args.motor), dtype=np.float32)
    cfg = Config(args.arm)
    dof = cfg.dof
    assert len(target) == dof

    start = get_motor_pos(args.arm)
    print(f"start motor: {np.round(start,4)}")
    print(f"target motor: {np.round(target,4)}")

    with Writer(f"{args.arm}.torque", Type("arm_torque")) as w_t, \
         Writer(f"{args.arm}.ctrl", Type("arm_ctrl")) as w_c:
        w_t["enable"] = np.ones(dof, dtype=np.bool_)
        t0 = time.monotonic()
        while True:
            if w_c.ready():
                alpha = min((time.monotonic() - t0) / args.duration, 1.0)
                cmd = start + alpha * (target - start)
                w_c["pos"] = cmd.astype(np.float32)
                if alpha >= 1.0:
                    break
            time.sleep(0.01)

    time.sleep(0.3)
    final = get_motor_pos(args.arm)
    print(f"final motor: {np.round(final,4)}")
    print(f"final vs target error (motor units): {np.round(final-target,4)}")
    save_wrist_photo(side, "playback")


if __name__ == "__main__":
    main()
