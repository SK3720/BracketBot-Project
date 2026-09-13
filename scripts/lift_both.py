# /// script
# dependencies = ["bbos", "numpy<2"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Just raise both arms together by lifting J0 (shoulder/mast) only -- no
marker detection, no search, no other joint changes. Both arms move by the
same real-world delta, simultaneously, in lockstep."""
import argparse
import time
import numpy as np
from bbos import Reader, Writer, Type, Config


def get_motor_pos(arm):
    with Reader(f"{arm}.state") as r:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if r.ready():
                return r.data["pos"].copy()
            time.sleep(0.01)
    raise RuntimeError(f"no state for {arm}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lift-cm", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=3.0)
    args = ap.parse_args()
    lift_m = args.lift_cm / 100.0

    cfgs = {"left": Config("arm_left"), "right": Config("arm_right")}
    dof = cfgs["left"].dof
    arm_name = {"left": "arm_left", "right": "arm_right"}

    starts = {s: get_motor_pos(arm_name[s]) for s in ("left", "right")}
    targets = {}
    for side, cfg in cfgs.items():
        denom = float(cfg.ik_sign[0]) * cfg.wheel_radius * 2 * np.pi
        delta = lift_m / denom
        t = starts[side].copy()
        t[0] += delta
        targets[side] = t.astype(np.float32)
        print(f"{side}: J0 {starts[side][0]:.4f} -> {t[0]:.4f} (delta {delta:.4f} turns) for {lift_m*100:.0f}cm")

    with Writer("arm_left.torque", Type("arm_torque")) as wt_l, \
         Writer("arm_right.torque", Type("arm_torque")) as wt_r, \
         Writer("arm_left.ctrl", Type("arm_ctrl")) as wc_l, \
         Writer("arm_right.ctrl", Type("arm_ctrl")) as wc_r:

        wt_l["enable"] = np.ones(dof, dtype=np.bool_)
        wt_r["enable"] = np.ones(dof, dtype=np.bool_)
        writers = {"left": wc_l, "right": wc_r}

        t0 = time.monotonic()
        print(f"lifting {args.lift_cm:.0f}cm over {args.duration:.1f}s...")
        while True:
            a = min((time.monotonic() - t0) / args.duration, 1.0)
            for side in ("left", "right"):
                if writers[side].ready():
                    cmd = starts[side] + a * (targets[side] - starts[side])
                    writers[side]["pos"] = cmd.astype(np.float32)
            if a >= 1.0:
                break
            time.sleep(0.01)

    time.sleep(0.3)
    for side in ("left", "right"):
        final = get_motor_pos(arm_name[side])
        commanded = targets[side][0] - starts[side][0]
        actual = final[0] - starts[side][0]
        status = "OK" if abs(actual - commanded) < 0.05 else "MISMATCH (possible stall/hardstop)"
        print(f"{side}: commanded J0 delta={commanded:.4f}, actual={actual:.4f} -- {status}")


if __name__ == "__main__":
    main()
