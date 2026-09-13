# /// script
# dependencies = ["bbos", "numpy<2"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Two steps, nothing else: (1) both arms play back their recorded mimic
preset pose, (2) both shoulders (J0 only, all other joints frozen) raise
together by --lift-cm. No marker detection, no search, no gating."""
import argparse
import time
import numpy as np
from bbos import Reader, Writer, Type, Config

GRIPPER_IDX = 7
TARGET_LEFT = np.array([0.5498, 0.0491, 0.0192, 0.2061, 0.0106, 0.2262, -0.2384, -0.2620], dtype=np.float32)
TARGET_RIGHT = np.array([-0.5769, -0.0714, 0.0311, -0.1765, -0.0830, -0.2426, 0.2457, 0.2515], dtype=np.float32)
ARM_NAME = {"left": "arm_left", "right": "arm_right"}


def get_motor_pos(arm):
    with Reader(f"{arm}.state") as r:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if r.ready():
                return r.data["pos"].copy()
            time.sleep(0.01)
    raise RuntimeError(f"no state for {arm}")


def ramp_both(writers, starts, targets, duration):
    t0 = time.monotonic()
    while True:
        a = min((time.monotonic() - t0) / duration, 1.0)
        for side in ("left", "right"):
            if writers[side].ready():
                cmd = starts[side] + a * (targets[side] - starts[side])
                writers[side]["pos"] = cmd.astype(np.float32)
        if a >= 1.0:
            break
        time.sleep(0.01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lift-cm", type=float, default=15.0)
    args = ap.parse_args()
    lift_m = args.lift_cm / 100.0

    cfgs = {"left": Config("arm_left"), "right": Config("arm_right")}
    dof = cfgs["left"].dof

    with Writer("arm_left.torque", Type("arm_torque")) as wt_l, \
         Writer("arm_right.torque", Type("arm_torque")) as wt_r, \
         Writer("arm_left.ctrl", Type("arm_ctrl")) as wc_l, \
         Writer("arm_right.ctrl", Type("arm_ctrl")) as wc_r:

        wt_l["enable"] = np.ones(dof, dtype=np.bool_)
        wt_r["enable"] = np.ones(dof, dtype=np.bool_)
        writers = {"left": wc_l, "right": wc_r}

        print("=== Step 1: mimic preset pose ===")
        starts = {s: get_motor_pos(ARM_NAME[s]) for s in ("left", "right")}
        targets = {"left": TARGET_LEFT, "right": TARGET_RIGHT}
        ramp_both(writers, starts, targets, 4.0)
        time.sleep(0.3)
        print("preset pose reached.")

        # Empirically determined: despite identical ik_sign[0]/wheel_radius in
        # config for both arms, the real hardware responds with OPPOSITE
        # physical direction to the same commanded sign on J0 (observed:
        # left rose correctly, right went down, for the same computed delta).
        # Likely a real wiring/encoder-direction asymmetry between the two
        # mast motors, not something the software constants capture.
        EMPIRICAL_SIGN = {"left": 1.0, "right": -1.0}

        print(f"\n=== Step 2: raise shoulders {args.lift_cm:.0f}cm (J0 only) ===")
        lift_starts = {s: get_motor_pos(ARM_NAME[s]) for s in ("left", "right")}
        lift_targets = {}
        for side, cfg in cfgs.items():
            denom = float(cfg.ik_sign[0]) * cfg.wheel_radius * 2 * np.pi
            delta = EMPIRICAL_SIGN[side] * lift_m / denom
            t = lift_starts[side].copy()
            t[0] += delta
            lift_targets[side] = t.astype(np.float32)
            print(f"  {side}: J0 delta = {delta:.4f} turns")

        ramp_both(writers, lift_starts, lift_targets, 3.0)
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
