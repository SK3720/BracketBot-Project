# /// script
# dependencies = [
#   "bbos",
#   "numpy<2",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Does the arm hold position after we stop writing ctrl, or sag?
Reads current pos, then holds it by continuously re-publishing for HOLD_S
seconds while logging FK position every 0.5s, then stops writing and keeps
reading (no writing) for TAIL_S seconds to see what happens after disconnect.
"""
import time
import numpy as np
from bbos import Reader, Writer, Type, Config

ARM = "arm_left"
HOLD_S = 4.0
TAIL_S = 4.0

cfg = Config(ARM)
dof = cfg.dof


def get_pos(r):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if r.ready():
            return r.data["pos"].copy()
        time.sleep(0.01)
    raise RuntimeError("no state")


def fk(motor_pos):
    urdf8 = cfg.q2urdf(motor_pos.astype(np.float64).copy())
    return np.array(cfg.ik.fk(urdf8[:7].tolist())[0])


with Reader(f"{ARM}.state") as r:
    start = get_pos(r)
cfg.ik.init()
print(f"start EE pos: {np.round(fk(start), 4)}")

print(f"--- holding for {HOLD_S}s, writing continuously ---")
with Reader(f"{ARM}.state") as r, \
     Writer(f"{ARM}.torque", Type("arm_torque")) as w_t, \
     Writer(f"{ARM}.ctrl", Type("arm_ctrl")) as w_c:
    w_t["enable"] = np.ones(dof, dtype=np.bool_)
    t0 = time.monotonic()
    last_log = 0.0
    while time.monotonic() - t0 < HOLD_S:
        if w_c.ready():
            w_c["pos"] = start.astype(np.float32)
        if r.ready() and time.monotonic() - last_log > 0.5:
            last_log = time.monotonic()
            print(f"  t={time.monotonic()-t0:4.1f}s  EE pos: {np.round(fk(r.data['pos']), 4)}")
        time.sleep(0.005)

print(f"--- writers closed, watching for {TAIL_S}s (no more writes) ---")
with Reader(f"{ARM}.state") as r:
    t0 = time.monotonic()
    last_log = 0.0
    while time.monotonic() - t0 < TAIL_S:
        if r.ready() and time.monotonic() - last_log > 0.5:
            last_log = time.monotonic()
            print(f"  t={time.monotonic()-t0:4.1f}s  EE pos: {np.round(fk(r.data['pos']), 4)}")
        time.sleep(0.005)
