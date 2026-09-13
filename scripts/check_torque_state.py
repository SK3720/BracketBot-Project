# /// script
# dependencies = ["bbos", "numpy<2"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
import time
from bbos import Reader

for arm in ("arm_left", "arm_right"):
    with Reader(f"{arm}.torque") as r:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if r.ready():
                d = r.data
                print(f"{arm}.torque: enable={d['enable']} tau_mode={d['tau_mode']} "
                      f"compliance_mode={d['compliance_mode']} axis_aligned={d['axis_aligned']} "
                      f"force_only={d['force_only']} j0_homing={d['j0_homing']} calibrating={d['calibrating']}")
                break
            time.sleep(0.02)
        else:
            print(f"{arm}.torque: NOT READY")
