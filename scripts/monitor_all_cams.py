# /// script
# dependencies = ["bbos", "numpy<2"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Poll all three cameras' status simultaneously to see whether a stall on
one camera correlates with stalls on the others (shared/blocking daemon
loop -> threading would help) or is isolated to just that camera (per-camera
issue, e.g. a flaky physical connection -> threading wouldn't change it)."""
import time
from bbos import Reader

DURATION_S = 30.0

with Reader("camera.head.status") as r_head, \
     Reader("camera.left.status") as r_left, \
     Reader("camera.right.status") as r_right:
    t0 = time.monotonic()
    while time.monotonic() - t0 < DURATION_S:
        t = time.monotonic() - t0
        parts = []
        for name, r in (("head", r_head), ("left", r_left), ("right", r_right)):
            if r.ready():
                d = r.data
                parts.append(f"{name}: streaming={int(d['streaming'])} fps={float(d['fps']):.1f}")
            else:
                parts.append(f"{name}: (no update)")
        print(f"t={t:5.1f}s  " + "  |  ".join(parts))
        time.sleep(0.3)
