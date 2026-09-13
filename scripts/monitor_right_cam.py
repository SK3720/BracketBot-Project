# /// script
# dependencies = ["bbos", "numpy<2"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Poll camera.right.status continuously to see if fps/streaming actually
drops out, and separately check camera.right.jpeg's own readiness/jpeg_len
each tick -- to tell a daemon-level freeze apart from a detection-side issue."""
import time
from bbos import Reader

DURATION_S = 25.0

with Reader("camera.right.status") as r_status, Reader("camera.right.jpeg") as r_jpeg:
    t0 = time.monotonic()
    last_len = None
    stuck_since = None
    while time.monotonic() - t0 < DURATION_S:
        t = time.monotonic() - t0
        status_line = ""
        if r_status.ready():
            d = r_status.data
            status_line = f"streaming={int(d['streaming'])} fps={float(d['fps']):.1f}"
        jpeg_line = "not ready"
        if r_jpeg.ready():
            n = int(r_jpeg.data["jpeg_len"])
            jpeg_line = f"jpeg_len={n}"
            if n == last_len:
                if stuck_since is None:
                    stuck_since = t
                stuck_line = f" [SAME LEN AS LAST TICK, stuck {t - stuck_since:.1f}s]"
            else:
                stuck_since = None
                stuck_line = ""
            last_len = n
        else:
            stuck_line = ""
        print(f"t={t:5.1f}s  status[{status_line}]  jpeg[{jpeg_line}]{stuck_line}")
        time.sleep(0.3)
