# /// script
# dependencies = ["bbos", "numpy<2"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
import time
from bbos import Reader

for name in ["camera.head.status", "camera.left.status", "camera.right.status", "camera.head.jpeg"]:
    try:
        with Reader(name) as r:
            deadline = time.monotonic() + 3.0
            ready = False
            while time.monotonic() < deadline:
                if r.ready():
                    ready = True
                    break
                time.sleep(0.02)
            if ready:
                d = r.data
                print(name, "READY", {k: (d[k] if d[k].shape == () else d[k].shape) for k in d.dtype.names} if hasattr(d, "dtype") else dict(d))
            else:
                print(name, "NOT READY")
    except Exception as e:
        print(name, "ERROR", e)
