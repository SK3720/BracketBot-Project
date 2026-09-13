# /// script
# dependencies = ["bbos", "numpy<2", "opencv-python"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
import sys
import time
import cv2
from bbos import Reader

side = sys.argv[1] if len(sys.argv) > 1 else "left"
with Reader(f"camera.{side}.jpeg") as r:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if r.ready():
            n = int(r.data["jpeg_len"])
            buf = r.data["jpeg"][:n].copy()
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            cv2.imwrite(f"/tmp/snap_{side}.jpg", img)
            print("saved")
            break
        time.sleep(0.02)
