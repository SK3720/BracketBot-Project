# /// script
# dependencies = ["bbos", "numpy<2", "opencv-python"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
import time
import cv2
from bbos import Reader

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())

with Reader("camera.head.jpeg") as r:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if r.ready():
            n = int(r.data["jpeg_len"])
            buf = r.data["jpeg"][:n].copy()
            stereo = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            half = stereo.shape[1] // 2
            gray = cv2.cvtColor(stereo[:, :half], cv2.COLOR_BGR2GRAY)
            corners, ids, _ = DETECTOR.detectMarkers(gray)
            print("frame size (w,h):", gray.shape[1], gray.shape[0])
            if ids is not None:
                for c, i in zip(corners, ids.flatten()):
                    w = c[0][:, 0].max() - c[0][:, 0].min()
                    h = c[0][:, 1].max() - c[0][:, 1].min()
                    print(f"  id={i} pixel size w={w:.1f} h={h:.1f}")
            else:
                print("  no markers found")
            break
        time.sleep(0.02)
