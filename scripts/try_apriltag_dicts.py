# /// script
# dependencies = ["bbos", "numpy<2", "opencv-python"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
import time
import cv2
from bbos import Reader

CANDIDATES = [
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
    "DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_100", "DICT_6X6_250",
]

with Reader("camera.head.jpeg") as r:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if r.ready():
            n = int(r.data["jpeg_len"])
            buf = r.data["jpeg"][:n].copy()
            stereo = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            half = stereo.shape[1] // 2
            gray = cv2.cvtColor(stereo[:, :half], cv2.COLOR_BGR2GRAY)
            for name in CANDIDATES:
                dict_id = getattr(cv2.aruco, name)
                d = cv2.aruco.getPredefinedDictionary(dict_id)
                params = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(d, params)
                corners, ids, _ = detector.detectMarkers(gray)
                found = ids.flatten().tolist() if ids is not None else []
                print(f"{name}: {found}")
            break
        time.sleep(0.02)
