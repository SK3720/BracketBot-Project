# /// script
# dependencies = ["bbos", "numpy<2", "opencv-python"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
import time
import cv2
import numpy as np
from bbos import Reader

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(DICT, PARAMS)

with Reader("camera.head.jpeg") as r:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if r.ready():
            n = int(r.data["jpeg_len"])
            buf = r.data["jpeg"][:n].copy()
            stereo = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            half = stereo.shape[1] // 2
            left_eye = stereo[:, :half]
            gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = DETECTOR.detectMarkers(gray)
            print(f"detected ids: {ids.flatten().tolist() if ids is not None else []}")
            print(f"num rejected candidates: {len(rejected)}")
            annotated = left_eye.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
                for c, i in zip(corners, ids.flatten()):
                    center = c[0].mean(axis=0)
                    print(f"  id={i} pixel_center={np.round(center,1)}")
            cv2.imwrite("/tmp/detect_head.jpg", annotated)
            print("saved /tmp/detect_head.jpg")
            break
        time.sleep(0.02)
    else:
        print("camera.head.jpeg NOT READY")
