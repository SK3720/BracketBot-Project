# /// script
# dependencies = [
#   "bbos",
#   "numpy<2",
#   "opencv-python",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Sanity check: can we read frames off each camera and run ArUco detection?
No markers are printed/placed yet -- this just validates the pipeline (cv2
available, aruco submodule present, frames decode, detector runs without
error) and saves one frame per camera to disk so we can eyeball framing/focus.
"""
import time
import cv2
import numpy as np
from bbos import Reader

print(f"cv2 version: {cv2.__version__}")
print(f"has aruco: {hasattr(cv2, 'aruco')}")

if hasattr(cv2, "aruco"):
    DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        PARAMS = cv2.aruco.DetectorParameters()
        DETECTOR = cv2.aruco.ArucoDetector(DICT, PARAMS)

        def detect(gray):
            corners, ids, _ = DETECTOR.detectMarkers(gray)
            return corners, ids
    else:
        PARAMS = cv2.aruco.DetectorParameters_create()

        def detect(gray):
            corners, ids, _ = cv2.aruco.detectMarkers(gray, DICT, parameters=PARAMS)
            return corners, ids


def wait_ready(r, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if r.ready():
            return True
        time.sleep(0.01)
    return False


def try_head():
    with Reader("camera.head.jpeg") as r:
        if not wait_ready(r, timeout=10.0):
            print("camera.head.jpeg: NOT READY")
            return
        n = int(r.data["jpeg_len"])
        buf = r.data["jpeg"][:n].copy()
        stereo = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR already
        if stereo is None:
            print(f"camera.head.jpeg: got {n} bytes but failed to decode")
            return
        half = stereo.shape[1] // 2
        left_eye = stereo[:, :half]
        print(f"camera.head.jpeg: got frame {stereo.shape}, left eye {left_eye.shape}")
        cv2.imwrite("/tmp/head_left.jpg", left_eye)
        if hasattr(cv2, "aruco"):
            gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
            corners, ids = detect(gray)
            print(f"camera.head.jpeg: detected {0 if ids is None else len(ids)} markers")


def try_wrist(side):
    name = f"camera.{side}.jpeg"
    with Reader(name) as r:
        if not wait_ready(r):
            print(f"{name}: NOT READY")
            return
        n = int(r.data["jpeg_len"])
        buf = r.data["jpeg"][:n].copy()
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            print(f"{name}: got {n} bytes but failed to decode")
            return
        print(f"{name}: got frame {img.shape} ({n} bytes jpeg)")
        cv2.imwrite(f"/tmp/wrist_{side}.jpg", img)
        if hasattr(cv2, "aruco"):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids = detect(gray)
            print(f"{name}: detected {0 if ids is None else len(ids)} markers")


try_head()
try_wrist("left")
try_wrist("right")
print("done. frames saved to /tmp/head_left.jpg, /tmp/wrist_left.jpg, /tmp/wrist_right.jpg")
