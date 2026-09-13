# /// script
# dependencies = [
#   "bbos",
#   "fastapi",
#   "uvicorn",
#   "opencv-python",
#   "numpy<2",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Live web viewer for all three cameras with an ArUco overlay -- same
FastAPI/uvicorn MJPEG-streaming structure as view_camera.py, just with
marker detection + a simple orientation line drawn on each frame before
it's re-encoded. Read-only (bbos Readers only), so it's safe to run
alongside full_delivery.py or any other script during a live test.

Detects on the raw (distorted) frame, no undistortion or camera-intrinsics
pose estimation -- this is a debug viewer, not the pipeline's own detector,
so "orientation" here is just the in-plane rotation (a line from each
marker's center to its top-edge midpoint), not a full 3D pose.
"""
import asyncio
import contextlib
import socket
import threading
from queue import Queue, Empty

import cv2
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn
from bbos import Reader

CAMS = ["head", "left", "right"]
FPS = 15  # lower than view_camera.py's 30 -- detection+redraw is real CPU work, and this runs alongside a live test
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# Same loosened tolerance as full_delivery.py's DETECTOR, so this overlay
# shows what the real pipeline will actually detect, not a stricter default.
_ARUCO_PARAMS = cv2.aruco.DetectorParameters()
_ARUCO_PARAMS.polygonalApproxAccuracyRate = 0.045
_ARUCO_PARAMS.minMarkerPerimeterRate = 0.025
_ARUCO_PARAMS.errorCorrectionRate = 0.7
_ARUCO_PARAMS.maxErroneousBitsInBorderRate = 0.425
_ARUCO_PARAMS.perspectiveRemoveIgnoredMarginPerCell = 0.165
_ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
DETECTOR = cv2.aruco.ArucoDetector(DICT, _ARUCO_PARAMS)

queues = {cam: Queue(maxsize=2) for cam in CAMS}


def overlay_aruco(jpeg_bytes):
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jpeg_bytes
    corners, ids, _ = DETECTOR.detectMarkers(frame)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for c in corners:
            pts = c.reshape(4, 2)
            center = pts.mean(axis=0)
            top_mid = (pts[0] + pts[1]) / 2.0  # corner order is TL,TR,BR,BL -- TL->TR is the "top" edge
            cv2.arrowedLine(frame, tuple(center.astype(int)), tuple(top_mid.astype(int)),
                             (0, 0, 255), 2, tipLength=0.3)
    ok, out = cv2.imencode(".jpg", frame)
    return out.tobytes() if ok else jpeg_bytes


# One loop for every Reader: Loop's global pacing state is not thread-safe.
def camera_reader():
    with contextlib.ExitStack() as stack:
        readers = {c: stack.enter_context(Reader(f"camera.{c}.jpeg")) for c in CAMS}
        while True:
            for cam, r in readers.items():
                if not r.ready():
                    continue
                jpeg_bytes = bytes(r.data['jpeg'][:r.data['jpeg_len']])
                jpeg_bytes = overlay_aruco(jpeg_bytes)
                q = queues[cam]
                try:
                    q.put_nowait(jpeg_bytes)
                except Exception:
                    try:
                        q.get_nowait()
                        q.put_nowait(jpeg_bytes)
                    except Exception:
                        pass


app = FastAPI()


def make_stream(cam):
    q = queues[cam]

    async def generate():
        while True:
            try:
                frame = q.get_nowait()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Empty:
                pass
            await asyncio.sleep(1 / FPS)
    return generate


for _cam in CAMS:
    _gen = make_stream(_cam)

    def _make_routes(cam, gen):
        @app.get(f"/{cam}/stream")
        async def stream(g=gen):
            return StreamingResponse(
                g(),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

    _make_routes(_cam, _gen)


@app.get("/")
async def index():
    imgs = "\n".join(
        f'<div><h2>{cam}</h2><img src="/{cam}/stream" /></div>'
        for cam in CAMS
    )
    html = f'''
    <html>
    <head>
        <title>ArUco Camera Streams</title>
        <style>
            body {{ margin: 0; padding: 20px; background: #000; color: #fff; font-family: sans-serif; }}
            div {{ margin-bottom: 20px; }}
            h2 {{ margin: 5px 0; }}
            img {{ max-width: 100%; height: auto; display: block; }}
        </style>
    </head>
    <body>{imgs}</body>
    </html>
    '''
    return Response(content=html, media_type="text/html")


def main():
    threading.Thread(target=camera_reader, daemon=True).start()

    host = socket.gethostname()
    print(f"[+] Streaming ArUco-overlaid cameras on http://{host}.local:8005/")

    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="error",
                access_log=False, timeout_graceful_shutdown=1)


if __name__ == "__main__":
    main()
