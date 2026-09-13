# /// script
# dependencies = [
#   "bbos",
#   "numpy<2",
#   "opencv-python",
#   "fastapi",
#   "uvicorn",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""
Drive the robot up to a specific ArUco tag and stop --target-distance-mm away.

This is a reactive visual servo, not a planner: there is no map, no path, no
lookahead, and no obstacle awareness. Every camera frame independently
produces one bearing + one distance number from the current PnP solve, and a
proportional controller turns those two numbers directly into a twist
command. There is no "planned route" at any point -- the only state carried
between frames is the temporal-confirmation tracker (is this detection real)
and, while driving, nothing else. Stopping is purely "the tag's own solvePnP
distance is now <= target" -- no separate depth sensing is used or expected;
this robot doesn't run a depth daemon, and camera.depth is not read here.

Perception is the same core as view_pnp.py (fisheye-undistort the head
camera's chosen eye, detect with cv2.aruco, solvePnP, temporal-confirmation
tracking) but tuned stricter (more confirm-hits, no cranked error-correction)
since a false positive now steers a moving robot instead of just lighting up
a web page.

Control: per confirmed-detection frame, bearing = atan2(tvec_x, tvec_z) and
distance = |tvec| come straight out of solvePnP in the camera's own frame --
no camera->base extrinsics needed, because the head camera has no pan/yaw
actuator anywhere in this codebase (checked), so camera-forward IS
robot-forward by construction. Within --bearing-deadband-deg of dead-center
the controller stops correcting heading entirely (that's the "roughly
centered" band) and just drives straight; outside it, angular velocity is
proportional to bearing error and forward speed is cut to 0 past a wide
bearing error (turn to face it before driving at it). The bearing->w sign
was set from direct observation driving the real robot (the theoretical
derivation from teleop.py's key bindings called for the opposite sign and
was wrong in practice -- some combination of camera/eye orientation this
script doesn't otherwise account for), not re-derived here: positive w =
positive bearing (tag right -> w commands right on this robot as wired up).
If a future change to --side or the camera flips this again, re-derive it
empirically (dry-run, watch which way it leans) rather than from first
principles -- that's what got it wrong the first time.

Detections flicker: a real, present tag can fail cv2.aruco's decode for a
single frame (motion blur, an unlucky threshold pass) and reappear the next.
Treating every such miss as "lost" reads as the tag disappearing every other
frame. MEMORY_S (0.5s) coasts on the last confirmed pose across a gap that
short before actually falling back to search.

State, per frame, no explicit state machine needed:
  target id has a CONFIRMED track this frame, or one within MEMORY_S  ->
                                                 approach it (or stop, if
                                                 already within range)
  otherwise                                    -> rotate slowly to search

Safety:
  - drive.ctrl's own daemon-side watchdog stops the robot if commands stop
    arriving for 100ms; this loop still writes an explicit zero twist on
    every exit path (Ctrl-C, exception, arrival) instead of relying on that
    alone.
  - --dry-run computes and prints everything (detection, control state) but
    never opens the drive Writer, so the full pipeline can be exercised on
    a live robot without it ever moving. Use this first, especially after
    changing any control gain.
  - Nothing here stops the robot from driving into an obstacle between it
    and the tag -- there is no obstacle sensing at all. That's a deliberate
    scope cut (this robot doesn't run a depth daemon), not an oversight; if
    that changes, it'd be a separate addition, not something to assume from
    this script's current behavior.

Run:    uv run examples/goto_tag.py --side left --dict 4X4_50 --marker-id 5
Open:   http://<robot>.local:8013/   (live annotated feed + control state)

Requires a calibration file from calibrate_fisheye.py for the SAME --side
(default /home/bracketbot/fisheye_calib_<side>.json).
"""
import argparse
import asyncio
import json
import signal
import socket
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn
from bbos import Reader, Writer, Type, Config

FPS = 20

ARUCO_DICTS = {
    "4X4_50": cv2.aruco.DICT_4X4_50, "4X4_100": cv2.aruco.DICT_4X4_100,
    "4X4_250": cv2.aruco.DICT_4X4_250, "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100, "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100, "ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}
PNP_FLAG = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE

TRACK_MATCH_PX = 60
TRACK_MAX_AGE_S = 1.0
MEMORY_S = 0.5  # coast on the last confirmed pose this long before treating the tag as lost

app = FastAPI()
STATE = {}
LOCK = threading.Lock()
FRAME_Q = Queue(maxsize=2)
LATEST = {"mode": "starting", "detail": {}}
STOP = False


def _sigint(*_):
    global STOP
    STOP = True


signal.signal(signal.SIGINT, _sigint)
signal.signal(signal.SIGTERM, _sigint)


def crop_eye(stereo, side):
    half = stereo.shape[1] // 2
    return stereo[:, :half] if side == "left" else stereo[:, half:]


def build_detector(dict_name, min_perimeter_rate, error_correction_rate):
    """Same knobs as view_pnp.py, but this script's own defaults are more
    conservative -- see argparse help. Navigation only needs to pick the
    target tag up somewhat farther than working range, not at the extreme
    edge of detectability; confirm-hits (not detector permissiveness) is
    the primary defense here since the cost of a false positive is now a
    moving robot, not a mislabeled web page."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])

    def configure(params):
        if hasattr(params, "cornerRefinementMethod"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.minMarkerPerimeterRate = min_perimeter_rate
        params.errorCorrectionRate = error_correction_rate
        return params

    if hasattr(cv2.aruco, "ArucoDetector"):
        params = configure(cv2.aruco.DetectorParameters())
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return lambda gray: detector.detectMarkers(gray)[:2]
    else:
        params = configure(cv2.aruco.DetectorParameters_create())
        return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)[:2]


def marker_object_points(size):
    s = size / 2.0
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)


def load_calibration(path, expected_size):
    data = json.loads(Path(path).read_text())
    K = np.array(data["K"], dtype=np.float64)
    D = np.array(data["D"], dtype=np.float64).reshape(-1, 1)
    size = tuple(data["image_size"])
    if size != tuple(expected_size):
        print(f"[goto_tag] WARNING: calibration image_size {size} != live eye size "
              f"{expected_size}; poses will be wrong. Recalibrate at this resolution.")
    return K, D


def update_tracks(tracks, now, det_id, center):
    for t in tracks:
        if t["id"] == det_id and np.linalg.norm(t["center"] - center) < TRACK_MATCH_PX:
            t["center"] = center
            t["hits"] += 1
            t["t"] = now
            return t
    t = {"id": det_id, "center": center, "hits": 1, "t": now}
    tracks.append(t)
    return t


def control_loop(args):
    detect = build_detector(args.dict_name, args.min_perimeter_rate, args.error_correction_rate)
    obj_pts = marker_object_points(args.marker_size)

    cam = Config("cam_head")
    eye_size = (cam.width // 2, cam.height)
    K, D = load_calibration(args.calib_path, eye_size)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, eye_size, cv2.CV_16SC2)

    tracks = []
    target_mm = args.target_distance_mm
    bearing_cutoff = np.radians(45.0)
    bearing_deadband = np.radians(args.bearing_deadband_deg)
    last_target_pose, last_target_t = None, -1e9

    writer_cm = Writer("drive.ctrl", Type("drive_ctrl")) if not args.dry_run else None
    if writer_cm is not None:
        writer_cm.__enter__()

    def send_twist(v, w):
        if writer_cm is not None:
            with writer_cm.buf() as buf:
                buf["twist"] = np.array([v, w], dtype=np.float32)

    try:
        with Reader("camera.head.jpeg") as r:
            while not STOP:
                if not r.ready():
                    time.sleep(0.005)
                    continue
                n = int(r.data["jpeg_len"])
                buf = r.data["jpeg"][:n].copy()
                stereo = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if stereo is None:
                    continue
                eye = crop_eye(stereo, args.side)
                gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                corners, ids = detect(gray)
                undistorted = cv2.remap(eye, map1, map2, interpolation=cv2.INTER_LINEAR)

                now = time.monotonic()
                tracks[:] = [t for t in tracks if now - t["t"] < TRACK_MAX_AGE_S]

                target_pose = None  # (bearing_rad, distance_mm) for args.marker_id this frame
                if ids is not None:
                    for c, i in zip(corners, ids.flatten()):
                        ud = cv2.fisheye.undistortPoints(
                            c.reshape(-1, 1, 2).astype(np.float64), K, D, P=K).reshape(-1, 2)
                        ok, rvec, tvec = cv2.solvePnP(obj_pts, ud, K, None, flags=PNP_FLAG)
                        if not ok:
                            continue
                        track = update_tracks(tracks, now, int(i), ud.mean(axis=0))
                        poly = ud.astype(np.int32)
                        confirmed = track["hits"] >= args.confirm_hits
                        is_target = (args.marker_id < 0) or (int(i) == args.marker_id)
                        color = (0, 255, 0) if (is_target and confirmed) else (
                            (0, 165, 255) if confirmed else (90, 90, 90))
                        cv2.polylines(undistorted, [poly], True, color, 2 if confirmed else 1)
                        dist_mm = float(np.linalg.norm(tvec) * 1000)
                        if confirmed:
                            cv2.putText(undistorted, f"id={int(i)} d={dist_mm:.0f}mm",
                                        (int(poly[0][0]), max(0, int(poly[0][1]) - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        if is_target and confirmed:
                            bearing = float(np.arctan2(tvec[0, 0], tvec[2, 0]))
                            target_pose = (bearing, dist_mm)
                            last_target_pose, last_target_t = target_pose, now

                # cv2.aruco flickers a real, present tag out for the odd single frame (motion
                # blur, an unlucky threshold pass, whatever) -- treating that as "lost" every
                # time is what read as "disappears every other frame". Coast on the last
                # confirmed pose for MEMORY_S before actually falling back to search.
                if target_pose is None and last_target_pose is not None and \
                        now - last_target_t < MEMORY_S:
                    target_pose = last_target_pose

                if target_pose is None:
                    mode = "searching"
                    v, w = 0.0, args.search_w
                elif target_pose[1] <= target_mm:
                    mode = "arrived"
                    v, w = 0.0, 0.0
                else:
                    mode = "approaching"
                    bearing, dist_mm = target_pose
                    # Sign per direct observation driving the real robot: positive bearing
                    # (tag to the right) needs positive w to turn toward it on this robot.
                    if abs(bearing) < bearing_deadband:
                        w = 0.0
                    else:
                        w = float(np.clip(-args.kp_yaw * bearing, -args.max_w, args.max_w))
                    fwd_gain = max(0.0, 1.0 - abs(bearing) / bearing_cutoff)
                    v = float(np.clip(args.kp_dist * (dist_mm - target_mm) / 1000.0,
                                       0.0, args.max_v)) * fwd_gain

                send_twist(v, w)

                detail = {"mode": mode, "v": round(v, 3), "w": round(w, 3)}
                if target_pose is not None:
                    detail["target_bearing_deg"] = round(np.degrees(target_pose[0]), 1)
                    detail["target_distance_mm"] = round(target_pose[1], 1)
                with LOCK:
                    LATEST["mode"] = mode
                    LATEST["detail"] = detail

                hud = f"mode={mode} v={v:.2f} w={w:.2f}"
                cv2.putText(undistorted, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

                ok, jpg = cv2.imencode(".jpg", undistorted, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    data = jpg.tobytes()
                    try:
                        FRAME_Q.put_nowait(data)
                    except Exception:
                        try:
                            FRAME_Q.get_nowait()
                            FRAME_Q.put_nowait(data)
                        except Exception:
                            pass

                if mode == "arrived":
                    print(f"[goto_tag] arrived: {target_pose[1]:.0f}mm <= {target_mm:.0f}mm target")
                    break
    finally:
        send_twist(0.0, 0.0)
        if writer_cm is not None:
            writer_cm.__exit__(None, None, None)


def make_stream():
    async def generate():
        while True:
            try:
                frame = FRAME_Q.get_nowait()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Empty:
                pass
            await asyncio.sleep(1 / FPS)
    return generate


_gen = make_stream()


@app.get("/stream")
async def stream():
    return StreamingResponse(
        _gen(), media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/frame")
async def frame():
    try:
        f = FRAME_Q.get(timeout=0.5)
        return Response(content=f, media_type="image/jpeg")
    except Exception:
        return Response(status_code=503)


@app.get("/status")
async def status():
    with LOCK:
        return {"mode": LATEST["mode"], **LATEST["detail"]}


@app.get("/")
async def index():
    html = f"""
    <html><head><title>Goto Tag ({STATE.get('marker_id')})</title>
    <style>
      body {{ background:#000; color:#eee; font-family:sans-serif; margin:0; padding:20px; }}
      img {{ max-width:100%; display:block; margin-bottom:14px; border:1px solid #333; }}
      #status {{ white-space:pre; background:#111; padding:12px; border-radius:6px; max-width:600px; }}
    </style></head>
    <body>
      <h2>Goto tag &mdash; dict={STATE.get('dict')} id={STATE.get('marker_id')}
          target={STATE.get('target_mm')}mm {"(DRY RUN)" if STATE.get('dry_run') else ""}</h2>
      <img src="/stream">
      <pre id="status">loading...</pre>
      <script>
        async function refresh() {{
          try {{
            const r = await fetch('/status');
            document.getElementById('status').textContent = JSON.stringify(await r.json(), null, 2);
          }} catch (e) {{}}
        }}
        setInterval(refresh, 300);
        refresh();
      </script>
    </body></html>
    """
    return Response(content=html, media_type="text/html")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["left", "right"], default="left")
    ap.add_argument("--calib", type=str, default=None,
                     help="calibration json path (default /home/bracketbot/fisheye_calib_<side>.json)")
    ap.add_argument("--dict", dest="dict_name", type=str, default="4X4_50", choices=list(ARUCO_DICTS))
    ap.add_argument("--marker-id", type=int, default=5,
                     help="the SPECIFIC tag id to drive to (default 5) -- unlike view_pnp.py, "
                          "this does not accept -1/any-id, navigation needs one named target")
    ap.add_argument("--marker-size", type=float, default=0.069,
                     help="marker side length in meters (default 0.069 = 69mm)")
    ap.add_argument("--target-distance-mm", type=float, default=700.0,
                     help="stop when within this many mm of the tag (default 700)")
    ap.add_argument("--min-perimeter-rate", type=float, default=0.015,
                     help="detector leniency (default 0.015 -- more conservative than "
                          "view_pnp.py's default, see module docstring)")
    ap.add_argument("--error-correction-rate", type=float, default=0.6,
                     help="id-decode bit-error tolerance (default 0.6, cv2's own stock default)")
    ap.add_argument("--confirm-hits", type=int, default=3,
                     help="consecutive near-same-spot detections required to trust/act on a "
                          "candidate (default 3, stricter than view_pnp.py's 2)")
    ap.add_argument("--max-v", type=float, default=0.15, help="forward speed cap, m/s (default 0.15)")
    ap.add_argument("--max-w", type=float, default=0.5, help="turn speed cap, rad/s (default 0.5)")
    ap.add_argument("--search-w", type=float, default=0.3,
                     help="in-place rotation speed while searching for the tag, rad/s (default 0.3)")
    ap.add_argument("--kp-yaw", type=float, default=1.5, help="P gain, rad/s per rad of bearing error")
    ap.add_argument("--kp-dist", type=float, default=0.5, help="P gain, m/s per meter of distance error")
    ap.add_argument("--bearing-deadband-deg", type=float, default=3.0,
                     help="stop correcting heading once the tag is within this many degrees "
                          "of dead-center (default 3.0) -- prevents constant micro-turning from "
                          "detection jitter once it's roughly lined up")
    ap.add_argument("--dry-run", action="store_true",
                     help="compute and display everything but never open the drive Writer -- "
                          "the robot cannot move. Use this to check the pipeline before a live run")
    ap.add_argument("--port", type=int, default=8013)
    args = ap.parse_args()
    args.calib_path = args.calib or f"/home/bracketbot/fisheye_calib_{args.side}.json"

    if not Path(args.calib_path).exists():
        raise SystemExit(
            f"No calibration file at {args.calib_path}.\n"
            f"Run this first:  uv run examples/calibrate_fisheye.py --side {args.side}\n"
        )

    STATE.update(side=args.side, dict=args.dict_name, marker_id=args.marker_id,
                 target_mm=args.target_distance_mm, dry_run=args.dry_run)

    threading.Thread(target=control_loop, args=(args,), daemon=True).start()

    host = socket.gethostname()
    print(f"[+] Goto tag ({'DRY RUN, ' if args.dry_run else ''}dict={args.dict_name} "
          f"id={args.marker_id}) on http://{host}.local:{args.port}/")
    print(f"[+] Calibration <- {args.calib_path}")
    print(f"[+] target={args.target_distance_mm}mm max_v={args.max_v} max_w={args.max_w} "
          f"bearing_deadband_deg={args.bearing_deadband_deg}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="error",
                access_log=False, timeout_graceful_shutdown=1)


if __name__ == "__main__":
    main()
