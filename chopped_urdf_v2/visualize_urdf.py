#!/usr/bin/env python3
"""Visualize the urdf_v2 robot in Rerun, with sliders to pose the joints.

Parses the URDF, builds the link/joint kinematic tree, loads the STL meshes
(resolving `package://urdf_v2/...` to the local meshes dir) and logs everything
to Rerun. Joint transforms are logged as a Rerun transform hierarchy, so forward
kinematics compose automatically and posing a joint is a single re-log.

The slider panel is a small local web page (stdlib http.server) rather than a Tk
window: macOS ships Tk 8.5.9, which paints nothing on current macOS.

Usage:
    ./view_urdf.sh                  # viewer + slider panel in the browser
    ./view_urdf.sh --no-sliders     # viewer only, robot at its zero pose
    ./view_urdf.sh --joint lj2=0.8  # one-shot pose (rad / m), repeatable
    ./view_urdf.sh --list-joints    # print the movable joints and exit
    ./view_urdf.sh --save bot.rrd   # write a recording instead (works headless)

Dependencies: rerun-sdk, numpy (provisioned in ./.venv by view_urdf.sh).
"""
from __future__ import annotations

import argparse
import math
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np
import rerun as rr

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URDF = os.path.join(HERE, "urdf_v2", "urdf", "urdf_v2.urdf")
DEFAULT_COLOR = (0.7, 0.7, 0.7, 1.0)  # used when a visual has no material color
ROTATIONAL = ("revolute", "continuous")


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def rpy_to_matrix(rpy):
    """URDF roll-pitch-yaw (fixed axis XYZ) -> 3x3 rotation matrix Rz @ Ry @ Rx."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def axis_angle_matrix(axis, angle):
    """Rotation matrix for `angle` rad about `axis` (Rodrigues)."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = a / n
    c, s = math.cos(angle), math.sin(angle)
    cc = 1.0 - c
    return np.array([
        [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
        [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
        [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
    ])


def joint_transform(joint, value):
    """(translation, 3x3 rotation) of a joint's child frame at the given value.

    value is radians for revolute/continuous, meters for prismatic.
    """
    r0 = rpy_to_matrix(joint["rpy"])
    t0 = np.asarray(joint["xyz"], dtype=float)
    jtype = joint["type"]
    if jtype in ROTATIONAL:
        return t0, r0 @ axis_angle_matrix(joint["axis"], value)
    if jtype == "prismatic":
        a = np.asarray(joint["axis"], dtype=float)
        a = a / (np.linalg.norm(a) or 1.0)
        return t0 + r0 @ (a * value), r0
    return t0, r0  # fixed


def parse_origin(elem):
    """Return (xyz, rpy) from an <origin> child, defaulting to identity."""
    xyz = (0.0, 0.0, 0.0)
    rpy = (0.0, 0.0, 0.0)
    if elem is not None:
        o = elem.find("origin")
        if o is not None:
            if o.get("xyz"):
                xyz = tuple(float(v) for v in o.get("xyz").split())
            if o.get("rpy"):
                rpy = tuple(float(v) for v in o.get("rpy").split())
    return xyz, rpy


def load_stl(path):
    """Load a binary or ASCII STL. Returns (vertices (V,3), faces (T,3), normals (V,3))."""
    with open(path, "rb") as fh:
        data = fh.read()
    is_ascii = data[:5].lower() == b"solid" and b"facet" in data[:512].lower()
    if is_ascii:
        verts = np.array(
            [[float(x) for x in m] for m in re.findall(
                rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)],
            dtype=np.float32,
        )
        n = len(verts) // 3
        verts = verts[: n * 3]
        faces = np.arange(n * 3, dtype=np.uint32).reshape(n, 3)
        # derive face normals from geometry
        tri = verts.reshape(n, 3, 3)
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norms = np.repeat(fn, 3, axis=0)
        return verts, faces, norms

    # binary STL: 80-byte header, uint32 count, then 50 bytes per triangle
    n = struct.unpack("<I", data[80:84])[0]
    buf = np.frombuffer(data, dtype=np.uint8, count=50 * n, offset=84).reshape(n, 50)
    floats = buf[:, 12:48].copy().view("<f4").reshape(n, 3, 3)  # 3 verts x 3 coords
    face_norm = buf[:, 0:12].copy().view("<f4").reshape(n, 3)
    verts = floats.reshape(-1, 3).astype(np.float32)
    faces = np.arange(n * 3, dtype=np.uint32).reshape(n, 3)
    norms = np.repeat(face_norm, 3, axis=0).astype(np.float32)
    return verts, faces, norms


# --------------------------------------------------------------------------- #
# URDF parsing
# --------------------------------------------------------------------------- #
def resolve_mesh(filename, pkg_dir):
    """Resolve a URDF mesh filename to a local filesystem path."""
    fn = re.sub(r"^package://[^/]+/", "", filename)
    fn = re.sub(r"^file://", "", fn)
    return fn if os.path.isabs(fn) else os.path.join(pkg_dir, fn)


def parse_urdf(urdf_path):
    """Return (links, joints, roots) — links maps name -> list of visuals."""
    root = ET.parse(urdf_path).getroot()
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(urdf_path)))  # .../<pkg>

    links = {}
    for link in root.findall("link"):
        visuals = []
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            xyz, rpy = parse_origin(vis)
            scale = (1.0, 1.0, 1.0)
            if mesh.get("scale"):
                scale = tuple(float(s) for s in mesh.get("scale").split())
            color = None
            col = vis.find("material/color")
            if col is not None and col.get("rgba"):
                color = tuple(float(c) for c in col.get("rgba").split())
            visuals.append({
                "xyz": xyz, "rpy": rpy, "scale": scale, "color": color,
                "mesh": resolve_mesh(mesh.get("filename"), pkg_dir),
            })
        links[link.get("name")] = visuals

    joints = []
    children = set()
    for j in root.findall("joint"):
        xyz, rpy = parse_origin(j)
        axis = (1.0, 0.0, 0.0)  # URDF default
        ax = j.find("axis")
        if ax is not None and ax.get("xyz"):
            axis = tuple(float(a) for a in ax.get("xyz").split())
        lower = upper = None
        lim = j.find("limit")
        if lim is not None:
            lower = float(lim.get("lower")) if lim.get("lower") else None
            upper = float(lim.get("upper")) if lim.get("upper") else None
        mimic = None
        m = j.find("mimic")
        if m is not None:
            mimic = {
                "joint": m.get("joint"),
                "multiplier": float(m.get("multiplier", 1.0)),
                "offset": float(m.get("offset", 0.0)),
            }
        child = j.find("child").get("link")
        joints.append({
            "name": j.get("name"), "type": j.get("type", "fixed"), "axis": axis,
            "parent": j.find("parent").get("link"), "child": child,
            "xyz": xyz, "rpy": rpy, "lower": lower, "upper": upper, "mimic": mimic,
        })
        children.add(child)

    roots = [name for name in links if name not in children]
    return links, joints, roots


def build_paths(links, joints, roots):
    """Walk the tree, returning [(link, rerun entity path, joint into it)] in tree order."""
    child_joint = {j["child"]: j for j in joints}
    adj = {}
    for j in joints:
        adj.setdefault(j["parent"], []).append(j["child"])

    order = []

    def walk(link, parent_path):
        path = f"{parent_path}/{link}" if parent_path else link
        order.append((link, path, child_joint.get(link)))
        for c in adj.get(link, []):
            walk(c, path)

    for r in roots:
        walk(r, "")
    return order


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def set_joint(joint, path, value, degrees=False):
    """(Re-)log a single joint's child transform at the given command value."""
    if degrees and joint["type"] in ROTATIONAL:
        value = math.radians(value)
    t, r = joint_transform(joint, value)
    rr.log(path, rr.Transform3D(translation=t.tolist(), mat3x3=r), static=True)


def apply_joint(joint, path, value, followers=(), degrees=False):
    """Set a driver joint, then cascade to every joint that <mimic>s it.

    A follower's value is `multiplier * value + offset` (URDF mimic semantics).
    """
    set_joint(joint, path, value, degrees)
    for mj, mpath, mult, off in followers:
        set_joint(mj, mpath, mult * value + off, degrees)


def log_robot(urdf_path, joint_values=None, degrees=False):
    """Log the whole robot. Returns {joint_name: (joint, child_path, followers)}
    for the movable joints (mimic joints are driven, so they aren't listed)."""
    joint_values = joint_values or {}
    links, joints, roots = parse_urdf(urdf_path)
    order = build_paths(links, joints, roots)

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # entity path each joint's transform is logged at, and the mimic followers
    # grouped by the joint that drives them
    path_by_name = {j["name"]: p for (_l, p, j) in order if j is not None}
    followers = {}  # driver name -> [(mimic_joint, mimic_path, multiplier, offset)]
    for j in joints:
        m = j["mimic"]
        if m is not None and j["name"] in path_by_name:
            followers.setdefault(m["joint"], []).append(
                (j, path_by_name[j["name"]], m["multiplier"], m["offset"]))

    commandable = {}
    n_links = n_visuals = n_tris = 0
    for link, path, joint in order:
        # the joint defines the transform from the parent link to this link
        if joint is not None:
            name = joint["name"]
            if joint["mimic"] is not None:
                # driven by another joint; never commanded directly
                m = joint["mimic"]
                value = m["multiplier"] * joint_values.get(m["joint"], 0.0) + m["offset"]
            else:
                value = joint_values.get(name, 0.0)
                if joint["type"] != "fixed":
                    commandable[name] = (joint, path, followers.get(name, []))
            set_joint(joint, path, value, degrees)
        n_links += 1

        for i, vis in enumerate(links[link]):
            try:
                verts, faces, norms = load_stl(vis["mesh"])
            except (OSError, ValueError) as e:
                print(f"  ! skipping {vis['mesh']}: {e}")
                continue
            verts = verts * np.asarray(vis["scale"], dtype=np.float32)
            vpath = f"{path}/visual_{i}"
            rr.log(
                vpath,
                rr.Transform3D(translation=list(vis["xyz"]),
                               mat3x3=rpy_to_matrix(vis["rpy"])),
                static=True,
            )
            rr.log(
                vpath,
                rr.Mesh3D(vertex_positions=verts, triangle_indices=faces,
                          vertex_normals=norms,
                          albedo_factor=vis["color"] or DEFAULT_COLOR),
                static=True,
            )
            n_visuals += 1
            n_tris += len(faces)

    print(f"logged {n_links} links, {n_visuals} meshes, {n_tris:,} triangles "
          f"(root: {', '.join(roots)})")
    return commandable


# --------------------------------------------------------------------------- #
# Command line / GUI
# --------------------------------------------------------------------------- #
def parse_joint_args(items):
    """Parse repeated NAME=VALUE / 'NAME VALUE' joint args into a dict."""
    out = {}
    for item in items or []:
        token = item.replace("=", " ").split()
        if len(token) != 2:
            raise SystemExit(f"bad --joint '{item}', expected NAME=VALUE")
        out[token[0]] = float(token[1])
    return out


def list_joints(urdf_path):
    _, joints, _ = parse_urdf(urdf_path)
    movable = [j for j in joints if j["type"] != "fixed"]
    drivers = [j for j in movable if j["mimic"] is None]
    print(f"{len(drivers)} movable joints (rotational = radians, prismatic = meters):")
    for j in drivers:
        unit = "rad" if j["type"] in ROTATIONAL else "m"
        print(f"  {j['name']:<24} {j['type']:<11} [{unit}]  -> {j['child']}")
    for j in (j for j in movable if j["mimic"] is not None):
        m = j["mimic"]
        print(f"  {j['name']:<24} mimics {m['joint']} "
              f"(x{m['multiplier']:g}, offset {m['offset']:g})")


ROT_RANGE = 180.0   # degrees, +/- , when the joint declares no usable limit
PRIS_RANGE = 0.30   # meters, +/-


def slider_range(joint):
    """(lo, hi, step, unit) for a joint's slider.

    Uses the URDF limits when they're usable, otherwise a sane default —
    continuous joints carry no limit at all, and the Onshape exporter writes
    +/-10000 placeholders on prismatic joints, which make for a useless slider.
    """
    lo, hi = joint["lower"], joint["upper"]
    usable = lo is not None and hi is not None and hi > lo
    if joint["type"] in ROTATIONAL:
        if usable and (hi - lo) <= 4 * math.pi:
            return math.degrees(lo), math.degrees(hi), 1.0, "deg"
        return -ROT_RANGE, ROT_RANGE, 1.0, "deg"
    if usable and max(abs(lo), abs(hi)) <= 2.0:
        return lo, hi, 0.001, "m"
    return -PRIS_RANGE, PRIS_RANGE, 0.005, "m"


def is_dummy(name):
    """True for the joints Onshape emits to break closed loops (ball_*_loop_closure)
    and to decompose a planar mate (planar_*). They're real DOF in the URDF, but
    moving one detaches the geometry it was closing the loop on."""
    return "loop_closure" in name or name.startswith("planar_")


PAGE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ — joints</title>
<style>
  :root { color-scheme: dark light }
  body { margin:0; padding:24px 20px 60px; background:#15171a; color:#e7e9ec;
         font:14px/1.4 ui-sans-serif,-apple-system,Helvetica,sans-serif }
  h1 { font-size:15px; font-weight:600; margin:0 0 2px }
  p.sub { margin:0 0 20px; color:#8b929c; font-size:13px }
  .row { display:flex; align-items:center; gap:10px; padding:3px 0 }
  .name { width:190px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
          font-size:12.5px; color:#c8ccd2; overflow:hidden; text-overflow:ellipsis }
  input[type=range] { flex:1; min-width:120px; accent-color:#5b9dd9; height:20px }
  .val { width:74px; text-align:right; font-family:ui-monospace,Menlo,monospace;
         font-size:12.5px; color:#9aa3ad; font-variant-numeric:tabular-nums }
  h2 { font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.06em;
       color:#8b929c; margin:26px 0 6px; border-top:1px solid #2a2e34; padding-top:14px }
  h2 span { text-transform:none; letter-spacing:0; font-weight:400; color:#6f767f }
  button { background:#252a30; color:#e7e9ec; border:1px solid #363c44; border-radius:6px;
           padding:7px 13px; font-size:13px; cursor:pointer; margin-right:8px }
  button:hover { background:#2d333a }
  #bar { position:fixed; left:0; right:0; bottom:0; padding:12px 20px;
         background:#1b1e22; border-top:1px solid #2a2e34; display:flex; align-items:center }
  #status { color:#6f767f; font-size:12px; margin-left:auto }
</style>
<h1>__TITLE__ — joint control</h1>
<p class="sub">Drag a slider; the Rerun viewer updates live.</p>
<div id="rows"></div>
<div id="bar">
  <button onclick="resetAll()">Reset all to 0</button>
  <button onclick="quit()">Quit</button>
  <span id="status"></span>
</div>
<script>
const JOINTS = __JOINTS__;
const rowsEl = document.getElementById("rows");
const statusEl = document.getElementById("status");
const inputs = {};

function addRow(j, parent) {
  const row = document.createElement("div"); row.className = "row";
  const name = document.createElement("div"); name.className = "name";
  name.textContent = j.name; name.title = j.name;
  const s = document.createElement("input");
  s.type = "range"; s.min = j.lo; s.max = j.hi; s.step = j.step; s.value = 0;
  const v = document.createElement("div"); v.className = "val";
  const show = x => v.textContent = (j.unit === "deg" ? Number(x).toFixed(0) + "\\u00b0"
                                                      : Number(x).toFixed(3) + " m");
  show(0);
  s.addEventListener("input", () => { show(s.value); queue(j.name, Number(s.value)); });
  row.append(name, s, v); parent.appendChild(row);
  inputs[j.name] = { input: s, show };
}

const main = JOINTS.filter(j => !j.dummy), dummies = JOINTS.filter(j => j.dummy);
main.forEach(j => addRow(j, rowsEl));
if (dummies.length) {
  const h = document.createElement("h2");
  h.innerHTML = 'closed-loop / planar joints <span>— Onshape dummy links; ' +
                'dragging these pulls the model apart</span>';
  rowsEl.appendChild(h);
  dummies.forEach(j => addRow(j, rowsEl));
}

// coalesce slider input into one request per animation frame
let pending = {}, scheduled = false;
function queue(name, value) {
  pending[name] = value;
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(flush);
}
async function flush() {
  const values = pending; pending = {}; scheduled = false;
  try {
    await fetch("/set", { method: "POST", body: JSON.stringify({ values }) });
    statusEl.textContent = "";
  } catch (e) { statusEl.textContent = "viewer disconnected"; }
}
function resetAll() {
  const values = {};
  for (const [name, o] of Object.entries(inputs)) { o.input.value = 0; o.show(0); values[name] = 0; }
  fetch("/set", { method: "POST", body: JSON.stringify({ values }) });
}
function quit() {
  fetch("/quit", { method: "POST" }).catch(() => {});
  statusEl.textContent = "stopped — you can close this tab";
}
</script>
"""


def slider_server(commandable, title="urdf_v2", port=0, open_browser=True):
    """Serve a browser slider panel on localhost; each change re-logs a joint.

    A web page instead of a Tk one because macOS ships Tk 8.5.9, whose Aqua
    backend paints nothing on current macOS — the window comes up empty.
    """
    import http.server
    import json
    import threading
    import webbrowser

    meta, lookup = [], {}
    for name, (joint, path, foll) in commandable.items():
        lo, hi, step, unit = slider_range(joint)
        meta.append({"name": name, "lo": lo, "hi": hi, "step": step,
                     "unit": unit, "dummy": is_dummy(name)})
        lookup[name] = (joint, path, foll, unit == "deg")

    page = (PAGE.replace("__JOINTS__", json.dumps(meta))
                .replace("__TITLE__", title)).encode()
    lock = threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, code, body=b"", ctype="text/plain; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._reply(200, page, "text/html; charset=utf-8")
            else:
                self._reply(404, b"not found")

        def do_POST(self):
            if self.path == "/quit":
                self._reply(200, b"bye")
                threading.Thread(target=httpd.shutdown, daemon=True).start()
                return
            if self.path != "/set":
                self._reply(404, b"not found")
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                values = json.loads(self.rfile.read(n) or b"{}").get("values", {})
            except ValueError:
                self._reply(400, b"bad json")
                return
            applied = 0
            with lock:  # one writer at a time into the recording
                for name, value in values.items():
                    if name in lookup:
                        joint, path, foll, degrees = lookup[name]
                        apply_joint(joint, path, float(value), foll, degrees)
                        applied += 1
            self._reply(200, json.dumps({"applied": applied}).encode(),
                        "application/json")

        def log_message(self, *_args):
            pass  # keep the terminal clean

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{httpd.server_port}/"
    print(f"slider panel: {url}  ({len(meta)} joints — Ctrl-C here to quit)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()


def main():
    ap = argparse.ArgumentParser(
        description="Visualize the urdf_v2 robot in Rerun and pose its joints.")
    ap.add_argument("--urdf", default=DEFAULT_URDF, help="path to the .urdf file")
    ap.add_argument("--no-sliders", dest="sliders", action="store_false",
                    help="just open the viewer, without the joint slider panel")
    ap.add_argument("--joint", action="append", metavar="NAME=VALUE", default=[],
                    help="set a joint value (repeatable); rad for rotational, m for prismatic")
    ap.add_argument("--degrees", action="store_true",
                    help="interpret --joint values for rotational joints as degrees")
    ap.add_argument("--list-joints", action="store_true",
                    help="print the movable joints and exit")
    ap.add_argument("--save", metavar="PATH.rrd",
                    help="write a recording instead of opening the viewer (headless)")
    ap.add_argument("--port", type=int, default=0,
                    help="port for the slider panel (default: pick a free one)")
    ap.add_argument("--no-browser", dest="browser", action="store_false",
                    help="don't open the slider panel automatically; just print its URL")
    ap.add_argument("--app-id", default="urdf_v2")
    args = ap.parse_args()

    if args.list_joints:
        list_joints(args.urdf)
        return

    joint_values = parse_joint_args(args.joint)

    # rr.spawn() searches PATH for the `rerun` viewer binary; it's installed next
    # to this venv's python, which isn't necessarily on PATH. Add it.
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

    rr.init(args.app_id, spawn=args.save is None)
    if args.save:
        rr.save(args.save)
    commandable = log_robot(args.urdf, joint_values, args.degrees)

    if args.save:
        print(f"saved recording to {args.save}  (open with: rerun {args.save})")
    elif args.sliders:
        slider_server(commandable, os.path.basename(args.urdf).replace(".urdf", ""),
                      port=args.port, open_browser=args.browser)


if __name__ == "__main__":
    main()
