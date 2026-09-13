# /// script
# dependencies = [
#   "bbos",
#   "numpy<2",
#   "trimesh<5.0.0",
#   "yourdfpy>=0.0.56",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""What is the wrist camera's pose relative to the end-effector link?
This is fixed (both are downstream of the last actuated joint lj6/rj6 via
fixed-only joints), so it doesn't depend on arm configuration -- compute it
once at any joint config (zeros) via yourdfpy FK on the full URDF, rather than
measuring by hand or hand-parsing the raw joint XML.
"""
import numpy as np
import yourdfpy
from bbos import Config

cfg = Config("arm_left")
urdf = yourdfpy.URDF.load(cfg.urdf_path, load_meshes=False, build_scene_graph=True,
                          build_collision_scene_graph=False)

zero_cfg = {j: 0.0 for j in urdf.joint_map.keys()}
urdf.update_cfg(zero_cfg)


def link_pose(name):
    T = urdf.get_transform(frame_to=name, frame_from=urdf.base_link)
    return T


pairs = [
    ("left_eef", "l_wrist_cam__wrist_cam"),
    ("l_hand__hand", "l_wrist_cam__wrist_cam"),
    ("right_eef", "wrist_cam__wrist_cam"),
    ("hand__hand", "wrist_cam__wrist_cam"),
]

for eef_name, cam_name in pairs:
    try:
        T_eef = link_pose(eef_name)
        T_cam = link_pose(cam_name)
        T_eef_cam = np.linalg.inv(T_eef) @ T_cam  # camera pose IN eef frame
        print(f"\n{eef_name} -> {cam_name}:")
        print("  translation (m):", np.round(T_eef_cam[:3, 3], 4))
        print("  rotation matrix:")
        for row in T_eef_cam[:3, :3]:
            print("   ", np.round(row, 4))
    except Exception as e:
        print(f"\n{eef_name} -> {cam_name}: FAILED ({e})")

print("\nAbsolute poses from arm_base (sanity check):")
for name in ["arm_base", "left_eef", "l_hand__hand", "l_wrist_cam__wrist_cam",
             "right_eef", "hand__hand", "wrist_cam__wrist_cam"]:
    try:
        T = link_pose(name)
        print(f"  {name}: xyz={np.round(T[:3, 3], 4)}")
    except Exception as e:
        print(f"  {name}: FAILED ({e})")

print("\nAll link names containing 'eef' or 'cam':")
for name in urdf.link_map.keys():
    if "eef" in name.lower() or "cam" in name.lower():
        print(" ", name)
