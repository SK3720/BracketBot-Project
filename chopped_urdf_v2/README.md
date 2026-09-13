# chopped_urdf_v2

Self-contained copy of the BracketBot `chopped_urdf_v2` model: the URDF, every
mesh it references, and a standalone Rerun viewer.

    chopped_urdf_v2/
      urdf/chopped_urdf_v2.urdf       the model (54 links, 53 joints)
      meshes/*.stl                    all 50 referenced meshes
      package.xml                     so package:// and $(find ...) resolve
      launch/chopped_urdf_v2.launch   RViz + robot_state_publisher (ROS 1)
      manifest.json, draco/*.glb      Draco-compressed meshes for web viewers
    visualize_urdf.py, view_urdf.sh   Rerun viewer (no ROS needed)
    pyproject.toml                    deps for the viewer

The URDF refers to its meshes as `package://chopped_urdf_v2/meshes/<name>.stl`,
so keep the `chopped_urdf_v2/` directory intact — loaders resolve those paths
relative to the package root.

## Load it in Rerun (no ROS)

    ./view_urdf.sh --urdf chopped_urdf_v2/urdf/chopped_urdf_v2.urdf

First run provisions `rerun-sdk` and `numpy` into `./.venv` (via `uv` if
installed, otherwise the system `python3`). This opens the Rerun viewer plus a
browser slider panel for posing the joints. Other modes:

    ./view_urdf.sh --urdf ... --no-sliders          # viewer only, zero pose
    ./view_urdf.sh --urdf ... --list-joints         # print the movable joints
    ./view_urdf.sh --urdf ... --joint lj2=0.8       # one-shot pose (rad / m)
    ./view_urdf.sh --urdf ... --save bot.rrd        # headless recording

## Load it in ROS 1

Drop `chopped_urdf_v2/` into a catkin workspace's `src/`, build, then:

    roslaunch chopped_urdf_v2 chopped_urdf_v2.launch

## The model

Root link is `root`, at the wheel-axle midpoint with the tyres on z=0.

18 movable joints:

| Joint | Type | Range |
| --- | --- | --- |
| `rj0` / `lj0` | prismatic | -1.03044 .. 0 m (mast carriage) |
| `rj1`-`rj6` / `lj1`-`lj6` | revolute | ±2.094395 rad |
| `right_left_gripper`, `right_right_gripper` | revolute | 0 .. 1 rad |
| `left_left_gripper`, `left_right_gripper` | revolute | 0 .. 1 rad |

Each arm's second gripper joint `mimic`s the first, so a gripper is one DOF.
Every joint axis is exactly `(0,0,1)` — this was prepared for QP IK, where a raw
Onshape export snapped to the nearest principal axis puts FK ~27 cm out with no
warning. For IK the arm base link is `arm_base` and the end effectors are
`right_eef` / `left_eef`.
