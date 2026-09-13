# /// script
# dependencies = ["bbos", "numpy<2"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
from bbos import Config

qcfg = Config("quest")
print("quest.robot_shoulder_height:", float(qcfg.robot_shoulder_height), "m")

cfg = Config("arm_left")
print("arm_left j0 wheel_radius (lead screw radius):", cfg.wheel_radius)
print("arm_left home (motor turns):", cfg.home)
