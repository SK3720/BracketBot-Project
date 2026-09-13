# /// script
# dependencies = ["bbos", "numpy<2"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
import numpy as np
from bbos import Config

TARGET_LEFT_MOTOR_J0 = 0.5498242378234863
TARGET_RIGHT_MOTOR_J0 = -0.5769238471984863

for name, motor_j0 in [("arm_left", TARGET_LEFT_MOTOR_J0), ("arm_right", TARGET_RIGHT_MOTOR_J0)]:
    cfg = Config(name)
    urdf_j0 = motor_j0 * 2 * np.pi * cfg.wheel_radius * cfg.ik_sign[0]
    print(f"{name}: wheel_radius={cfg.wheel_radius} ik_sign[0]={cfg.ik_sign[0]}")
    print(f"  preset motor_j0={motor_j0:.4f} -> urdf_j0={urdf_j0:.4f} m")
    print(f"  headroom to top (urdf=0): {-urdf_j0*100:.1f} cm")
    print(f"  headroom to bottom (urdf=-1.03): {(urdf_j0 - (-1.03))*100:.1f} cm")
