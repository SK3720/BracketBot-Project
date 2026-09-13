# /// script
# dependencies = [
#   "bbos",
#   "numpy<2",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Minimal diagnostic: is arm_left.state ever ready, and what's in it?"""
import sys
import time
from bbos import Reader, Config

arm = sys.argv[1] if len(sys.argv) > 1 else "arm_left"

cfg = Config(arm)
print(f"Config({arm!r}) loaded. dof={cfg.dof}")

print(f"Opening Reader('{arm}.state')...")
with Reader(f"{arm}.state") as r:
    for i in range(100):
        ready = r.ready()
        print(f"tick {i:3d}  ready={ready}")
        if ready:
            print("DATA pos:", r.data["pos"])
            print("DATA vel:", r.data["vel"])
            break
        time.sleep(0.05)
    else:
        print("Never became ready after 5s.")
