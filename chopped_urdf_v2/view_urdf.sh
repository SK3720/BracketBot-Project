#!/usr/bin/env bash
# View a robot in Rerun, with sliders to pose the joints.
#   --urdf chopped_urdf_v2/urdf/chopped_urdf_v2.urdf   for the simplified model
#
#   ./view_urdf.sh                 # viewer + joint sliders
#   ./view_urdf.sh --no-sliders    # viewer only
#   ./view_urdf.sh --list-joints   # print the movable joints
#   ./view_urdf.sh --save bot.rrd  # write a recording (headless)
#
# Deps (rerun-sdk, numpy) are provisioned into ./.venv on first run — via uv if
# it's installed, otherwise with the system python3.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v uv >/dev/null 2>&1; then
    exec uv run --project "$DIR" python "$DIR/visualize_urdf.py" "$@"
fi

VENV="$DIR/.venv"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
    command -v python3 >/dev/null 2>&1 || {
        echo "need python3 (install the Xcode command line tools: xcode-select --install)" >&2
        exit 1
    }
    echo "creating $VENV ..."
    python3 -m venv "$VENV"
fi

# also covers a half-provisioned venv from an interrupted first run
if ! "$PY" -c "import rerun, numpy" >/dev/null 2>&1; then
    echo "installing rerun-sdk + numpy (one time, ~90 MB) ..."
    "$PY" -m pip install --quiet --upgrade pip
    "$PY" -m pip install --quiet "rerun-sdk>=0.26" "numpy>=1.24"
fi

exec "$PY" "$DIR/visualize_urdf.py" "$@"
