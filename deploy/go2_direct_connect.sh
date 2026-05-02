#!/bin/bash
echo "Remember to run this file with: source deploy/go2_direct_connect.sh"

if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/ros2" ]; then
	export PATH="$CONDA_PREFIX/bin:$PATH"
fi

# Avoid accidentally importing message packages from another ROS workspace
# such as ~/go2_ws, which may have been built against a different Python ABI.
unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH

if [ -n "$CONDA_PREFIX" ]; then
	export AMENT_PREFIX_PATH="$CONDA_PREFIX"
	export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
fi

export ROS_LOCALHOST_ONLY=0
unset ROS_DISCOVERY_SERVER
unset ROS_SUPER_CLIENT
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_WS="$SCRIPT_DIR/ros2_ws"

if [ -f "$ROS2_WS/install/setup.bash" ]; then
	source "$ROS2_WS/install/setup.bash"
else
	echo "Warning: $ROS2_WS/install/setup.bash not found. Build with:"
	echo "  cd $ROS2_WS && colcon build --symlink-install"
	return 1
fi

ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start >/dev/null 2>&1 || true
echo "Go2 direct DDS environment configured."
