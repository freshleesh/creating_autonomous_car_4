#!/bin/bash
###############################################################################
# build_packages_on_car.sh
#
# Build script for the F1TENTH car (Ubuntu 24.04 + ROS2 Jazzy).
# Builds ALL packages including sensor drivers, SLAM, and particle filter.
# Removes COLCON_IGNORE files from car-only packages that are excluded by default.
#
# Usage: bash build_packages_on_car.sh
###############################################################################

set -eo pipefail

# ============================================================================
# Resolve paths relative to this script
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "============================================"
echo " Building packages for F1TENTH CAR"
echo "============================================"
echo "Workspace : ${WS_DIR}"
echo "Source    : ${SCRIPT_DIR}"
echo ""

# ============================================================================
# 1. Install apt dependencies (all packages including car-only)
# ============================================================================
echo "[1/8] Installing apt dependencies..."
sudo apt update
sudo apt install -y \
    python3-rosdep \
    python3-pip \
    python3-skimage \
    python3-numpy \
    cython3 \
    ros-jazzy-xacro \
    ros-jazzy-rmw-cyclonedds-cpp \
    libboost-dev \
    libboost-iostreams-dev \
    libcairo2-dev \
    libceres-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    liblua5.2-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libabsl-dev \
    libpcl-dev \
    google-mock \
    gedit

# ============================================================================
# 2. Remove COLCON_IGNORE from car-only packages
#    (these are present by default in the repository for local-PC builds)
# ============================================================================
echo ""
echo "[2/8] Removing COLCON_IGNORE from car-only packages..."

CAR_ONLY_IGNORES=(
    "${SCRIPT_DIR}/sensor/vesc/COLCON_IGNORE"
    "${SCRIPT_DIR}/sensor/urg_node/COLCON_IGNORE"
    "${SCRIPT_DIR}/slam/cartographer/COLCON_IGNORE"
    "${SCRIPT_DIR}/slam/cartographer_ros/COLCON_IGNORE"
    "${SCRIPT_DIR}/slam/particle_filter/COLCON_IGNORE"
)

for ignore_file in "${CAR_ONLY_IGNORES[@]}"; do
    if [ -f "${ignore_file}" ]; then
        rm "${ignore_file}"
        echo "  Removed -> ${ignore_file}"
    else
        echo "  Not present, skipping -> ${ignore_file}"
    fi
done

# Verify permanent COLCON_IGNORE files are intact (these should never be removed)
echo ""
echo "  Verifying permanent COLCON_IGNORE files..."
PERMANENT_IGNORES=(
    "${SCRIPT_DIR}/simulator/f1tenth_gym/COLCON_IGNORE"
    "${SCRIPT_DIR}/slam/range_libc/COLCON_IGNORE"
    "${SCRIPT_DIR}/slam/cartographer_ros/docs/COLCON_IGNORE"
)
for perm_file in "${PERMANENT_IGNORES[@]}"; do
    if [ -f "${perm_file}" ]; then
        echo "  OK: ${perm_file}"
    else
        echo "  WARNING: Missing, restoring -> ${perm_file}"
        touch "${perm_file}"
    fi
done

# ============================================================================
# 3. Install Python dependencies (f1tenth_gym + transforms3d)
# ============================================================================
echo ""
echo "[3/8] Installing Python dependencies..."

echo "  Installing f1tenth_gym..."
pip install -e "${SCRIPT_DIR}/simulator/f1tenth_gym" --break-system-packages

echo "  Installing transforms3d..."
pip install transforms3d --break-system-packages

echo "  Upgrading coverage..."
pip install --upgrade coverage --break-system-packages

# pynput (keyboard/mouse input for teleop)
echo "  Installing pynput..."
pip install pynput --break-system-packages

# ============================================================================
# 4. Install range_libc (for particle filter localization)
# ============================================================================
echo ""
echo "[4/8] Installing range_libc..."
cd "${SCRIPT_DIR}/slam/range_libc/pywrapper"
pip3 install . --user --break-system-packages
cd "${WS_DIR}"

echo "  Verifying range_libc installation..."
python3 -c "import range_libc; print('  range_libc import OK')" || {
    echo "  WARNING: range_libc import failed. Check installation."
}

# ============================================================================
# 5. Initialize rosdep and install ROS dependencies
# ============================================================================
echo ""
echo "[5/8] Running rosdep..."

if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
else
    echo "  rosdep already initialized, skipping init."
fi
rosdep update

rosdep install --from-paths "${WS_DIR}/src" --ignore-src -r -y

# Check for missing dependencies and retry via apt
echo "  Checking for remaining missing dependencies..."
MISSING=$(rosdep check --from-paths "${WS_DIR}/src" --ignore-src 2>&1 \
    | grep "apt\b" | sed 's/.*apt\t//;s/^ *//' | sort -u | tr '\n' ' ' || true)

if [ -n "${MISSING}" ]; then
    echo "  Missing packages found: ${MISSING}"
    echo "  Retrying installation via apt..."
    sudo apt install -y ${MISSING} || echo "  Warning: some packages could not be installed."
else
    echo "  All dependencies are satisfied."
fi

# Ensure colcon is installed
if ! command -v colcon &> /dev/null; then
    echo "  colcon not found, installing..."
    sudo apt install -y python3-colcon-common-extensions
fi

# ============================================================================
# 6. Build workspace
# ============================================================================
echo ""
echo "[6/8] Building workspace with colcon..."
cd "${WS_DIR}"
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# ============================================================================
# 7. Source workspace
# ============================================================================
echo ""
echo "[7/8] Sourcing workspace..."
source "${WS_DIR}/install/setup.bash"

# ============================================================================
# 8. Update ~/.bashrc (idempotent)
# ============================================================================
echo ""
echo "[8/8] Updating ~/.bashrc..."

BASHRC="${HOME}/.bashrc"
SOURCE_LINE="source ${WS_DIR}/install/setup.bash"

if ! grep -qF "${SOURCE_LINE}" "${BASHRC}" 2>/dev/null; then
    echo "" >> "${BASHRC}"
    echo "# ROS2 creating_autonomous_car workspace" >> "${BASHRC}"
    echo "${SOURCE_LINE}" >> "${BASHRC}"
    echo "  Added source line to ~/.bashrc"
else
    echo "  Source line already in ~/.bashrc, skipping."
fi

# Increase socket buffer limits for CycloneDDS (idempotent)
if ! grep -q "net.core.rmem_max=26214400" /etc/sysctl.conf 2>/dev/null; then
    echo "net.core.rmem_max=26214400" | sudo tee -a /etc/sysctl.conf
    echo "net.core.wmem_max=26214400" | sudo tee -a /etc/sysctl.conf
    sudo sysctl -p
    echo "  Added socket buffer settings to /etc/sysctl.conf"
else
    echo "  Socket buffer settings already configured, skipping."
fi

# Add RMW_IMPLEMENTATION (idempotent)
if ! grep -qF "RMW_IMPLEMENTATION" "${BASHRC}" 2>/dev/null; then
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> "${BASHRC}"
    echo "  Added RMW_IMPLEMENTATION to ~/.bashrc"
else
    echo "  RMW_IMPLEMENTATION already in ~/.bashrc, skipping."
fi

# Add CYCLONEDDS_URI (commented out by default - uncomment and edit cyclonedds.xml to use)
if ! grep -qF "CYCLONEDDS_URI" "${BASHRC}" 2>/dev/null; then
    echo "# export CYCLONEDDS_URI=file://${SCRIPT_DIR}/cyclonedds.xml" >> "${BASHRC}"
    echo "  Added CYCLONEDDS_URI (commented) to ~/.bashrc"
else
    echo "  CYCLONEDDS_URI already in ~/.bashrc, skipping."
fi

ALIASES=(
    "alias cb='cd ${WS_DIR} && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'"
    "alias sauce='source ${WS_DIR}/install/setup.bash'"
    "alias sb='source ~/.bashrc'"
    "alias gb='gedit ~/.bashrc'"
)
ALIAS_KEYS=("alias cb=" "alias sauce=" "alias sb=" "alias gb=")

for i in "${!ALIASES[@]}"; do
    if ! grep -qF "${ALIAS_KEYS[$i]}" "${BASHRC}" 2>/dev/null; then
        echo "${ALIASES[$i]}" >> "${BASHRC}"
        echo "  Added '${ALIAS_KEYS[$i]}' to ~/.bashrc"
    fi
done

echo ""
echo "============================================"
echo " Build complete! (F1TENTH CAR)"
echo " Run 'source ~/.bashrc' or open a new terminal."
echo "============================================"
