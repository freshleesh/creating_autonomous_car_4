# Creating Autonomous Car - ROS2 Workspace

ROS2 Jazzy workspace for autonomous vehicle development.

## Features

- **VESC**: VESC-based Ackermann steering control
- **2D LiDAR**: Hokuyo LiDAR and pre-built IMU sensor
- **SLAM**: Cartographer-based 2D SLAM mapping and localization
- **Particle Filter**: Alternative 2D LiDAR-based localization
- **Simulator**: F1TENTH gym-based simulator with obstacle support (static + dynamic)
- **Planner**: Centerline extraction from map + trajectory optimization (student implementation)

## Prerequisites

- Ubuntu 24.04
- ROS2 Jazzy (local install) **or** Docker Engine (container-based install)

---

# INSTALLATION

## 1. Clone Repository

```bash
sudo apt update && sudo apt install -y git
mkdir -p ~/creating_autonomous_car_ws/src
cd ~/creating_autonomous_car_ws/src
git clone https://github.com/HMCL-UNIST/creating_autonomous_car.git
cd creating_autonomous_car
```

> ⚠️ **Complete only the section that applies to you.**

<details open>
<summary><big><big><b>2-A. For Students — Local PC, Simulation Only (Recommended)</b></big></big></summary>

<br>

```bash
~/creating_autonomous_car_ws/src/creating_autonomous_car/build_packages_on_local_pc.sh
```

After the script finishes, run `source ~/.bashrc` or open a new terminal.

Registered aliases in `~/.bashrc`:

| Alias | Description |
|-------|-------------|
| `cb` | Build workspace (`colcon build --symlink-install`) |
| `sauce` | Source workspace |
| `sb` | Source `~/.bashrc` |
| `gb` | Edit `~/.bashrc` with gedit |

</details>

<details>
<summary><big><big><b>2-B. For TAs — Car, Full Build</b></big></big></summary>

<br>

```bash
~/creating_autonomous_car_ws/src/creating_autonomous_car/build_packages_on_car.sh
```

After the script finishes, run `source ~/.bashrc` or open a new terminal.

Registered aliases in `~/.bashrc`:

| Alias | Description |
|-------|-------------|
| `cb` | Build workspace (`colcon build --symlink-install`) |
| `sauce` | Source workspace |
| `sb` | Source `~/.bashrc` |
| `gb` | Edit `~/.bashrc` with gedit |

</details>

<details>
<summary><big><big><b>2-C. Optional — Manual Install, Step by Step (Simulation Only)</b></big></big></summary>

<br>

```bash
cd ~/creating_autonomous_car_ws

# Install base tools
sudo apt update
sudo apt install -y python3-rosdep python3-pip python3-skimage ros-jazzy-xacro

# Initialize rosdep (first time only)
sudo rosdep init
rosdep update

# Install ROS package dependencies
rosdep install --from-paths src --ignore-src -r -y

# Install f1tenth_gym simulator (editable mode)
pip install -e ~/creating_autonomous_car_ws/src/creating_autonomous_car/simulator/f1tenth_gym --break-system-packages

# Install transforms3d (required by simulator bridge)
pip install transforms3d --break-system-packages

# Fix numba + coverage conflict
pip install --upgrade coverage --break-system-packages

# Build
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source workspace
source ~/creating_autonomous_car_ws/install/setup.bash
echo "source ~/creating_autonomous_car_ws/install/setup.bash" >> ~/.bashrc
```

**(Optional)** Add useful aliases to `~/.bashrc`:

```bash
echo "alias cb='cd ~/creating_autonomous_car_ws && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> ~/.bashrc
echo "alias sauce='source ~/creating_autonomous_car_ws/install/setup.bash'" >> ~/.bashrc
echo "alias sb='source ~/.bashrc'" >> ~/.bashrc
echo "alias gb='gedit ~/.bashrc'" >> ~/.bashrc
```

> **Note:** By default, car-only packages (sensor drivers, SLAM, particle filter) are excluded via `COLCON_IGNORE`.
> To build all packages on the car, run `build_packages_on_car.sh` or manually remove the `COLCON_IGNORE` files from `sensor/`, `slam/` directories.

</details>

<details>
<summary><big><big><b>2-D. Not Recommended — Docker-based Setup (Only if Ubuntu 24.04 is not available)</b></big></big></summary>

<br>

This workflow builds and runs the workspace inside a Docker container.
No local ROS2 installation is needed — only Docker Engine and VS Code.

### Step 1 — Install Docker Engine

```bash
# Add Docker's official GPG key
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the Docker repository
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
```

### Step 2 — Add your user to the `docker` group

```bash
sudo usermod -aG docker $USER
newgrp docker
```

> **Note:** `newgrp docker` applies the group change to the current shell only.
> For a permanent effect across all terminals, **log out and log back in** (or reboot).

### Step 3 — Verify Docker is working

```bash
docker ps
```

Expected output: an empty table with column headers (no permission errors).

### Step 4 — Build the Docker image and open VS Code

Run the build script from anywhere:

```bash
~/creating_autonomous_car_ws/src/creating_autonomous_car/build_docker.sh
```

This script will:
1. Create the `colcon` cache directories (`cache/build`, `cache/install`, `cache/log`)
2. Move into the repository directory
3. Build the `creating_autonomous_car_ros2:jazzy` image (defined in `.devcontainer/Dockerfile`)
4. Open VS Code in the repository directory

### Step 5 — Open in VS Code Dev Container

1. Install the **Dev Containers** extension in VS Code (if not already installed).
2. Press `Ctrl+Shift+P`, type **"Dev Containers: Reopen in Container"**, and select it.

VS Code will start the container using the pre-built image, mount the cache
directories, and automatically run `build_packages_on_local_pc.sh`
(via `postCreateCommand` in `.devcontainer/devcontainer.json`).

</details>

---

# USAGE -- SIMULATOR

## Launch Simulator

```bash
ros2 launch stack_master low_level.launch.xml sim:=true map:=<map_name>
```

### 💡 No custom map? Use the default map `f` in the `maps` folder:

```bash
ros2 launch stack_master low_level.launch.xml map:=f sim:=true
```

This launches:
- F1TENTH gym simulator (`gym_bridge`)
- Static obstacle manager (interactive markers in RViz)
- Simple command multiplexer
- LiDAR is configured to 2160 beams to match the class vehicle setup

## Keyboard Control (Simulator)

To manually drive the car in simulation, use `keyboard_joy_node`.
It publishes to the `/joy` topic at 50 Hz — the same interface as a physical joystick.

```bash
ros2 run stack_master keyboard_joy_node.py
```

Key bindings:

| Key | Action |
|-----|--------|
| `↑` | Forward |
| `↓` | Reverse |
| `←` | Steer left |
| `→` | Steer right |
| `Space` | Stop |
| `H` | Human drive mode |
| `A` | Auto drive mode |
| `Q` / `Ctrl+C` | Quit |

> **Note:** Requires `pynput`. Installed automatically by `build_packages_on_local_pc.sh` and `build_packages_on_car.sh`.

## Static Obstacles

- **Place**: Click in RViz using the **"Publish Point"** tool to add a static obstacle at that location
- **Clear all**: Left-click the green **"Clear Obstacles"** button in RViz (located at (0, -5) in map frame)

## Dynamic Obstacle

Launch a dynamic obstacle:

```bash
ros2 launch obstacle_publisher dynamic_obstacle_publisher.launch.xml map:=<map_name>
```

By default, if no trajectory CSV exists for the given map, a **circular dummy trajectory** (radius 5m) is used automatically. To make the obstacle follow a specific path, provide a trajectory CSV via launch argument or extract a centerline first (see below).

Options:
- `trajectory_csv` - CSV file to follow (default: `centerline.csv`)
- `speed_scaler` - Speed multiplier (default: `0.7`)
- `constant_speed` - Use constant speed (default: `false`)
- `reactive` - Enable lateral oscillation mode (default: `false`)
- `reactive_freq` - Oscillation frequency in Hz (default: `0.3`)
- `starting_s` - Starting position on trajectory in meters (default: `0.0`)
- `obstacle_length_m` - Obstacle length (default: `0.6`)
- `obstacle_width_m` - Obstacle width (default: `0.3`)

## Centerline Extraction & Trajectory Optimization

Extract centerline from a map image. The extracted centerline can be used as a reference trajectory for planning or as a dynamic obstacle path.

```bash
ros2 launch planner create_path.launch.xml map:=<map_name>
```

This will:
1. Extract track centerline via skeletonization
2. Save `centerline.csv` to `stack_master/maps/<map_name>/`
3. Show matplotlib visualization (skeleton, centerline, track bounds)

The default extraction direction is **counter-clockwise (CCW)**. To change to clockwise:

```bash
ros2 launch planner create_path.launch.xml map:=<map_name> reverse:=true
```

Other options:
- `output_csv` - Output filename (default: `centerline.csv`)
- `optimize` - Run trajectory optimization (default: `false`)

Output CSV format: `x_m, y_m, w_tr_right_m, w_tr_left_m`

Additionally, `boundary_right.csv` and `boundary_left.csv` (wall contour points) are saved for accurate wall-distance computation.

## Waypoint CSV Standard Format

All waypoint CSVs (centerline, global_waypoints, etc.) follow this column order:

| Column | Name | Description | Required |
|--------|------|-------------|----------|
| 1 | `x_m` | X coordinate [m] | Yes |
| 2 | `y_m` | Y coordinate [m] | Yes |
| 3 | `w_tr_right_m` | Right track width [m] | No |
| 4 | `w_tr_left_m` | Left track width [m] | No |
| 5 | `psi_rad` | Heading [rad] | No |
| 6 | `kappa_radpm` | Curvature [rad/m] | No |
| 7 | `vx_mps` | Velocity [m/s] | No |

Missing columns are handled automatically by `waypoint_publisher`:
- `psi_rad` -> computed from atan2(dy, dx)
- `kappa_radpm` -> 0.0
- `vx_mps` -> 0.0
- `d_right/d_left` -> computed from boundary CSVs if available, otherwise from `w_tr` columns

---

# USAGE -- CAR ONLY

The following sections apply only when running on the physical F1TENTH car.

## Launch Sensors

```bash
ros2 launch stack_master low_level.launch.xml
```

This launches:
- URG LiDAR node
- VESC motor driver
- Ackermann to VESC converter
- VESC to odometry converter
- Simple command multiplexer

## Run Joy Node

Run the joy node at the appropriate location (e.g., onboard, remote laptop).
Depending on the connection method (bluetooth, wired, wireless receiver), the joystick key mapping may vary.

```bash
ros2 run joy joy_node
```

## Cartographer Mapping

Start mapping to create a new map:

```bash
ros2 launch stack_master middle_level.launch.xml map:=<map_name> mapping:=true
```

**Save the map**:
1. Open RViz
2. Click **"2D Goal Pose"** button
3. Click anywhere on the map
4. Map will be saved to `stack_master/maps/<map_name>/`

Files saved:
- `<map_name>.pbstream` - Cartographer state file
- `<map_name>.png` - Occupancy grid image
- `<map_name>.yaml` - Map metadata

## Cartographer Localization

Use an existing map for localization:

```bash
ros2 launch stack_master middle_level.launch.xml map:=<map_name>
```

**Set initial pose**:
1. Open RViz
2. Click **"2D Pose Estimate"** button
3. Click on the map where the robot is located
4. Drag to set orientation
5. Cartographer will reinitialize localization at the specified pose

## Particle Filter Localization

```bash
ros2 launch stack_master middle_level.launch.xml map:=<map_name> localization:=pf
```

Requires `range_libc` installed (handled by `build_packages_on_car.sh`).

---

## License

Copyright 2026 HMCL-UNIST

Licensed under the Apache License, Version 2.0

## Contributors

- Jeongsang Ryu (ryujs@unist.ac.kr)
- Hyeongjoon Yang (shineejoon@unist.ac.kr)
