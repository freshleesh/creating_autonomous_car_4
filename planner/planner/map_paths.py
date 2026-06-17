"""Canonical map-directory resolution for the whole stack.

Maps are *data* that change far more often than code, so the stack reads them
from the SOURCE tree (src/creating_autonomous_car/stack_master/maps) rather than
the install space. This way an edited / newly recorded map is picked up without a
`colcon build` copy step, and it removes the install↔src desync that previously
let a stale (or symlink-corrupted) install copy shadow the real map.

Resolution order:
  1. An explicit override (e.g. a ROS `maps_dir` param passed from a launch file).
  2. The source tree, derived from the package install prefix the same way the
     launch files do:  <prefix>/../../src/creating_autonomous_car/stack_master/maps
  3. Fallback to the install share dir (old behaviour) if the source tree is not
     present — e.g. a deployed robot with no checkout.

Using the install *prefix* (not `__file__`) makes this independent of whether the
package was built with `--symlink-install` or plain copy mode.
"""

import os

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)

# Repo folder name under <workspace>/src that holds stack_master. Kept in one
# place; matches the hard-coded path already used in the launch files.
_REPO_DIR = 'creating_autonomous_car'


def maps_root(override: str = '') -> str:
    """Return the directory that contains the per-map folders."""
    if override:
        return override

    try:
        prefix = get_package_prefix('stack_master')      # .../install/stack_master
        src = os.path.normpath(os.path.join(
            prefix, os.pardir, os.pardir,
            'src', _REPO_DIR, 'stack_master', 'maps'))
        if os.path.isdir(src):
            return src
    except Exception:
        pass

    # Fallback: install share (no source checkout available).
    return os.path.join(get_package_share_directory('stack_master'), 'maps')


def map_dir(map_name: str, override: str = '') -> str:
    """Return the folder for a single map, e.g. .../maps/<map_name>."""
    return os.path.join(maps_root(override), map_name)
