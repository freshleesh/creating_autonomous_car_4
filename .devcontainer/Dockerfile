FROM osrf/ros:jazzy-desktop
ARG USERNAME
ARG UID
ARG GID
ARG INPUT_GID

ENV SHELL=/bin/bash

# Create user
RUN userdel ubuntu 2>/dev/null || true
RUN groupadd --gid $GID $USERNAME \
    && useradd --uid $UID --gid $GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && { groupadd --gid $INPUT_GID input 2>/dev/null || true; } \
    && usermod -aG input $USERNAME

# Base build tools + display libraries
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential cmake git python3-pip \
    libgl1-mesa-dev mesa-utils libglu1-mesa-dev \
    ros-jazzy-rmw-cyclonedds-cpp \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# rosdep: copy source to resolve all ROS dependencies from package.xml files
COPY --chown=root:root . /tmp/src/creating_autonomous_car/
RUN apt-get update \
    && rosdep update \
    && PIP_BREAK_SYSTEM_PACKAGES=1 rosdep install \
        --from-paths /tmp/src \
        --ignore-src \
        --rosdistro jazzy \
        -r -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/src

# Python packages not available in rosdep
ADD .install_utils/pip_req.txt /tmp/.install_utils/pip_req.txt
RUN pip3 install -r /tmp/.install_utils/pip_req.txt --break-system-packages \
    && rm -rf /tmp/.install_utils

USER $USERNAME

# .bashrc setup
RUN echo "source /opt/ros/jazzy/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "export ROS_DOMAIN_ID=1" >> /home/$USERNAME/.bashrc \
    && echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /home/$USERNAME/.bashrc \
    && echo "alias cb='colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> /home/$USERNAME/.bashrc \
    && echo "alias sauce='source /ws/install/setup.bash'" >> /home/$USERNAME/.bashrc \
    && echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> /home/$USERNAME/.bashrc

CMD ["/bin/bash"]
