# Docker that can run ROS, Matterport3DSimulator, PyTorch, etc
# Requires nvidia gpu with driver 396.37 or higher


FROM nvidia/cudagl:9.2-devel-ubuntu16.04

# Install cudnn
ENV CUDNN_VERSION 7.6.4.38
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda9.2 \
libcudnn7-dev=$CUDNN_VERSION-1+cuda9.2 \
&& \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*


# Install a few libraries 
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y wget doxygen curl python-opencv python-setuptools python-dev python-pip
RUN pip install --upgrade pip
RUN pip install torch==1.1.0 torchvision==0.3.0
RUN pip install numpy==1.13.3 pandas==0.24.1 networkx==2.2 easydict protobuf pyyaml

# Install ROS kinetic
RUN apt-get install -y lsb-release vim
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update
RUN apt-get install -y ros-kinetic-desktop-full
RUN rosdep init
RUN rosdep update

# Install additional ROS stuff
RUN apt-get update
RUN apt-get install -y ros-kinetic-urg-node ros-kinetic-ros-numpy ros-kinetic-move-base-msgs ros-kinetic-multimaster-fkie

# Goodies for rviz
RUN apt-get install -y ros-kinetic-kobuki-description ros-kinetic-turtlebot-description ros-kinetic-hector-sensors-description

RUN pip install scipy==1.2.2
