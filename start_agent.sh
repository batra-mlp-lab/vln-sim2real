#!/bin/bash

export ROS_MASTER_URI=http://128.61.117.147:11311
export ROS_HOSTNAME=143.215.114.158
source /opt/ros/kinetic/setup.bash
source devel/setup.bash


# For full setting - ros master is here

roscore -p 11312&
export ROS_MASTER_URI=http://143.215.114.158:11312
export ROS_HOSTNAME=143.215.114.158
source /opt/ros/kinetic/setup.bash
source devel/setup.bash
