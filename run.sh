#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=8
cd "$HOME/scene_understanding"
exec .venv/bin/python scene_node.py "$@"
