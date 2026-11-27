#!/usr/bin/env bash
set -e
cd "$HOME/scene_understanding"
exec .venv/bin/python direct_yolo.py "$@"
