#!/usr/bin/env bash
# Install PointNeXt with uv
# Command: source install.sh

# Git submodule setup
git submodule update --init --recursive

# Sync all dependencies including C++ extensions
uv sync