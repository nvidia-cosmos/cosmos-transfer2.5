#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

set -e
# Always change to i4 root regardless of current directory
GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "")
if [ -n "$GIT_ROOT" ]; then
    cd "$GIT_ROOT"
    echo "Changed directory to $GIT_ROOT"
else
    echo "Error: Not in a git repository"
    exit 1
fi

cd packages/cosmos-gradio

# Setup: install dependencies, activate venv
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/nfs/common/gradio_endpoints/uv_cache}"
export UV_LINK_MODE=copy

export WORKSPACE_DIR=${WORKSPACE_DIR:-/mnt/nfs/common/cosmos-eval}
export GRADIO_SAVE_DIR=${GRADIO_SAVE_DIR:-${WORKSPACE_DIR}/}
export LOG_FILE=${LOG_FILE:-${WORKSPACE_DIR}/file_server/$(date +%Y%m%d_%H%M%S).txt}
export GRADIO_ENABLE_UPLOAD=${GRADIO_ENABLE_UPLOAD:-0}

mkdir -p "$GRADIO_SAVE_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Starting Gradio file server: GRADIO_SAVE_DIR=$GRADIO_SAVE_DIR"
uv run python -m cosmos_gradio.gradio_app.gradio_file_server 2>&1 | tee -a "$LOG_FILE"
