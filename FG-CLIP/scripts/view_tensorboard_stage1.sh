#!/bin/bash

# TensorBoard for Stage 1 global-only training
LOGDIR="./checkpoints/fgclip_stage1_global/tensorboard"
PORT=6006
echo "Opening TensorBoard for Stage 1 (global-only) at port ${PORT}"
tensorboard --logdir "${LOGDIR}" --port ${PORT}
