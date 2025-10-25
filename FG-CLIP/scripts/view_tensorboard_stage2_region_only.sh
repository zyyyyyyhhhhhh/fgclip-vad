#!/bin/bash

# TensorBoard viewer for Stage 2 Region-only training
LOGDIR="./checkpoints/fgclip_stage2_region_only/tensorboard"
PORT=6008

echo "Opening TensorBoard for Stage 2 (Region-only) at port ${PORT}"
tensorboard --logdir "${LOGDIR}" --port ${PORT}
