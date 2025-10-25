#!/bin/bash

# TensorBoard viewer for Stage 2 Region + Normal ROI training
LOGDIR="./checkpoints/fgclip_stage2_region_with_normal/tensorboard"
PORT=6010

echo "Opening TensorBoard for Stage 2 (Region + Normal ROI) at port ${PORT}"
tensorboard --logdir "${LOGDIR}" --port ${PORT}
