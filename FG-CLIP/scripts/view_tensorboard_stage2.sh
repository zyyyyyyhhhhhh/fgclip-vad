#!/bin/bash

# TensorBoard for Stage 2 global+region training
LOGDIR="./checkpoints/fgclip_stage2_joint/tensorboard"
PORT=6007
echo "Opening TensorBoard for Stage 2 (global+region) at port ${PORT}"
tensorboard --logdir "${LOGDIR}" --port ${PORT}
