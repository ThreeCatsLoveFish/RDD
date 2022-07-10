#!/usr/bin/env bash
set -e
set -x

# Manipulation method. One of ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'].
METHOD=NeuralTextures
# video compression rate. One of ['c23', 'c40'].
COMPRESSION=c40

python3 -m torch.distributed.launch --nproc_per_node=1 train.py -c configs/ffpp_x3d_base.yaml --method ${METHOD} --compression ${COMPRESSION}
