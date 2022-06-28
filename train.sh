#!/usr/bin/env bash
set -e

# Manipulation method. One of ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'].
# METHOD=Deepfakes
# video compression rate. One of ['c23', 'c40'].
COMPRESSION=c40

for cfg in "$@"; do
    for METHOD in Deepfakes Face2Face FaceSwap NeuralTextures; do
        python3 -m torch.distributed.launch --nproc_per_node=1 train.py -c $cfg --method ${METHOD} --compression ${COMPRESSION}
    done
done
