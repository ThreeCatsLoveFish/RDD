#!/usr/bin/env bash
set -e

# Manipulation method. One of ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'].
# METHOD=Deepfakes
# video compression rate. One of ['c23', 'c40'].
COMPRESSION=c40

# "NONE" "TF HF" "TF VF HF" "TF HF CJHSV" "TF VF HF CJHSV" "TF VF HF RS" "TF HF RS" "TF VF HF CJHSV RS" "RS" "RS CJHSV"
for cfg in "$@"; do
    for METHOD in Deepfakes; do
        for AUG in "TF VF HF CJHSV RS"; do
            echo ${METHOD} ${AUG}
            python3 -W ignore -m torch.distributed.launch --nproc_per_node=1 train.py -c $cfg --method ${METHOD} --compression ${COMPRESSION} --augmentation ${AUG} 
        done
    done
done