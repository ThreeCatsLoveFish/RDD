#!/usr/bin/env bash
set -x

# Manipulation method. One of ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'].
# METHOD=Deepfakes
# video compression rate. One of ['c23', 'c40'].
COMPRESSION=c40

for cfg in "$@"; do
    OUTPUT_PATH=$(echo $cfg | sed -e "s/^configs/exps_cj/g" | sed -e "s/.yaml$//g")
    mkdir -p $OUTPUT_PATH
    cp $cfg $OUTPUT_PATH
    ln -s /data/projects/RDD/data $OUTPUT_PATH
    cp -r common datasets models utils train.py $OUTPUT_PATH
done

for cfg in "$@"; do
    OUTPUT_PATH=$(echo $cfg | sed -e "s/^configs/exps_cj/g" | sed -e "s/.yaml$//g")
    pushd $OUTPUT_PATH
    for METHOD in Deepfakes Face2Face FaceSwap NeuralTextures; do
        python3 -m torch.distributed.launch --nproc_per_node=8 train.py -c $(basename $cfg) --method ${METHOD} --compression ${COMPRESSION}
    done
    popd
done
