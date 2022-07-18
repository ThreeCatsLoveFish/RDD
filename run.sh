#!/usr/bin/env bash
set -e
set -x

python3 -m torch.distributed.launch --nproc_per_node=2 train.py -c configs/ffpp_x3d_full_newdet.yaml
