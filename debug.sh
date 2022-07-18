#!/usr/bin/env bash
set -e
set -x

for cfg in "$@"; do
    python3 -m torch.distributed.launch --nproc_per_node=1 train.py -c $cfg
done
