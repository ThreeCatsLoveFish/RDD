set -e
set -x

CUDA_VISIBLE_DEVICES=4,5 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12347 train.py -c configs/ffpp_stil_1out.yaml --method FaceSwap --compression c40

CUDA_VISIBLE_DEVICES=4,5 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12347 train.py -c configs/ffpp_stil_1out.yaml --method NeuralTextures --compression c40

CUDA_VISIBLE_DEVICES=4,5 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12347 train.py -c configs/ffpp_x3d_1out.yaml --method FaceSwap --compression c40

CUDA_VISIBLE_DEVICES=4,5 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12347 train.py -c configs/ffpp_x3d_1out.yaml --method NeuralTextures --compression c40

CUDA_VISIBLE_DEVICES=4,5 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12347 train.py -c configs/ffpp_2stream_1out.yaml --method FaceSwap --compression c40

CUDA_VISIBLE_DEVICES=4,5 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12347 train.py -c configs/ffpp_2stream_1out.yaml --method NeuralTextures --compression c40
