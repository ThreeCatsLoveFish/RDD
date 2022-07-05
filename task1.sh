set -e

echo "CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12346 train.py -c configs/ffpp_stil_1out.yaml --method Deepfakes --compression c40"
CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12346 train.py -c configs/ffpp_stil_1out.yaml --method Deepfakes --compression c40

echo "CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12346 train.py -c configs/ffpp_stil_1out.yaml --method Face2Face --compression c40"
CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12346 train.py -c configs/ffpp_stil_1out.yaml --method Face2Face --compression c40

echo "CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12346 train.py -c configs/ffpp_x3d_1out.yaml --method Deepfakes --compression c40"
CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12346 train.py -c configs/ffpp_x3d_1out.yaml --method Deepfakes --compression c40

echo "CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12346 train.py -c configs/ffpp_x3d_1out.yaml --method Face2Face --compression c40"
CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12346 train.py -c configs/ffpp_x3d_1out.yaml --method Face2Face --compression c40

echo "CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12346 train.py -c configs/ffpp_2stream_1out.yaml --method Deepfakes --compression c40"
CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12346 train.py -c configs/ffpp_2stream_1out.yaml --method Deepfakes --compression c40

echo "CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12346 train.py -c configs/ffpp_2stream_1out.yaml --method Face2Face --compression c40"
CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12346 train.py -c configs/ffpp_2stream_1out.yaml --method Face2Face --compression c40
