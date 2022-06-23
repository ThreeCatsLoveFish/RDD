set -e

# Manipulation method. One of ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'].
METHOD=Deepfakes
# video compression rate. One of ['c23', 'c40'].
COMPRESSION=c40

python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 train.py -c configs/ffpp_2stream.yaml --method ${METHOD} --compression ${COMPRESSION}