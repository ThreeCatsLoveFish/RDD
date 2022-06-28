set -e

# Manipulation method. One of ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'].
METHOD=FaceSwap
# video compression rate. One of ['c23', 'c40'].
COMPRESSION=c40
# model path
MODEL_PATH=exps/ffpp_stil/STIL_Model_FaceSwap_c40/ckpt/best.pth

python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 test.py -c configs/ffpp_stil.yaml  --method ${METHOD} --compression ${COMPRESSION} --model.resume ${MODEL_PATH}