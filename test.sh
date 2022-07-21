set -e

# Manipulation method. One of ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'].
METHOD=FaceSwap
# video compression rate. One of ['c23', 'c40'].
COMPRESSION=c40
# model path

# NONE TF_HF TF_HF_CJHSV TF_HF_RS TF_VF_HF TF_VF_HF_CJHSV TF_VF_HF_CJHSV_RS TF_VF_HF_RS;

for METHOD in Deepfakes Face2Face FaceSwap NeuralTextures; do
    for AUG in NONE RS RS_CJHSV TF_HF TF_HF_CJHSV TF_HF_RS TF_VF_HF TF_VF_HF_CJHSV TF_VF_HF_CJHSV_RS TF_VF_HF_RS  ; do
        echo $METHOD
        MODEL_PATH=exps/ffpp_x3d_3v1/X3D_${METHOD}_c40_${AUG}/ckpt/best.pth
        echo $MODEL_PATH
        python3 -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 test.py -c configs/ffpp_x3d_3v1.yaml  --method ${METHOD} --compression ${COMPRESSION} --model.resume ${MODEL_PATH}
    done
done