# CUDA_VISIBLE_DEVICES=xxx sh scripts/script.sh configs/ffpp_x3d_sbis_3v1.yaml configs/ffpp_x3d_3v1.yaml configs/ffpp_x3d_sbis.yaml
COMPRESSION=c40

for cfg in "$@"; do
    for METHOD in Deepfakes Face2Face FaceSwap NeuralTextures; do
        exam_dir=./exps/$(echo $cfg | cut -d "/" -f 2 | cut -d '.' -f 1)/${METHOD}_${COMPRESSION}
        python3 -m torch.distributed.launch --nproc_per_node=8 train.py -c $cfg --method ${METHOD} --compression ${COMPRESSION} --exam_dir ${exam_dir}
        python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 test.py -c $cfg --method ${METHOD} --compression ${COMPRESSION} --model.resume ${exam_dir}/ckpt/best.pth >> ${exam_dir}/test.log
    done
done
