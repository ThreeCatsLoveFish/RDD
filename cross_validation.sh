set -e

config_name=ffpp_x3d
model_name=X3D
compression=c40

for m1 in Deepfakes Face2Face FaceSwap NeuralTextures; do
    for m2 in Deepfakes Face2Face FaceSwap NeuralTextures; do
        mkdir -p exps/${config_name}/${model_name}_${m1}_${compression}/ffpp_eval/
        CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 test.py -c configs/${config_name}.yaml --method ${m2} --compression ${compression} --model.resume exps/${config_name}/${model_name}_${m1}_${compression}/ckpt/best.pth > exps/${config_name}/${model_name}_${m1}_${compression}/ffpp_eval/${m1}_on_${m2}.log
    done
done