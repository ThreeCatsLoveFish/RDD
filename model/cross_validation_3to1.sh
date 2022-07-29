set -e

config_name=ffpp_2stream
model_name=X3D_2Stream
compression=c40

for m1 in Deepfakes Face2Face FaceSwap NeuralTextures; do
    mkdir -p exps/${config_name}_1out/${model_name}_${m1}_out_${compression}/ffpp_eval/
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 test.py -c configs/${config_name}.yaml --method ${m1} --compression ${compression} --model.resume exps/${config_name}_1out/${model_name}_${m1}_out_${compression}/ckpt/best.pth > exps/${config_name}_1out/${model_name}_${m1}_out_${compression}/ffpp_eval/${m1}_out_on_${m1}.log
done