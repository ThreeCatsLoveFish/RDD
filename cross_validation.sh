set -e

for m1 in Deepfakes Face2Face FaceSwap NeuralTextures; do
    for m2 in Deepfakes Face2Face FaceSwap NeuralTextures; do
        mkdir -p exps/ffpp_2stream/X3D_2Stream_${m1}_c40/ffpp_eval/
        CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 test.py -c configs/ffpp_2stream.yaml --method ${m2} --compression c40 --model.resume exps/ffpp_2stream/X3D_2Stream_${m1}_c40/ckpt/best.pth > exps/ffpp_2stream/X3D_2Stream_${m1}_c40/ffpp_eval/${m1}_on_${m2}.log
    done
done