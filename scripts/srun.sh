for cfg in ffpp_x3d_3v1 ffpp_x3d_sbis ffpp_x3d_sbis_3v1; do
    srun -p DI --gres=gpu:8 --cpus-per-task=32 --job-name=$cfg sh scripts/script.sh configs/$cfg.yaml &
done
