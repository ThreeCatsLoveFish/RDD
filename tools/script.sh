RLAUNCH_REPLICA_TOTAL=$1
for RLAUNCH_REPLICA in $(seq 0 $(($1 - 1))); do
    python tools/pre_crop_videos.py --video --moviepy &
done
