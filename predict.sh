for obj_name in `cd objects && ls`; do
    for video_id in `cat ./objects/${obj_name} | sed -n 2p`;
    do
        CUDA_VISIBLE_DEVICES=0 python /home/hsp-user/iros20-6d-pose-tracking/predict.py --object_name ${obj_name} --video_sequence ${video_id}
    done
done
