for item in mustard_translation_x_gt_velocity_1m_s; do
    if [ -d "/home/user/iros20-6d-pose-tracking/datasets/$item/output/" ]; then
       rm /home/user/iros20-6d-pose-tracking/datasets/$item/output/*
    fi
    python /home/user/iros20-6d-pose-tracking/predict_custom.py --sequence_path /home/user/iros20-6d-pose-tracking/datasets/$item
    cd /home/user/iros20-6d-pose-tracking/datasets/$item/output
    ffmpeg -framerate 60.0 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p "./${item}_rendering.mp4"
    cd /home/user/iros20-6d-pose-tracking/datasets/
    zip -r /home/user/iros20-6d-pose-tracking/results/mustard_translation_x_gt_velocity_1m_s_se3_results.zip $item/output/$item*
    cd -
done
