for item in mustard_translation_x_gt_velocity_1m_s; do
    # python /home/user/iros20-6d-pose-tracking/predict_custom.py --sequence_path /home/user/iros20-6d-pose-tracking/datasets/$item
    cd /home/user/iros20-6d-pose-tracking/datasets/
    zip -r /home/user/iros20-6d-pose-tracking/results/mustard_translation_x_gt_velocity_1m_s_se3_results.zip $item/output/*
done
