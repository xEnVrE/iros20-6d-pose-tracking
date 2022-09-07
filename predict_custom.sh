for item in 003_cracker_box_pitch_4_rad_s 003_cracker_box_special_pitch_1_rad_s 003_cracker_box_translation_x_1_m_s 003_cracker_box_translation_z_1_m_s 003_cracker_box_roll_4_rad_s 003_cracker_box_special_yaw_1_rad_s 003_cracker_box_translation_y_1_m_s 003_cracker_box_yaw_4_rad_s; do
    echo "******************************************************************"
    echo $item
    echo "******************************************************************"
    cd /home/user/iros20-6d-pose-tracking
    if [ -d "/home/user/iros20-6d-pose-tracking/datasets/$item/output/" ]; then
       rm -r /home/user/iros20-6d-pose-tracking/datasets/$item/output
    fi
    python /home/user/iros20-6d-pose-tracking/predict_custom.py --sequence_path /home/user/iros20-6d-pose-tracking/datasets/$item
    cd /home/user/iros20-6d-pose-tracking/datasets/$item/output
    ffmpeg -framerate 60.0 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p "./${item}_rendering.mp4"
    cd /home/user/iros20-6d-pose-tracking/datasets/
    if [ -f /home/user/iros20-6d-pose-tracking/results/${item}_se3_results.zip ]; then
        rm /home/user/iros20-6d-pose-tracking/results/${item}_se3_results.zip
    fi
    zip -r /home/user/iros20-6d-pose-tracking/results/${item}_se3_results.zip $item/output/$item*
done
