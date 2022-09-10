for object in <objects> ; do
    cd /home/user/iros20-6d-pose-tracking/datasets/$object
    for item in `ls`; do
        echo "******************************************************************"
        echo $item
        echo "******************************************************************"
        cd /home/user/iros20-6d-pose-tracking
        if [ -d "/home/user/iros20-6d-pose-tracking/datasets/$object/$item/output/" ]; then
           rm -r /home/user/iros20-6d-pose-tracking/datasets/$object/$item/output
        fi
        python /home/user/iros20-6d-pose-tracking/predict_custom.py --sequence_type real --sequence_path /home/user/iros20-6d-pose-tracking/datasets/$object/$item
        cd /home/user/iros20-6d-pose-tracking/datasets/$object/$item/output
        ffmpeg -framerate 60.0 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p "./${item}_rendering.mp4"
        cd /home/user/iros20-6d-pose-tracking/datasets/
        if [ -f /home/user/iros20-6d-pose-tracking/results/${item}_se3_results.zip ]; then
            rm /home/user/iros20-6d-pose-tracking/results/${item}_se3_results.zip
        fi
        zip -r /home/user/iros20-6d-pose-tracking/results/${item}_se3_results.zip $object/$item/output/$item*
    done
done
