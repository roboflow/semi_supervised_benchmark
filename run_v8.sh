#!/bin/bash
FAIL_LOG="8_errors.txt"
> "$FAIL_LOG"

run_and_log() {
    "$@"
    if [ $? -ne 0 ]; then
        echo "FAILED: $*" >> "$FAIL_LOG"
    fi
}

run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8n --skip_stac=True --batch=1 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8n --skip_stac=True --batch=2 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8m --skip_stac=True --batch=128 &
wait

run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8s --skip_stac=True --batch=1 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8s --skip_stac=True --batch=2 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8m --skip_stac=True --batch=64 &
wait

run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8m --skip_stac=True --batch=1 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8m --skip_stac=True --batch=16 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8m --skip_stac=True --batch=32 &
wait

run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8m --skip_stac=True --batch=2 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8n --skip_stac=True --batch=4 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8s --skip_stac=True --batch=128 &
wait

run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8m --skip_stac=True --batch=4 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8s --skip_stac=True --batch=4 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8n --skip_stac=True --batch=128 &
wait

run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8s --skip_stac=True --batch=32 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8n --skip_stac=True --batch=32 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8s --skip_stac=True --batch=8 &
wait

run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8m --skip_stac=True --batch=8 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8s --skip_stac=True --batch=16 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8n --skip_stac=True --batch=64 &
wait

run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8n --skip_stac=True --batch=16 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8s --skip_stac=True --batch=64 &
run_and_log python dispatcher.py stac.py url_list.txt --model_name=yolov8n --skip_stac=True --batch=8 &
wait