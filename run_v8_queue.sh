#!/bin/bash
# Smart GPU queue for YOLOv8 - jobs start as soon as VRAM is available
python gpu_queue.py --model=yolov8 --max-vram=75 --fail-log=8_errors.txt
