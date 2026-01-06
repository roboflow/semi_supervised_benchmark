#!/bin/bash
# Smart GPU queue for YOLO11 - jobs start as soon as VRAM is available
python3 gpu_queue.py --model=yolo11 --max-vram=75 --fail-log=11_errors.txt
