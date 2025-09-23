# some boilerplate code because ultralytics messes up the environment

import torch
assert torch.cuda.is_available()

import os
os.environ["YOLO_SKIP_UPDATE"] = "1"

from ultralytics import YOLO
import roboflow
import json

from sab.models.utils import ArtifactBenchmarkRequest, run_benchmark_on_artifact
from sab.models.benchmark_yolov8 import YOLOv8ONNXInference, YOLOv8TRTInference
from sab.models.benchmark_yolov11 import YOLOv11ONNXInference


def run_benchmark(dataset_name: str, dataset_base_dir: str, models_base_dir: str, model_name: str='yolov8n', max_dets: int=500, buffer_time: int=0.2):
    # dirs are named differently in the models base dir so we need to search to get the matching one
    for model_dir in os.listdir(models_base_dir):
        if model_dir.startswith(dataset_name) and model_name in model_dir:
            model_dir = os.path.join(models_base_dir, model_dir)
            break

    pt_location = os.path.join(model_dir, "supervised_reference", "weights", "best.pt")

    model = YOLO(pt_location)

    model.export(format="onnx", nms=True, conf=0.01)

    onnx_location = os.path.join(model_dir, "supervised_reference", "weights", "best.onnx")

    print(f"ONNX model exported to {onnx_location}")

    benchmark_request = ArtifactBenchmarkRequest(
        onnx_path=onnx_location,
        inference_class=YOLOv8TRTInference,
        needs_class_remapping=True,
        needs_fp16=True,
        max_dets=max_dets,
        buffer_time=buffer_time
    )

    images_dir = os.path.join(dataset_base_dir, dataset_name, "test")
    annotations_path = os.path.join(images_dir, "_annotations.coco.json")

    results = run_benchmark_on_artifact(benchmark_request, images_dir, annotations_path)

    print(results)

    results_json_path = os.path.join(model_dir, "sab_results.json")

    with open(results_json_path, "w") as f:
        json.dump(results, f)
    
    print(f"Results saved to {results_json_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(run_benchmark)