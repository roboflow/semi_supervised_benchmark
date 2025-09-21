from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss, TaskAlignedAssigner, BboxLoss
import roboflow
import os
import shutil
import json
import random
random.seed(37)
import fire
import torch
import torch.nn.functional as F

from stac import proper_val, compute_pycocotools_metrics, fix_gt_annotation_ids


class Federatedv8DetectionLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""
    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        # self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bce = self.sigmoid_cross_entropy_loss
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    # Implementation from https://github.com/xingyizhou/CenterNet2/blob/master/projects/CenterNet2/centernet/modeling/roi_heads/custom_fast_rcnn.py#L113  # noqa
    # with slight modifications
    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes):
        """
        Args:
            pred_class_logits: shape (N, K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
        """
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits,
            gt_classes,
            reduction="none"
        )

        # since the dataset is federated, if a class is NOT annotated in an image, it is NOT necessary not present
        # so we only compute loss for annotated classes, and ignore the loss for unannotated classes
        mask = torch.any(gt_classes > 0, dim=1, keepdim=True)

        cls_loss = cls_loss * mask

        return cls_loss



class FederatedDetectionModel(DetectionModel):
    def init_criterion(self):
        return Federatedv8DetectionLoss(self)


class FederatedDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = FederatedDetectionModel(cfg, nc=self.data["nc"], ch=3, verbose=verbose)
        if weights:
            model.load(weights)
        return model

class FederatedYOLO(YOLO):
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        task_map = super().task_map
        task_map["detect"]["model"] = FederatedDetectionModel
        task_map["detect"]["trainer"] = FederatedDetectionTrainer
        return task_map


def run_federated_benchmark(dataset_url: str, force_rerun: bool=False, model_name: str='yolov8n', max_det: int=500):
    train_params = dict(
        epochs=100,
        batch=16,
    )  # standardized training params for rf100 benchmarking

    # example url:
    # dataset_url = "https://universe.roboflow.com/brad-dwyer/aquarium-combined/dataset/6"

    print("Downloading labeled dataset...")
    labeled_dataset = roboflow.download_dataset(dataset_url, "yolov8")
    fully_supervised_dataset_yaml = os.path.join(labeled_dataset.location, "data.yaml")

    experiment_name = f"{labeled_dataset.name}v{labeled_dataset.version}-{model_name}-federated"
    base_dir = os.path.join(os.path.dirname(__file__), experiment_name)

    results_json_path = os.path.join(base_dir, "results.json")
    if os.path.exists(results_json_path) and not force_rerun:
        print(f"Found existing results.json at {results_json_path}")
        print("Exiting...")
        return

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)


    print("Running fully supervised baseline...")
    model = FederatedYOLO(f"{model_name}.pt")
    model.train(
        data=fully_supervised_dataset_yaml,
        project=experiment_name,
        name="federated",
        exist_ok=True,
        **train_params
    )

    proper_val(model, split="test", max_det=max_det)

    coco_format_dataset = roboflow.download_dataset(dataset_url, "coco", location=labeled_dataset.location + "_coco")
    coco_format_test_annotations = os.path.join(coco_format_dataset.location, "test", "_annotations.coco.json")
    fix_gt_annotation_ids(coco_format_test_annotations)

    federated_test_metrics = compute_pycocotools_metrics(coco_format_test_annotations, os.path.join(experiment_name, "federated", "predictions.json"), max_det)

    fully_supervised_test_map = federated_test_metrics[0]
    fully_supervised_test_map_50 = federated_test_metrics[1]

    results_dict = {
        "fully_supervised_ap": fully_supervised_test_map,
        "fully_supervised_ap_50": fully_supervised_test_map_50,
        "url": dataset_url,
    }
    
    print(results_dict)

    with open(results_json_path, "w") as f:
        json.dump(results_dict, f)


if __name__ == "__main__":
    fire.Fire(run_federated_benchmark)
