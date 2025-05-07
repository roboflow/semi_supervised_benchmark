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
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from stac import proper_val, fix_pred_annotation_image_ids, summarize


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

    # def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
    #     """
    #     Args:
    #         gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
    #         num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
    #         Will sample negative classes if number of unique gt_classes is smaller than this value.
    #         num_classes: number of foreground classes
    #         weight: probabilities used to sample negative classes

    #     Returns:
    #         Tensor:
    #             classes to keep when calculating the federated loss, including both unique gt
    #             classes and sampled negative classes.
    #     """
    #     unique_gt_classes = torch.unique(gt_classes)
    #     prob = unique_gt_classes.new_ones(num_classes + 1).float()
    #     prob[-1] = 0
    #     if len(unique_gt_classes) < num_fed_loss_classes:
    #         prob[:num_classes] = weight.float().clone()
    #         prob[unique_gt_classes] = 0
    #         sampled_negative_classes = torch.multinomial(
    #             prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
    #         )
    #         fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
    #     else:
    #         fed_loss_classes = unique_gt_classes
    #     return fed_loss_classes

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

        # print(f"pred_class_logits: {pred_class_logits.shape}")
        # print(f"gt_classes: {gt_classes.shape}")

        # N = pred_class_logits.shape[0]
        # K = pred_class_logits.shape[1] - 1

        # target = pred_class_logits.new_zeros(N, K + 1)
        # target[range(len(gt_classes)), gt_classes.long()] = 1
        # target = target[:, :K]

        # cls_loss = F.binary_cross_entropy_with_logits(
        #     pred_class_logits[:, :-1], target, reduction="none"
        # )
        # if self.use_fed_loss:
        #     fed_loss_classes = self.get_fed_loss_classes(
        #         gt_classes,
        #         num_fed_loss_classes=self.fed_loss_num_classes,
        #         num_classes=K,
        #         weight=self.fed_loss_cls_weights,
        #     )
        #     fed_loss_classes_mask = fed_loss_classes.new_zeros(K + 1)
        #     fed_loss_classes_mask[fed_loss_classes] = 1     # fed_loss_classes contain GT+sampled negative classes as 1.
        #     fed_loss_classes_mask = fed_loss_classes_mask[:K]             # get rid of bg class
        #     weight = fed_loss_classes_mask.view(1, K).expand(N, K).float()           # modify shape to then multiplt 
        # else:
        #     weight = 1

        # # # cls_loss is NxK, where N is number of predicted boxes and K are the number of object categories.
        # # loss = torch.sum(cls_loss * weight) / N     # mask out the classification loss for classes corresponding to sampled subset of classes for federated loss.
        # # return loss
        # return cls_loss * weight

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits,
            gt_classes,
            reduction="none"
        )

        # mask = torch.any(gt_classes > 0, dim=1, keepdim=True)

        # cls_loss = cls_loss * mask

        return cls_loss



class FederatedDetectionModel(DetectionModel):
    def init_criterion(self):
        return Federatedv8DetectionLoss(self)


class FederatedDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = FederatedDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
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

    test_gt_annotations = COCO(coco_format_test_annotations)
    fix_pred_annotation_image_ids(os.path.join(experiment_name, "federated", "predictions.json"), coco_format_test_annotations)
    test_pred_annotations = test_gt_annotations.loadRes(os.path.join(experiment_name, "federated", "predictions.json"))
    coco_eval = COCOeval(test_gt_annotations, test_pred_annotations, "bbox")
    coco_eval.params.maxDets = [1, 10, max_det]
    coco_eval.evaluate()
    coco_eval.accumulate()
    summarize(coco_eval)

    fully_supervised_test_map = coco_eval.stats[0]
    fully_supervised_test_map_50 = coco_eval.stats[1]

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
