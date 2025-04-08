from ultralytics import YOLO
import roboflow
import os
import shutil
import json
import random
random.seed(37)
import fire
import torch
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def proper_val(
    self,
    validator=None,
    **kwargs,
):
    custom = {"rect": True}  # method defaults
    args = {**self.overrides, **custom, **kwargs, "mode": "val", "save_json": True}  # highest priority args on the right

    validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
    validator(model=self.model)
    self.metrics = validator.metrics
    # return validator.metrics
    stats = validator.eval_json(validator.stats)
    
    stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in stats.items()}  # to numpy
    self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
    self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
    stats.pop("target_img", None)
    if len(stats):
        self.metrics.process(**stats, on_plot=False)
    return self.metrics


def fix_pred_annotation_image_ids(pred_annotations_path: str, gt_annotations_path: str):
    with open(pred_annotations_path, "r") as f:
        pred_annotations = json.load(f)
    
    with open(gt_annotations_path, "r") as f:
        gt_annotations = json.load(f)
    
    image_name_to_id = {image["extra"]["name"]: image["id"] for image in gt_annotations["images"]}

    for pred_annotation in pred_annotations:
        pred_annotation["image_id"] = image_name_to_id[pred_annotation["image_id"].split(".")[0]]

    with open(pred_annotations_path, "w") as f:
        json.dump(pred_annotations, f)


def summarize(self):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s
    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
        stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
        stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
        stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
        stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
        return stats
    def _summarizeKps():
        stats = np.zeros((10,))
        stats[0] = _summarize(1, maxDets=20)
        stats[1] = _summarize(1, maxDets=20, iouThr=.5)
        stats[2] = _summarize(1, maxDets=20, iouThr=.75)
        stats[3] = _summarize(1, maxDets=20, areaRng='medium')
        stats[4] = _summarize(1, maxDets=20, areaRng='large')
        stats[5] = _summarize(0, maxDets=20)
        stats[6] = _summarize(0, maxDets=20, iouThr=.5)
        stats[7] = _summarize(0, maxDets=20, iouThr=.75)
        stats[8] = _summarize(0, maxDets=20, areaRng='medium')
        stats[9] = _summarize(0, maxDets=20, areaRng='large')
        return stats
    if not self.eval:
        raise Exception('Please run accumulate() first')
    iouType = self.params.iouType
    if iouType == 'segm' or iouType == 'bbox':
        summarize = _summarizeDets
    elif iouType == 'keypoints':
        summarize = _summarizeKps
    self.stats = summarize()


def run_benchmark(dataset_url: str, label_percentage: float=0.1, force_rerun: bool=False, model_name: str='yolov8n', skip_stac: bool=False, max_det: int=500):
    train_params = dict(
        epochs=100,
        batch=16,
    )  # standardized training params for rf100 benchmarking

    # example url:
    # dataset_url = "https://universe.roboflow.com/brad-dwyer/aquarium-combined/dataset/6"

    print("Downloading labeled dataset...")
    labeled_dataset = roboflow.download_dataset(dataset_url, "yolov8")
    fully_supervised_dataset_yaml = os.path.join(labeled_dataset.location, "data.yaml")

    experiment_name = f"{labeled_dataset.name}v{labeled_dataset.version}-{model_name}-stac-semi-{label_percentage}"
    base_dir = os.path.join(os.path.dirname(__file__), experiment_name)

    results_json_path = os.path.join(base_dir, "results.json")
    if os.path.exists(results_json_path) and not force_rerun:
        print(f"Found existing results.json at {results_json_path}")
        print("Exiting...")
        return

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    # target dir to copy the labeled dataset to and then modify to get to some % of the data
    supervised_dataset_dir = os.path.join(base_dir, "supervised_dataset")
    unlabeled_subset_dir = os.path.join(supervised_dataset_dir, "unlabeled")
    unlabeled_images_dir = os.path.join(unlabeled_subset_dir, "images")

    if os.path.exists(supervised_dataset_dir):
        shutil.rmtree(supervised_dataset_dir)
    os.makedirs(supervised_dataset_dir, exist_ok=True)
    os.makedirs(unlabeled_images_dir, exist_ok=True)

    # the directory to store the self-labeled dataset merged with the initial supervisory data
    # this is used to train the student model
    semi_supervised_dataset_dir = os.path.join(base_dir, "semi_supervised_dataset")

    print("Running fully supervised baseline...")
    model = YOLO(f"{model_name}.pt")
    model.train(
        data=fully_supervised_dataset_yaml,
        project=experiment_name,
        name="supervised_reference",
        exist_ok=True,
        **train_params
    )

    proper_val(model, split="test", max_det=max_det)

    coco_format_dataset = roboflow.download_dataset(dataset_url, "coco", location=labeled_dataset.location + "_coco")
    coco_format_test_annotations = os.path.join(coco_format_dataset.location, "test", "_annotations.coco.json")

    test_gt_annotations = COCO(coco_format_test_annotations)
    fix_pred_annotation_image_ids(os.path.join(experiment_name, "supervised_reference", "predictions.json"), coco_format_test_annotations)
    test_pred_annotations = test_gt_annotations.loadRes(os.path.join(experiment_name, "supervised_reference", "predictions.json"))
    coco_eval = COCOeval(test_gt_annotations, test_pred_annotations, "bbox")
    coco_eval.params.maxDets = [1, 10, max_det]
    coco_eval.evaluate()
    coco_eval.accumulate()
    summarize(coco_eval)

    fully_supervised_test_map = coco_eval.stats[0]
    fully_supervised_test_map_50 = coco_eval.stats[1]

    if not skip_stac:
        shutil.copytree(labeled_dataset.location, supervised_dataset_dir, dirs_exist_ok=True)

        labeled_dataset_yaml = os.path.join(supervised_dataset_dir, "data.yaml")

        # determine the images to keep from the labeled dataset
        all_images = os.listdir(os.path.join(supervised_dataset_dir, "train", "images"))
        random.shuffle(all_images)
        images_to_move = all_images[int(len(all_images) * label_percentage):]

        # strip the images and labels from the labeled dataset and store images in unlabeled_subset_dir
        for image in images_to_move:
            shutil.move(os.path.join(supervised_dataset_dir, "train", "images", image), os.path.join(unlabeled_images_dir, image))
            labels_file = os.path.join(supervised_dataset_dir, "train", "labels", image[:-4] + '.txt')
            if os.path.exists(labels_file):
                os.remove(labels_file)

        print("Training teacher model...")
        model = YOLO(f"{model_name}.pt")

        # train the teacher model on the labeled dataset
        model.train(
            data=labeled_dataset_yaml,
            project=experiment_name,
            name="teacher",
            exist_ok=True,
            **train_params
        )

        proper_val(model, split="test", max_det=max_det)

        fix_pred_annotation_image_ids(os.path.join(experiment_name, "teacher", "predictions.json"), coco_format_test_annotations)
        test_pred_annotations = test_gt_annotations.loadRes(os.path.join(experiment_name, "teacher", "predictions.json"))
        coco_eval = COCOeval(test_gt_annotations, test_pred_annotations, "bbox")
        coco_eval.params.maxDets = [1, 10, max_det]
        coco_eval.evaluate()
        coco_eval.accumulate()
        summarize(coco_eval)

        teacher_test_map = coco_eval.stats[0]
        teacher_test_map_50 = coco_eval.stats[1]

        model.val(
            split="test",
        )

        f1_curve = model.trainer.validator.metrics.box.f1_curve
        f1_score_maximizing_confidence = f1_curve.mean(0).argmax() / f1_curve.shape[1]
        print(f"F1 score maximizing confidence: {f1_score_maximizing_confidence}")

        print("Predicting unlabeled images...")
        # make sure to clear existing predictions
        if os.path.exists(os.path.join("stac", "teacher_predictions")):
            shutil.rmtree(os.path.join("stac", "teacher_predictions"))
        model.predict(unlabeled_images_dir, save=False, save_txt=True, name="teacher_predictions", exist_ok=True, conf=f1_score_maximizing_confidence)

        # generate the semi-supervised dataset

        def copy_files(src_dir, dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            for filename in os.listdir(src_dir):
                shutil.copy2(os.path.join(src_dir, filename), dest_dir)

        print("Generating semi-supervised dataset...")

        # clear the semi-supervised dataset directory if it exists
        if os.path.exists(semi_supervised_dataset_dir):
            shutil.rmtree(semi_supervised_dataset_dir)
            print(f"Cleared existing directory: {semi_supervised_dataset_dir}")

        # first copy the 'unlabeled' reference images and their predictions
        copy_files(unlabeled_images_dir, os.path.join(semi_supervised_dataset_dir, "train", "images"))
        copy_files(os.path.join(experiment_name, "teacher_predictions", "labels"), os.path.join(semi_supervised_dataset_dir, "train", "labels"))

        # filter out all images that have no predictions
        for image in os.listdir(os.path.join(semi_supervised_dataset_dir, "train", "images")):
            if not os.path.exists(os.path.join(semi_supervised_dataset_dir, "train", "labels", image[:-4] + '.txt')):
                os.remove(os.path.join(semi_supervised_dataset_dir, "train", "images", image))

        # then copy the existing dataset structure
        # note that this may overwrite some of the unlabeled images / predictions, but if that happens, the labels will be strictly better
        copy_files(os.path.join(supervised_dataset_dir, "train", "images"), os.path.join(semi_supervised_dataset_dir, "train", "images"))
        copy_files(os.path.join(supervised_dataset_dir, "train", "labels"), os.path.join(semi_supervised_dataset_dir, "train", "labels"))

        copy_files(os.path.join(supervised_dataset_dir, "valid", "images"), os.path.join(semi_supervised_dataset_dir, "valid", "images"))
        copy_files(os.path.join(supervised_dataset_dir, "valid", "labels"), os.path.join(semi_supervised_dataset_dir, "valid", "labels"))

        copy_files(os.path.join(supervised_dataset_dir, "test", "images"), os.path.join(semi_supervised_dataset_dir, "test", "images"))
        copy_files(os.path.join(supervised_dataset_dir, "test", "labels"), os.path.join(semi_supervised_dataset_dir, "test", "labels"))

        shutil.copy2(labeled_dataset_yaml, os.path.join(semi_supervised_dataset_dir, "data.yaml"))

        print("Training student model...")
        model = YOLO(f"{model_name}.pt")

        # note that ultralytics train has lots of augmentations by default, so we don't need to actively add any to mimic STAC
        model.train(
            data=os.path.join(semi_supervised_dataset_dir, "data.yaml"),
            project=experiment_name,
            name="student",
            exist_ok=True,
            **train_params
        )

        proper_val(model, split="test", max_det=max_det)

        fix_pred_annotation_image_ids(os.path.join(experiment_name, "student", "predictions.json"), coco_format_test_annotations)
        test_pred_annotations = test_gt_annotations.loadRes(os.path.join(experiment_name, "student", "predictions.json"))
        coco_eval = COCOeval(test_gt_annotations, test_pred_annotations, "bbox")
        coco_eval.params.maxDets = [1, 10, max_det]
        coco_eval.evaluate()
        coco_eval.accumulate()
        summarize(coco_eval)

        student_test_map = coco_eval.stats[0]
        student_test_map_50 = coco_eval.stats[1]

        results_dict = {
            "fully_supervised_ap": fully_supervised_test_map,
            "teacher_ap": teacher_test_map,
            "student_ap": student_test_map,
            "fully_supervised_ap_50": fully_supervised_test_map_50,
            "teacher_ap_50": teacher_test_map_50,
            "student_ap_50": student_test_map_50,
            "url": dataset_url,
            "label_percentage": label_percentage,
        }
    else:
        results_dict = {
            "fully_supervised_ap": fully_supervised_test_map,
            "fully_supervised_ap_50": fully_supervised_test_map_50,
            "url": dataset_url,
        }

    print(results_dict)

    with open(results_json_path, "w") as f:
        json.dump(results_dict, f)


if __name__ == "__main__":
    fire.Fire(run_benchmark)