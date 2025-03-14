from ultralytics import YOLO
import roboflow
import os
import shutil
import json
import random
random.seed(37)
import fire


def run_benchmark(dataset_url: str, label_percentage: float=0.1):
    model_name = 'yolov8n'
    train_params = dict(
        epochs=300,
        batch=16,
    )

    # example url:
    # dataset_url = "https://universe.roboflow.com/brad-dwyer/aquarium-combined/dataset/6"

    print("Downloading labeled dataset...")
    labeled_dataset = roboflow.download_dataset(dataset_url, "yolov8")
    fully_supervised_dataset_yaml = os.path.join(labeled_dataset.location, "data.yaml")

    experiment_name = f"{labeled_dataset.name}v{labeled_dataset.version}-{model_name}-stac-semi-{label_percentage}"
    base_dir = os.path.join(os.path.dirname(__file__), experiment_name)

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

    fully_supervised_test_metrics = model.val(
        split="test",
    )
    fully_supervised_test_map = fully_supervised_test_metrics.box.map

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

    teacher_test_metrics = model.val(
        split="test",
    )
    teacher_test_map = teacher_test_metrics.box.map

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

    student_test_metrics = model.val(
        split="test",
    )
    student_test_map = student_test_metrics.box.map

    results_dict = {
        "fully_supervised_ap": fully_supervised_test_map,
        "teacher_ap": teacher_test_map,
        "student_ap": student_test_map,
        "url": dataset_url,
        "label_percentage": label_percentage,
    }

    print(results_dict)

    with open(os.path.join(base_dir, "results.json"), "w") as f:
        json.dump(results_dict, f)


if __name__ == "__main__":
    fire.Fire(run_benchmark)