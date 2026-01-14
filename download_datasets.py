"""
Pre-download all datasets from url_list.txt before running training.

Usage:
    python download_datasets.py url_list.txt

This will download both yolov8 and coco format for each dataset,
so that stac.py can use them without downloading.
"""

import roboflow
import fire
import os


def download_all_datasets(url_file: str):
    """
    Downloads all datasets from the URL file in both yolov8 and coco formats.

    Args:
        url_file: Path to file containing dataset URLs (one per line)
    """
    with open(url_file, "r") as f:
        dataset_urls = [line.strip() for line in f if line.strip()]
        dataset_urls = [url[:-1] if url.endswith('/') else url for url in dataset_urls]

    print(f"Found {len(dataset_urls)} datasets to download")

    for i, dataset_url in enumerate(dataset_urls):
        print(f"\n[{i+1}/{len(dataset_urls)}] Downloading: {dataset_url}")

        try:
            # Download yolov8 format
            print("  Downloading yolov8 format...")
            labeled_dataset = roboflow.download_dataset(dataset_url, "yolov8")
            print(f"  Downloaded to: {labeled_dataset.location}")

            # Download coco format
            print("  Downloading coco format...")
            coco_location = labeled_dataset.location + "_coco"
            coco_format_dataset = roboflow.download_dataset(dataset_url, "coco", location=coco_location)
            print(f"  Downloaded to: {coco_format_dataset.location}")

        except Exception as e:
            print(f"  ERROR downloading {dataset_url}: {e}")
            continue

    print("\n" + "="*50)
    print("All downloads complete!")
    print("="*50)


if __name__ == "__main__":
    fire.Fire(download_all_datasets)