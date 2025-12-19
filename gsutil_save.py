import subprocess
import glob
import os

base_dir = "."
gcs_dest = "gs://rf-detr-rf100-vl/yolo-bs-experiment"

# Find all folders ending with -batch{N}
folders = [
    f for f in glob.glob(os.path.join(base_dir, "*-batch*"))
    if os.path.isdir(f)
]

print(f"Found {len(folders)} folders to upload")

for folder in sorted(folders):
    print(f"Uploading: {folder}")
    subprocess.run(["gsutil", "-m", "cp", "-r", folder, gcs_dest])

print("Done!")