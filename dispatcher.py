import os
import subprocess
import time
import multiprocessing
import fire
import torch
import json


def get_all_gpus():
    """
    Detects available GPU IDs using torch.cuda.
    Returns a list of GPU IDs if available; otherwise, asserts failure.
    """
    assert torch.cuda.is_available(), "No GPUs available"
    count = torch.cuda.device_count()
    gpu_ids = list(range(count))
    print(f"Detected GPUs via torch: {gpu_ids}")
    return gpu_ids


def run_job(script_path, dataset_url, gpu_id, suppress_output=False, force_rerun=False):
    """
    Runs the training script with the given dataset URL on the specified GPU.
    The GPU is set via the CUDA_VISIBLE_DEVICES environment variable.
    
    Args:
        script_path (str): Path to the training script.
        dataset_url (str): URL for the dataset.
        gpu_id (int): GPU ID to use.
        suppress_output (bool): If True, suppress stdout and stderr.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"[GPU {gpu_id}] Running {script_path} with URL: {dataset_url}")
    if suppress_output:
        subprocess.run(
            ["python", script_path, dataset_url, "--force_rerun", str(force_rerun)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    else:
        subprocess.run(["python", script_path, dataset_url], env=env)


def collect_results_jsons(base_dir, output_file):
    """
    Searches for "results.json" in each immediate subdirectory of base_dir.
    Aggregates the JSON contents into a list and saves it to output_file.

    Args:
        base_dir (str): The directory whose immediate subdirectories will be checked.
        output_file (str): The file path to write the aggregated JSON list.
    """
    aggregated_results = {}

    # List entries in base_dir and process only directories.
    for entry in os.listdir(base_dir):
        subdir = os.path.join(base_dir, entry)
        if os.path.isdir(subdir):
            json_file = os.path.join(subdir, "results.json")
            if os.path.isfile(json_file):
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        aggregated_results[entry] = data
                    print(f"Collected JSON from: {json_file}")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {json_file}: {e}")
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")

    # Write the aggregated results to the output file.
    try:
        with open(output_file, "w") as f:
            json.dump(aggregated_results, f, indent=2)
        print(f"Aggregated results saved to: {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")


def main(script, url_file, suppress_output=False, output_file=None, force_rerun=False):
    """
    Manages GPU training jobs.
    
    Args:
        script (str): Path to the training script to run.
        url_file (str): File path containing dataset URLs (one per line).
        suppress_output (bool): If True, suppress output from the training scripts.
        output_file (str): File path to save aggregated results.
        force_rerun (bool): If True, rerun the script even if results.json already exists.
    """
    if suppress_output:
        print("Suppressing output")
    
    if output_file is None:
        output_file = os.path.join(os.path.dirname(__file__), "aggregated_results.json")
    
    if script.startswith("/"):
        script_dir = os.path.dirname(script)
    else:
        script_dir = os.path.dirname(__file__)
    
    print(f"After running the script, will gather results from {script_dir} subdirectories, and save an aggregation to {output_file}")

    # Determine GPU IDs: auto-detect using torch.
    gpu_ids = get_all_gpus()
    
    # Load dataset URLs from the file (ignoring empty lines).
    with open(url_file, "r") as f:
        dataset_urls = [line.strip() for line in f if line.strip()]
        dataset_urls = [url[:-1] if url.endswith('/') else url for url in dataset_urls]
    
    # Dictionary to track active processes per GPU.
    processes = {gpu: None for gpu in gpu_ids}
    
    # Main loop: launch jobs until all URLs have been processed and finished.
    while dataset_urls or any(proc is not None for proc in processes.values()):
        for gpu in gpu_ids:
            proc = processes[gpu]
            # If a process is finished, join it and mark the GPU as free.
            if proc is not None and not proc.is_alive():
                proc.join()
                processes[gpu] = None
            
            # If the GPU is free and there is a URL waiting, start a new job.
            if processes[gpu] is None and dataset_urls:
                url = dataset_urls.pop(0)
                new_proc = multiprocessing.Process(
                    target=run_job,
                    args=(script, url, gpu, suppress_output, force_rerun)
                )
                new_proc.start()
                processes[gpu] = new_proc
        
        # Sleep briefly before polling again.
        time.sleep(1)
    
    print("All training jobs have been completed.")

    # Collect results from all subdirectories.
    print("Collecting results from all subdirectories...")
    collect_results_jsons(script_dir, output_file)


if __name__ == '__main__':
    # example usage:
    # python dispatcher.py stac.py url_list.txt
    fire.Fire(main)