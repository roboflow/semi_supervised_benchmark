import os
import sys
import time
import subprocess
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import torch


def _list_dataset_names(datasets_base_dir: str) -> List[str]:
    """Return sorted list of dataset directory names under datasets_base_dir.

    A valid dataset directory is a subdirectory that contains a "test" subfolder,
    matching what single_artifact_eval expects: <datasets_base_dir>/<dataset_name>/test
    """
    if not os.path.isdir(datasets_base_dir):
        raise FileNotFoundError(f"datasets_base_dir not found: {datasets_base_dir}")

    dataset_names: List[str] = []
    for entry in os.listdir(datasets_base_dir):
        full_path = os.path.join(datasets_base_dir, entry)
        if os.path.isdir(full_path) and os.path.isdir(os.path.join(full_path, "test")):
            dataset_names.append(entry)
    dataset_names.sort()
    return dataset_names


def _get_available_gpus() -> List[int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but GPU dispatching was requested.")
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("No CUDA devices detected.")
    return list(range(num_gpus))


def _build_eval_command(
    eval_script_path: str,
    dataset_name: str,
    datasets_base_dir: str,
    models_base_dir: str,
    model_name: str,
    max_dets: int,
    buffer_time: float,
) -> List[str]:
    # single_artifact_eval.py exposes Fire(run_benchmark) directly, so no subcommand
    return [
        sys.executable,
        eval_script_path,
        f"--dataset_name={dataset_name}",
        f"--dataset_base_dir={datasets_base_dir}",
        f"--models_base_dir={models_base_dir}",
        f"--model_name={model_name}",
        f"--max_dets={max_dets}",
        f"--buffer_time={buffer_time}",
    ]


def _spawn_on_gpu(
    gpu_id: int,
    cmd: List[str],
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if extra_env:
        env.update(extra_env)
    # Inherit stdio so logs stream to console; change to PIPE/files if needed
    return subprocess.Popen(cmd, env=env)


def dispatch(
    datasets_base_dir: str,
    models_base_dir: str,
    model_name: str = "yolov8n",
    max_dets: int = 500,
    buffer_time: float = 0.2,
    datasets: Optional[List[str]] = None,
    max_parallel: Optional[int] = None,
    poll_interval_seconds: float = 1.0,
) -> None:
    """Dispatch evaluations across GPUs, one subprocess per GPU.

    Args:
        datasets_base_dir: Root directory containing dataset subdirectories.
        models_base_dir: Root directory containing per-dataset trained models.
        model_name: Model variant identifier to select the correct model directory.
        max_dets: Max detections to pass to the evaluator.
        buffer_time: Buffer time between frames for the evaluator.
        datasets: Optional explicit list of dataset names (subdirs) to run. If None, auto-discovers.
        max_parallel: Optional limit lower than GPU count; defaults to all available GPUs.
        poll_interval_seconds: How often to poll child processes for completion.
    """
    eval_script_path = os.path.join(os.path.dirname(__file__), "single_artifact_eval.py")
    if not os.path.isfile(eval_script_path):
        raise FileNotFoundError(f"single_artifact_eval.py not found at {eval_script_path}")

    # Determine datasets to process
    if datasets is None:
        dataset_names = _list_dataset_names(datasets_base_dir)
    else:
        dataset_names = sorted(datasets)
    if not dataset_names:
        raise ValueError("No datasets found to process.")

    # Determine GPU pool
    gpu_ids = _get_available_gpus()
    if max_parallel is not None:
        gpu_ids = gpu_ids[: max(0, min(max_parallel, len(gpu_ids)))]
    if not gpu_ids:
        raise ValueError("No GPUs selected for dispatching.")

    print(f"Dispatching {len(dataset_names)} datasets across {len(gpu_ids)} GPU(s): {gpu_ids}")

    jobs: Deque[Tuple[str, List[str]]] = deque()
    for ds_name in dataset_names:
        cmd = _build_eval_command(
            eval_script_path=eval_script_path,
            dataset_name=ds_name,
            datasets_base_dir=datasets_base_dir,
            models_base_dir=models_base_dir,
            model_name=model_name,
            max_dets=max_dets,
            buffer_time=buffer_time,
        )
        jobs.append((ds_name, cmd))

    active: Dict[int, Optional[Tuple[str, subprocess.Popen]]] = {gpu_id: None for gpu_id in gpu_ids}
    completed: List[Tuple[str, int]] = []

    try:
        # Prime GPUs with initial jobs
        for gpu_id in gpu_ids:
            if jobs:
                ds_name, cmd = jobs.popleft()
                print(f"[GPU {gpu_id}] START {ds_name}: {' '.join(cmd)}")
                proc = _spawn_on_gpu(gpu_id, cmd)
                active[gpu_id] = (ds_name, proc)

        # Main scheduling loop
        while jobs or any(active.values()):
            time.sleep(poll_interval_seconds)

            # Check for finished processes and schedule new ones
            for gpu_id, slot in list(active.items()):
                if slot is None:
                    # Idle GPU, schedule next if available
                    if jobs:
                        ds_name, cmd = jobs.popleft()
                        print(f"[GPU {gpu_id}] START {ds_name}: {' '.join(cmd)}")
                        proc = _spawn_on_gpu(gpu_id, cmd)
                        active[gpu_id] = (ds_name, proc)
                    continue

                ds_name, proc = slot
                ret = proc.poll()
                if ret is None:
                    continue  # still running

                # Process finished
                print(f"[GPU {gpu_id}] END   {ds_name} (exit={ret})")
                completed.append((ds_name, ret))
                active[gpu_id] = None

        # Final report
        failures = [(n, rc) for n, rc in completed if rc != 0]
        print(
            f"All jobs completed. Success: {len(completed) - len(failures)}, "
            f"Failures: {len(failures)}"
        )
        if failures:
            print("Failed jobs:")
            for name, rc in failures:
                print(f"  - {name}: exit code {rc}")

    except KeyboardInterrupt:
        print("Interrupted. Terminating child processes...")
        for slot in active.values():
            if slot is not None:
                _, proc = slot
                try:
                    proc.terminate()
                except Exception:
                    pass
        raise


if __name__ == "__main__":
    import fire

    fire.Fire(dispatch)


