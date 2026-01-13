#!/usr/bin/env python3
"""
Multi-GPU job queue that distributes work across all available GPUs.
Uses VRAM packing to maximize utilization on each GPU.
Automatically detects incomplete jobs and skips completed ones.
"""

import subprocess
import threading
import time
import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from datetime import datetime

# VRAM estimates in GB (from gpu_queue.py)
VRAM_ESTIMATES = {
    ('n', 1): 2.5, ('n', 2): 3.0, ('n', 4): 3.5, ('n', 8): 5.0,
    ('n', 16): 7.0, ('n', 32): 12.0, ('n', 64): 19.0, ('n', 128): 29.0,
    ('s', 1): 3.5, ('s', 2): 4.0, ('s', 4): 8.5, ('s', 8): 10.0,
    ('s', 16): 14.0, ('s', 32): 17.0, ('s', 64): 26.0, ('s', 128): 60.0,
    ('m', 1): 6.0, ('m', 2): 6.0, ('m', 4): 8.5, ('m', 8): 14.0,
    ('m', 16): 22.0, ('m', 32): 31.0, ('m', 64): 46.0, ('m', 128): 62.0,
}

EXPECTED_DATASETS = 100
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]


@dataclass
class Job:
    model_name: str  # e.g., 'yolov8n', 'yolo11s'
    model_size: str  # 'n', 's', 'm'
    batch_size: int
    vram_gb: float
    datasets_remaining: int

    def __str__(self):
        return f"{self.model_name}-b{self.batch_size} ({self.datasets_remaining} datasets, {self.vram_gb}GB)"


def get_available_gpus() -> List[int]:
    """Detect available GPUs."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [int(x.strip()) for x in visible.split(',') if x.strip()]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
    except:
        pass

    return [0]  # Default to GPU 0


def find_incomplete_jobs(base_dir: str, model_prefix: str) -> List[Job]:
    """Scan for incomplete model/batch combinations."""

    # Pattern to match experiment folders
    pattern = re.compile(rf'^(.+)-({model_prefix}[nsm])-stac-semi-[\d.]+-batch(\d+)$')

    # Count completed datasets per model/batch
    completed: Dict[Tuple[str, int], Set[str]] = defaultdict(set)

    try:
        entries = os.listdir(base_dir)
    except FileNotFoundError:
        print(f"Directory not found: {base_dir}")
        return []

    for entry in entries:
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        match = pattern.match(entry)
        if not match:
            continue

        dataset_name = match.group(1)
        model = match.group(2)
        batch = int(match.group(3))

        results_path = os.path.join(entry_path, 'results.json')
        if os.path.isfile(results_path):
            completed[(model, batch)].add(dataset_name)

    # Build list of incomplete jobs
    jobs = []
    sizes = ['n', 's', 'm']

    for size in sizes:
        model_name = f"{model_prefix}{size}"
        for batch in BATCH_SIZES:
            done_count = len(completed.get((model_name, batch), set()))
            remaining = EXPECTED_DATASETS - done_count

            if remaining > 0:
                vram = VRAM_ESTIMATES.get((size, batch), 40.0)
                jobs.append(Job(
                    model_name=model_name,
                    model_size=size,
                    batch_size=batch,
                    vram_gb=vram,
                    datasets_remaining=remaining
                ))

    return jobs


def pack_jobs_to_gpus(jobs: List[Job], num_gpus: int, max_vram_per_gpu: float) -> List[List[Job]]:
    """
    Pack jobs onto GPUs using first-fit decreasing algorithm.
    Prioritizes jobs with more remaining datasets and higher VRAM (to finish big jobs first).
    """
    # Sort by remaining datasets (desc), then by VRAM (desc)
    sorted_jobs = sorted(jobs, key=lambda j: (-j.datasets_remaining, -j.vram_gb))

    # Initialize GPU allocations
    gpu_jobs: List[List[Job]] = [[] for _ in range(num_gpus)]
    gpu_vram: List[float] = [0.0] * num_gpus

    for job in sorted_jobs:
        # Find GPU with most remaining capacity that can fit this job
        best_gpu = -1
        best_remaining = -1

        for i in range(num_gpus):
            remaining = max_vram_per_gpu - gpu_vram[i]
            if job.vram_gb <= remaining and remaining > best_remaining:
                best_gpu = i
                best_remaining = remaining

        if best_gpu >= 0:
            gpu_jobs[best_gpu].append(job)
            gpu_vram[best_gpu] += job.vram_gb
        else:
            # Find GPU with least load (job will wait)
            min_gpu = min(range(num_gpus), key=lambda i: gpu_vram[i])
            gpu_jobs[min_gpu].append(job)
            # Don't add to gpu_vram since it will run after others finish

    return gpu_jobs


class GPUWorker(threading.Thread):
    """Worker thread that processes jobs on a specific GPU."""

    def __init__(self, gpu_id: int, jobs: List[Job], max_vram: float, log_file: str):
        super().__init__()
        self.gpu_id = gpu_id
        self.jobs = jobs
        self.max_vram = max_vram
        self.log_file = log_file
        self.completed = 0
        self.failed = 0
        self.current_jobs: List[Job] = []
        self.lock = threading.Lock()

    def log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}][GPU {self.gpu_id}] {msg}"
        print(line)
        with open(self.log_file, 'a') as f:
            f.write(line + "\n")

    def run_job(self, job: Job):
        """Run a single dispatcher job."""
        cmd = [
            'python3', 'dispatcher.py', 'stac.py', 'url_list.txt',
            f'--model_name={job.model_name}',
            '--skip_stac=True',
            f'--batch={job.batch_size}'
        ]

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        self.log(f"Starting: {job}")

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                self.log(f"Completed: {job}")
                with self.lock:
                    self.completed += 1
            else:
                self.log(f"Failed: {job} - {result.stderr[:200] if result.stderr else 'No stderr'}")
                with self.lock:
                    self.failed += 1

        except Exception as e:
            self.log(f"Exception: {job} - {e}")
            with self.lock:
                self.failed += 1

    def run(self):
        """Process jobs with VRAM packing."""
        pending = list(self.jobs)
        running: List[Tuple[Job, threading.Thread]] = []
        current_vram = 0.0

        while pending or running:
            # Clean up finished jobs
            still_running = []
            for job, thread in running:
                if thread.is_alive():
                    still_running.append((job, thread))
                else:
                    thread.join()
                    current_vram -= job.vram_gb
                    with self.lock:
                        self.current_jobs = [j for j, _ in still_running]
            running = still_running

            # Start new jobs if VRAM available
            jobs_to_start = []
            for job in pending:
                if current_vram + job.vram_gb <= self.max_vram:
                    jobs_to_start.append(job)
                    current_vram += job.vram_gb

            for job in jobs_to_start:
                pending.remove(job)
                thread = threading.Thread(target=self.run_job, args=(job,))
                thread.start()
                running.append((job, thread))
                with self.lock:
                    self.current_jobs.append(job)

            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU job queue')
    parser.add_argument('--model', type=str, choices=['yolov8', 'yolo11'], required=True,
                        help='Model family to use')
    parser.add_argument('--max-vram', type=float, default=75.0,
                        help='Max VRAM per GPU in GB (default: 75)')
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base directory to scan for completed experiments')
    parser.add_argument('--log-file', type=str, default='multi_gpu_queue.log',
                        help='Log file path')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show plan without executing')
    args = parser.parse_args()

    base_dir = args.base_dir or os.path.dirname(os.path.abspath(__file__))
    gpus = get_available_gpus()

    print("=" * 70)
    print("Multi-GPU Queue")
    print("=" * 70)
    print(f"Model family: {args.model}")
    print(f"Available GPUs: {gpus}")
    print(f"Max VRAM per GPU: {args.max_vram}GB")
    print(f"Base directory: {base_dir}")
    print()

    # Find incomplete jobs
    jobs = find_incomplete_jobs(base_dir, args.model)

    if not jobs:
        print("No incomplete jobs found. All done!")
        return

    print(f"Found {len(jobs)} incomplete job(s):")
    for job in sorted(jobs, key=lambda j: (-j.datasets_remaining, j.model_name, j.batch_size)):
        print(f"  {job}")
    print()

    # Pack jobs to GPUs
    gpu_jobs = pack_jobs_to_gpus(jobs, len(gpus), args.max_vram)

    print("GPU Assignment Plan:")
    print("-" * 70)
    for i, (gpu_id, assigned) in enumerate(zip(gpus, gpu_jobs)):
        total_vram = sum(j.vram_gb for j in assigned)
        total_datasets = sum(j.datasets_remaining for j in assigned)
        print(f"GPU {gpu_id}: {len(assigned)} jobs, ~{total_vram:.1f}GB VRAM, {total_datasets} datasets")
        for job in assigned:
            print(f"    - {job}")
    print("-" * 70)

    if args.dry_run:
        print("\nDry run - not executing. Remove --dry-run to start.")
        return

    # Clear log file
    with open(args.log_file, 'w') as f:
        f.write(f"=== Multi-GPU Queue started at {datetime.now()} ===\n")

    # Start workers
    print(f"\nStarting {len(gpus)} GPU workers...")
    workers = []
    for gpu_id, assigned in zip(gpus, gpu_jobs):
        if assigned:
            worker = GPUWorker(gpu_id, assigned, args.max_vram, args.log_file)
            worker.start()
            workers.append(worker)

    # Monitor progress
    try:
        while any(w.is_alive() for w in workers):
            time.sleep(10)
            print("\n" + "=" * 70)
            print(f"Status at {datetime.now().strftime('%H:%M:%S')}:")
            for w in workers:
                with w.lock:
                    current = ", ".join(f"{j.model_name}-b{j.batch_size}" for j in w.current_jobs) or "idle"
                print(f"  GPU {w.gpu_id}: {w.completed} done, {w.failed} failed, running: [{current}]")
            print("=" * 70)
    except KeyboardInterrupt:
        print("\nInterrupted! Waiting for current jobs to finish...")

    # Wait for all workers
    for w in workers:
        w.join()

    # Summary
    total_completed = sum(w.completed for w in workers)
    total_failed = sum(w.failed for w in workers)

    print("\n" + "=" * 70)
    print("Final Summary:")
    print(f"  Completed: {total_completed}")
    print(f"  Failed: {total_failed}")
    print("=" * 70)


if __name__ == '__main__':
    main()
