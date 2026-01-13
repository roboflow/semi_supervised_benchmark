#!/usr/bin/env python3
"""
Multi-GPU job queue with:
- Work-stealing: idle GPUs grab jobs from shared queue
- Pure VRAM-based packing: jobs stack if they fit
"""

import subprocess
import threading
import time
import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
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
        return f"{self.model_name}-b{self.batch_size} ({self.datasets_remaining} left, {self.vram_gb}GB)"


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

    return [0]


def find_incomplete_jobs(base_dir: str, model_prefix: str) -> List[Job]:
    """Scan for incomplete model/batch combinations."""
    pattern = re.compile(rf'^(.+)-({model_prefix}[nsm])-stac-semi-[\d.]+-batch(\d+)$')

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


class SharedJobQueue:
    """Thread-safe shared job queue with work-stealing."""

    def __init__(self, jobs: List[Job]):
        self.lock = threading.Lock()
        # Sort: prioritize jobs with more remaining, then larger VRAM
        self.pending = sorted(jobs, key=lambda j: (-j.datasets_remaining, -j.vram_gb))
        self.completed = 0
        self.failed = 0

    def get_job(self, available_vram: float) -> Optional[Job]:
        """Get a job that fits in available VRAM."""
        with self.lock:
            for i, job in enumerate(self.pending):
                if job.vram_gb <= available_vram:
                    return self.pending.pop(i)
            return None

    def mark_completed(self):
        with self.lock:
            self.completed += 1

    def mark_failed(self):
        with self.lock:
            self.failed += 1

    def is_empty(self) -> bool:
        with self.lock:
            return len(self.pending) == 0

    def remaining_count(self) -> int:
        with self.lock:
            return len(self.pending)

    def get_stats(self) -> Tuple[int, int, int]:
        with self.lock:
            return len(self.pending), self.completed, self.failed


class GPUWorker(threading.Thread):
    """Worker that processes jobs on a GPU with work-stealing and VRAM packing."""

    def __init__(self, gpu_id: int, job_queue: SharedJobQueue, max_vram: float, log_file: str):
        super().__init__()
        self.gpu_id = gpu_id
        self.job_queue = job_queue
        self.max_vram = max_vram
        self.log_file = log_file
        self.lock = threading.Lock()

        # Tracking
        self.current_vram = 0.0
        self.current_jobs: List[Job] = []
        self.local_completed = 0
        self.local_failed = 0
        self.running = True

    def log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}][GPU {self.gpu_id}] {msg}"
        print(line)
        try:
            with open(self.log_file, 'a') as f:
                f.write(line + "\n")
        except:
            pass

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

        self.log(f"START: {job}")

        try:
            # Don't capture output - let it stream to console for debugging
            result = subprocess.run(cmd, env=env)

            if result.returncode == 0:
                self.log(f"DONE: {job}")
                self.job_queue.mark_completed()
                with self.lock:
                    self.local_completed += 1
            else:
                self.log(f"FAIL: {job} - returncode={result.returncode}")
                self.job_queue.mark_failed()
                with self.lock:
                    self.local_failed += 1

        except Exception as e:
            self.log(f"ERROR: {job} - {e}")
            self.job_queue.mark_failed()
            with self.lock:
                self.local_failed += 1

        # Release VRAM
        with self.lock:
            self.current_vram -= job.vram_gb
            self.current_jobs = [j for j in self.current_jobs if j != job]

    def run(self):
        """Main loop: grab jobs from shared queue if VRAM allows."""
        running_threads: List[Tuple[Job, threading.Thread]] = []

        while self.running:
            # Clean up finished threads
            still_running = []
            for job, thread in running_threads:
                if thread.is_alive():
                    still_running.append((job, thread))
                else:
                    thread.join()
            running_threads = still_running

            # Get available VRAM
            with self.lock:
                available_vram = self.max_vram - self.current_vram

            # Try to get a job that fits
            job = self.job_queue.get_job(available_vram)

            if job:
                # Reserve VRAM
                with self.lock:
                    self.current_vram += job.vram_gb
                    self.current_jobs.append(job)

                # Start job thread
                thread = threading.Thread(target=self.run_job, args=(job,))
                thread.start()
                running_threads.append((job, thread))
            else:
                # No job available
                if self.job_queue.is_empty() and not running_threads:
                    break
                time.sleep(2)

        # Wait for remaining
        for job, thread in running_threads:
            thread.join()

        self.log("Worker finished")

    def stop(self):
        self.running = False

    def get_status(self) -> str:
        with self.lock:
            jobs_str = ", ".join(f"{j.model_name}-b{j.batch_size}" for j in self.current_jobs) or "idle"
            return f"vram={self.current_vram:.1f}/{self.max_vram:.0f}GB, [{jobs_str}]"


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU queue with work-stealing')
    parser.add_argument('--model', type=str, choices=['yolov8', 'yolo11'], required=True)
    parser.add_argument('--max-vram', type=float, default=75.0, help='Max VRAM per GPU (default: 75)')
    parser.add_argument('--base-dir', type=str, default=None)
    parser.add_argument('--log-file', type=str, default='multi_gpu_queue.log')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    base_dir = args.base_dir or os.path.dirname(os.path.abspath(__file__))
    gpus = get_available_gpus()

    print("=" * 70)
    print("Multi-GPU Queue v2 (work-stealing + VRAM packing)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"GPUs: {gpus}")
    print(f"Max VRAM/GPU: {args.max_vram}GB")
    print()

    # Find incomplete jobs
    jobs = find_incomplete_jobs(base_dir, args.model)

    if not jobs:
        print("All jobs complete!")
        return

    print(f"Found {len(jobs)} incomplete jobs:")
    for job in sorted(jobs, key=lambda j: (-j.datasets_remaining, j.batch_size)):
        print(f"  [{job.vram_gb:.1f}GB] {job}")
    print()

    total_datasets = sum(j.datasets_remaining for j in jobs)
    print(f"Total datasets to process: {total_datasets}")
    print()

    if args.dry_run:
        print("--dry-run: not executing")
        return

    # Init log
    with open(args.log_file, 'w') as f:
        f.write(f"=== Started {datetime.now()} ===\n")

    # Create shared queue and workers
    job_queue = SharedJobQueue(jobs)
    workers = [GPUWorker(gpu_id, job_queue, args.max_vram, args.log_file) for gpu_id in gpus]

    print(f"Starting {len(workers)} GPU workers...")
    for w in workers:
        w.start()

    # Monitor
    try:
        while any(w.is_alive() for w in workers):
            time.sleep(15)
            pending, completed, failed = job_queue.get_stats()
            print("\n" + "=" * 70)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Pending: {pending} | Done: {completed} | Failed: {failed}")
            for w in workers:
                alive = "RUN" if w.is_alive() else "END"
                print(f"  GPU {w.gpu_id} [{alive}]: {w.get_status()}")
            print("=" * 70)
    except KeyboardInterrupt:
        print("\nStopping...")
        for w in workers:
            w.stop()

    for w in workers:
        w.join()

    # Final
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print(f"  Completed: {job_queue.completed}")
    print(f"  Failed: {job_queue.failed}")
    for w in workers:
        print(f"  GPU {w.gpu_id}: {w.local_completed} done, {w.local_failed} failed")
    print("=" * 70)


if __name__ == '__main__':
    main()
