#!/usr/bin/env python3
"""
Multi-GPU job queue v3:
- Individual dataset-level jobs (not batch-level)
- Work-stealing across all GPUs
- VRAM-based packing per GPU
- Maximum parallelization
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

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]


@dataclass
class Job:
    model_name: str      # e.g., 'yolov8n', 'yolo11s'
    model_size: str      # 'n', 's', 'm'
    batch_size: int
    dataset_url: str     # individual dataset URL
    vram_gb: float

    def __str__(self):
        # Short dataset name for display
        parts = self.dataset_url.split('/')
        dataset_short = parts[-3] if len(parts) >= 3 else self.dataset_url[-30:]
        return f"{self.model_name}-b{self.batch_size}/{dataset_short}"

    def short_id(self):
        parts = self.dataset_url.split('/')
        dataset_short = parts[-3][:15] if len(parts) >= 3 else "?"
        return f"{self.model_name}-b{self.batch_size}/{dataset_short}"


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


def load_dataset_urls(url_file: str) -> List[str]:
    """Load dataset URLs from file."""
    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
        urls = [url[:-1] if url.endswith('/') else url for url in urls]
    return urls


def get_experiment_name(dataset_url: str, model_name: str, batch: int) -> str:
    """Parse experiment folder name from URL (matches stac.py logic)."""
    url = dataset_url.rstrip('/')
    parts = url.split('/')
    try:
        dataset_idx = parts.index('dataset')
        dataset_name = parts[dataset_idx - 1]
        version = parts[dataset_idx + 1]
        return f"{dataset_name}v{version}-{model_name}-stac-semi-0.1-batch{batch}"
    except (ValueError, IndexError):
        return None


def find_incomplete_jobs(base_dir: str, model_prefix: str, url_file: str) -> List[Job]:
    """Find all incomplete model/batch/dataset combinations."""

    urls = load_dataset_urls(url_file)
    jobs = []
    sizes = ['n', 's', 'm']

    for size in sizes:
        model_name = f"{model_prefix}{size}"
        vram = lambda b: VRAM_ESTIMATES.get((size, b), 40.0)

        for batch in BATCH_SIZES:
            for url in urls:
                exp_name = get_experiment_name(url, model_name, batch)
                if exp_name is None:
                    continue

                results_path = os.path.join(base_dir, exp_name, 'results.json')

                # If results.json doesn't exist, this dataset needs to run
                if not os.path.isfile(results_path):
                    jobs.append(Job(
                        model_name=model_name,
                        model_size=size,
                        batch_size=batch,
                        dataset_url=url,
                        vram_gb=vram(batch)
                    ))

    return jobs


class SharedJobQueue:
    """Thread-safe shared job queue."""

    def __init__(self, jobs: List[Job]):
        self.lock = threading.Lock()
        # Sort: larger VRAM first (to pack efficiently), then by batch size
        self.pending = sorted(jobs, key=lambda j: (-j.vram_gb, -j.batch_size))
        self.completed = 0
        self.failed = 0
        self.total = len(jobs)

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

    def get_stats(self) -> Tuple[int, int, int, int]:
        with self.lock:
            return len(self.pending), self.completed, self.failed, self.total


class GPUWorker(threading.Thread):
    """Worker that processes individual dataset jobs on a GPU."""

    def __init__(self, gpu_id: int, job_queue: SharedJobQueue, max_vram: float, log_file: str):
        super().__init__()
        self.gpu_id = gpu_id
        self.job_queue = job_queue
        self.max_vram = max_vram
        self.log_file = log_file
        self.lock = threading.Lock()

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
        """Run stac.py directly for a single dataset."""
        cmd = [
            'python3', 'stac.py', job.dataset_url,
            f'--model_name={job.model_name}',
            '--skip_stac=True',
            f'--batch={job.batch_size}'
        ]

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        self.log(f"START: {job.short_id()}")

        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                self.log(f"DONE: {job.short_id()}")
                self.job_queue.mark_completed()
                with self.lock:
                    self.local_completed += 1
            else:
                err_snippet = result.stderr[:150] if result.stderr else "No stderr"
                self.log(f"FAIL: {job.short_id()} - {err_snippet}")
                self.job_queue.mark_failed()
                with self.lock:
                    self.local_failed += 1

        except Exception as e:
            self.log(f"ERROR: {job.short_id()} - {e}")
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
                with self.lock:
                    self.current_vram += job.vram_gb
                    self.current_jobs.append(job)

                thread = threading.Thread(target=self.run_job, args=(job,))
                thread.start()
                running_threads.append((job, thread))
            else:
                if self.job_queue.is_empty() and not running_threads:
                    break
                time.sleep(1)

        # Wait for remaining
        for job, thread in running_threads:
            thread.join()

        self.log("Worker finished")

    def stop(self):
        self.running = False

    def get_status(self) -> str:
        with self.lock:
            jobs_str = ", ".join(j.short_id() for j in self.current_jobs[:3])
            if len(self.current_jobs) > 3:
                jobs_str += f" +{len(self.current_jobs)-3}"
            return f"{self.current_vram:.0f}/{self.max_vram:.0f}GB [{jobs_str or 'idle'}]"


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU queue v3 (per-dataset jobs)')
    parser.add_argument('--model', type=str, choices=['yolov8', 'yolo11'], required=True)
    parser.add_argument('--max-vram', type=float, default=75.0)
    parser.add_argument('--base-dir', type=str, default=None)
    parser.add_argument('--url-file', type=str, default='url_list.txt')
    parser.add_argument('--log-file', type=str, default='multi_gpu_queue.log')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    base_dir = args.base_dir or os.path.dirname(os.path.abspath(__file__))
    gpus = get_available_gpus()

    print("=" * 70)
    print("Multi-GPU Queue v3 (per-dataset parallelization)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"GPUs: {gpus}")
    print(f"Max VRAM/GPU: {args.max_vram}GB")
    print()

    # Find all incomplete dataset jobs
    print("Scanning for incomplete jobs...")
    jobs = find_incomplete_jobs(base_dir, args.model, args.url_file)

    if not jobs:
        print("All jobs complete!")
        return

    # Summarize by model/batch
    summary: Dict[str, int] = defaultdict(int)
    for job in jobs:
        key = f"{job.model_name}-b{job.batch_size}"
        summary[key] += 1

    print(f"\nFound {len(jobs)} incomplete dataset jobs:")
    for key in sorted(summary.keys()):
        print(f"  {key}: {summary[key]} datasets")
    print()

    if args.dry_run:
        print("--dry-run: not executing")
        return

    # Init log
    with open(args.log_file, 'w') as f:
        f.write(f"=== Started {datetime.now()} ===\n")
        f.write(f"Total jobs: {len(jobs)}\n\n")

    # Create queue and workers
    job_queue = SharedJobQueue(jobs)
    workers = [GPUWorker(gpu_id, job_queue, args.max_vram, args.log_file) for gpu_id in gpus]

    print(f"Starting {len(workers)} GPU workers for {len(jobs)} jobs...")
    for w in workers:
        w.start()

    # Monitor
    try:
        while any(w.is_alive() for w in workers):
            time.sleep(30)
            pending, completed, failed, total = job_queue.get_stats()
            pct = 100 * completed / total if total > 0 else 0
            print("\n" + "=" * 70)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {completed}/{total} ({pct:.1f}%) | Pending: {pending} | Failed: {failed}")
            for w in workers:
                status = "RUN" if w.is_alive() else "END"
                print(f"  GPU {w.gpu_id} [{status}]: {w.get_status()}")
            print("=" * 70)
    except KeyboardInterrupt:
        print("\nStopping...")
        for w in workers:
            w.stop()

    for w in workers:
        w.join()

    # Final summary
    pending, completed, failed, total = job_queue.get_stats()
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print(f"  Total: {total}")
    print(f"  Completed: {completed}")
    print(f"  Failed: {failed}")
    print("-" * 40)
    for w in workers:
        print(f"  GPU {w.gpu_id}: {w.local_completed} done, {w.local_failed} failed")
    print("=" * 70)


if __name__ == '__main__':
    main()
