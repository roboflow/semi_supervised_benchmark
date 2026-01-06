#!/usr/bin/env python3
"""
GPU-aware job queue that maximizes GPU utilization while preventing OOM.
Jobs start as soon as there's enough VRAM available, rather than waiting for batches.
"""

import subprocess
import threading
import time
import argparse
from dataclasses import dataclass
from typing import List
from queue import Queue
import sys

# VRAM estimates in GB - derived from working 3-job combinations in run_v8.sh/run_11.sh
# These ensure 3 jobs can run together as in the original groupings
VRAM_ESTIMATES = {
    # YOLOv8/YOLO11 nano
    ('n', 1): 2.0,
    ('n', 2): 2.5,
    ('n', 4): 3.0,
    ('n', 8): 4.0,
    ('n', 16): 6.0,
    ('n', 32): 10.0,
    ('n', 64): 16.0,
    ('n', 128): 24.0,   # m4+s4+n128 must fit: ~7+7+24=38 ✓
    # YOLOv8/YOLO11 small
    ('s', 1): 3.0,
    ('s', 2): 3.5,
    ('s', 4): 7.0,
    ('s', 8): 8.0,
    ('s', 16): 12.0,
    ('s', 32): 14.0,
    ('s', 64): 22.0,
    ('s', 128): 50.0,   # m2+n4+s128 must fit: ~5+3+50=58 ✓
    # YOLOv8/YOLO11 medium
    ('m', 1): 5.0,
    ('m', 2): 5.0,
    ('m', 4): 7.0,
    ('m', 8): 12.0,
    ('m', 16): 18.0,
    ('m', 32): 26.0,    # m1+m16+m32 must fit: 5+18+26=49 ✓
    ('m', 64): 38.0,    # s1+s2+m64 must fit: 3+3.5+38=44.5 ✓
    ('m', 128): 52.0,   # n1+n2+m128 must fit: 2+2.5+52=56.5 ✓
}

@dataclass
class Job:
    cmd: List[str]
    model_size: str  # 'n', 's', 'm'
    batch_size: int
    vram_gb: float

    def __str__(self):
        return f"model={self.model_size} batch={self.batch_size} vram={self.vram_gb}GB"


class GPUQueue:
    def __init__(self, max_vram_gb: float = 75.0, fail_log: str = "queue_errors.txt"):
        self.max_vram_gb = max_vram_gb
        self.fail_log = fail_log
        self.current_vram = 0.0
        self.lock = threading.Lock()
        self.jobs_pending: List[Job] = []
        self.jobs_running: List[tuple] = []  # (Job, subprocess, thread)
        self.completed = 0
        self.failed = 0

        # Clear fail log
        with open(fail_log, 'w') as f:
            pass

    def add_job(self, cmd: List[str]):
        """Parse command and add to queue."""
        model_size = None
        batch_size = None

        for arg in cmd:
            if '--model_name=' in arg:
                name = arg.split('=')[1]
                if 'n' in name:
                    model_size = 'n'
                elif 's' in name:
                    model_size = 's'
                elif 'm' in name:
                    model_size = 'm'
            elif '--batch=' in arg:
                batch_size = int(arg.split('=')[1])

        if model_size is None or batch_size is None:
            print(f"WARNING: Could not parse job: {' '.join(cmd)}")
            return

        vram = VRAM_ESTIMATES.get((model_size, batch_size), 40.0)  # Default high if unknown
        job = Job(cmd=cmd, model_size=model_size, batch_size=batch_size, vram_gb=vram)
        self.jobs_pending.append(job)
        print(f"Added job: {job}")

    def _run_job(self, job: Job):
        """Run a single job and track completion."""
        cmd_str = ' '.join(job.cmd)
        print(f"\n>>> STARTING: {job} | Current VRAM: {self.current_vram:.1f}GB")

        try:
            result = subprocess.run(job.cmd, capture_output=False)
            if result.returncode != 0:
                with open(self.fail_log, 'a') as f:
                    f.write(f"FAILED: {cmd_str}\n")
                with self.lock:
                    self.failed += 1
            else:
                with self.lock:
                    self.completed += 1
        except Exception as e:
            with open(self.fail_log, 'a') as f:
                f.write(f"FAILED (exception): {cmd_str} - {e}\n")
            with self.lock:
                self.failed += 1

        with self.lock:
            self.current_vram -= job.vram_gb
            self.jobs_running = [(j, p, t) for j, p, t in self.jobs_running if j != job]

        print(f"\n<<< FINISHED: {job} | VRAM freed: {job.vram_gb}GB | Remaining: {self.current_vram:.1f}GB")

    def _try_start_jobs(self):
        """Try to start as many pending jobs as VRAM allows."""
        with self.lock:
            # Sort by VRAM (largest first) to pack efficiently
            self.jobs_pending.sort(key=lambda j: -j.vram_gb)

            started = []
            for job in self.jobs_pending:
                if self.current_vram + job.vram_gb <= self.max_vram_gb:
                    self.current_vram += job.vram_gb
                    thread = threading.Thread(target=self._run_job, args=(job,))
                    thread.start()
                    self.jobs_running.append((job, None, thread))
                    started.append(job)

            for job in started:
                self.jobs_pending.remove(job)

    def run(self):
        """Process all jobs in the queue."""
        total_jobs = len(self.jobs_pending)
        print(f"\n{'='*60}")
        print(f"Starting GPU Queue with {total_jobs} jobs")
        print(f"Max VRAM: {self.max_vram_gb}GB")
        print(f"{'='*60}\n")

        while self.jobs_pending or self.jobs_running:
            self._try_start_jobs()

            # Wait a bit before checking again
            time.sleep(1)

            # Show status periodically
            with self.lock:
                pending = len(self.jobs_pending)
                running = len(self.jobs_running)
                if running > 0:
                    running_info = ", ".join([f"{j.model_size}{j.batch_size}" for j, _, _ in self.jobs_running])
                    print(f"\rRunning: [{running_info}] | VRAM: {self.current_vram:.1f}/{self.max_vram_gb}GB | Pending: {pending} | Done: {self.completed}", end="", flush=True)

        print(f"\n\n{'='*60}")
        print(f"Queue complete! Completed: {self.completed}, Failed: {self.failed}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='GPU-aware job queue')
    parser.add_argument('--max-vram', type=float, default=75.0,
                        help='Max VRAM to use in GB (default: 75, leaving 5GB headroom on 80GB)')
    parser.add_argument('--fail-log', type=str, default='queue_errors.txt',
                        help='File to log failed jobs')
    parser.add_argument('--jobs-file', type=str, help='File containing jobs (one per line)')
    parser.add_argument('--model', type=str, choices=['yolov8', 'yolo11'], default='yolov8',
                        help='Model family to use')
    parser.add_argument('--skip-stac', action='store_true', default=True,
                        help='Skip STAC (default: True)')
    args = parser.parse_args()

    queue = GPUQueue(max_vram_gb=args.max_vram, fail_log=args.fail_log)

    if args.jobs_file:
        # Read jobs from file
        with open(args.jobs_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    queue.add_job(line.split())
    else:
        # Default: add all model/batch combinations
        model_prefix = args.model
        sizes = ['n', 's', 'm']
        batches = [1, 2, 4, 8, 16, 32, 64, 128]

        for size in sizes:
            for batch in batches:
                model_name = f"{model_prefix}{size}"
                cmd = [
                    'python', 'dispatcher.py', 'stac.py', 'url_list.txt',
                    f'--model_name={model_name}',
                    f'--skip_stac=True',
                    f'--batch={batch}'
                ]
                queue.add_job(cmd)

    queue.run()


if __name__ == '__main__':
    main()
