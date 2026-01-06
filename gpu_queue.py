#!/usr/bin/env python3
"""
GPU-aware job queue that maximizes GPU utilization while preventing OOM.
Jobs start as soon as there's enough VRAM available, rather than waiting for batches.
"""

import subprocess
import threading
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Dict
from queue import Queue
import sys
import os
import fcntl
from datetime import datetime

# VRAM estimates in GB - with ~20% safety buffer
# Original combos still fit under 75GB:
#   n1 + n2 + m128 = 2.5 + 3 + 62 = 67.5GB ✓
#   m2 + n4 + s128 = 6 + 3.5 + 60 = 69.5GB ✓
#   s1 + s2 + m64  = 3.5 + 4 + 46 = 53.5GB ✓
#   m1 + m16 + m32 = 6 + 22 + 31 = 59GB ✓
VRAM_ESTIMATES = {
    # YOLOv8/YOLO11 nano (+20% buffer)
    ('n', 1): 2.5,
    ('n', 2): 3.0,
    ('n', 4): 3.5,
    ('n', 8): 5.0,
    ('n', 16): 7.0,
    ('n', 32): 12.0,
    ('n', 64): 19.0,
    ('n', 128): 29.0,
    # YOLOv8/YOLO11 small (+20% buffer)
    ('s', 1): 3.5,
    ('s', 2): 4.0,
    ('s', 4): 8.5,
    ('s', 8): 10.0,
    ('s', 16): 14.0,
    ('s', 32): 17.0,
    ('s', 64): 26.0,
    ('s', 128): 60.0,
    # YOLOv8/YOLO11 medium (+20% buffer)
    ('m', 1): 6.0,
    ('m', 2): 6.0,
    ('m', 4): 8.5,
    ('m', 8): 14.0,
    ('m', 16): 22.0,
    ('m', 32): 31.0,
    ('m', 64): 46.0,
    ('m', 128): 62.0,
}


def safe_write_to_file(filepath: str, content: str, mode: str = 'a'):
    """Write to file with file locking for multiprocess safety."""
    with open(filepath, mode) as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(content)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@dataclass
class Job:
    cmd: List[str]
    model_size: str  # 'n', 's', 'm'
    batch_size: int
    vram_gb: float
    retries: int = 0
    max_retries: int = 1

    def __str__(self):
        return f"model={self.model_size} batch={self.batch_size} vram={self.vram_gb}GB"

    def job_key(self):
        return (self.model_size, self.batch_size)


class GPUQueue:
    def __init__(self, max_vram_gb: float = 75.0, fail_log: str = "queue_errors.txt",
                 oom_log: str = "oom_events.txt", safety_margin: float = 5.0):
        self.max_vram_gb = max_vram_gb
        self.fail_log = fail_log
        self.oom_log = oom_log
        self.safety_margin = safety_margin  # Extra buffer after OOMs
        self.current_vram = 0.0
        self.lock = threading.Lock()
        self.jobs_pending: List[Job] = []
        self.jobs_running: List[tuple] = []  # (Job, subprocess, thread)
        self.completed = 0
        self.failed = 0
        self.oom_count = 0
        self.oom_cooldown_until = 0  # timestamp until which we wait

        # Track VRAM adjustments per job type
        self.vram_adjustments: Dict[tuple, float] = {}

        # Clear logs
        safe_write_to_file(fail_log, f"=== Queue started at {datetime.now()} ===\n", 'w')
        safe_write_to_file(oom_log, f"=== Queue started at {datetime.now()} ===\n", 'w')

    def _get_adjusted_vram(self, model_size: str, batch_size: int) -> float:
        """Get VRAM estimate with any adjustments from OOMs."""
        base_vram = VRAM_ESTIMATES.get((model_size, batch_size), 40.0)
        adjustment = self.vram_adjustments.get((model_size, batch_size), 0.0)
        return base_vram + adjustment

    def _record_oom(self, job: Job, stderr_output: str):
        """Record OOM event and adjust VRAM estimates."""
        timestamp = datetime.now().isoformat()

        with self.lock:
            self.oom_count += 1
            # Increase VRAM estimate for this job type by 20%
            key = job.job_key()
            current_adjustment = self.vram_adjustments.get(key, 0.0)
            base_vram = VRAM_ESTIMATES.get(key, 40.0)
            new_adjustment = current_adjustment + (base_vram * 0.2)
            self.vram_adjustments[key] = new_adjustment

            # Set cooldown - wait 10 seconds before starting new jobs
            self.oom_cooldown_until = time.time() + 10

            # Reduce max_vram by safety margin after OOM (but not below 40GB)
            if self.oom_count <= 3:  # Only reduce first 3 times
                self.max_vram_gb = max(40.0, self.max_vram_gb - self.safety_margin)

        # Log OOM event (with file locking)
        log_entry = (
            f"\n{'='*60}\n"
            f"OOM EVENT #{self.oom_count} at {timestamp}\n"
            f"Job: {job}\n"
            f"Command: {' '.join(job.cmd)}\n"
            f"VRAM adjustment for {key}: +{new_adjustment:.1f}GB (was +{current_adjustment:.1f}GB)\n"
            f"New max VRAM: {self.max_vram_gb:.1f}GB\n"
            f"Stderr snippet: {stderr_output[:500] if stderr_output else 'N/A'}...\n"
            f"{'='*60}\n"
        )
        safe_write_to_file(self.oom_log, log_entry)

        print(f"\n!!! OOM DETECTED for {job} - VRAM estimate increased, cooldown active !!!")

    def _is_oom_error(self, stderr: str, returncode: int) -> bool:
        """Check if the error was an OOM."""
        if stderr is None:
            return False
        stderr_lower = stderr.lower()
        oom_indicators = [
            'out of memory',
            'cuda out of memory',
            'cudnn error',
            'cuda error',
            'memory allocation',
            'oom',
            'cannot allocate',
            'runtime error',  # Often accompanies CUDA OOM
        ]
        return any(indicator in stderr_lower for indicator in oom_indicators)

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

        vram = self._get_adjusted_vram(model_size, batch_size)
        job = Job(cmd=cmd, model_size=model_size, batch_size=batch_size, vram_gb=vram)
        self.jobs_pending.append(job)
        print(f"Added job: {job}")

    def _run_job(self, job: Job):
        """Run a single job and track completion."""
        cmd_str = ' '.join(job.cmd)
        print(f"\n>>> STARTING: {job} | Current VRAM: {self.current_vram:.1f}GB")

        try:
            # Capture stderr to detect OOM
            result = subprocess.run(
                job.cmd,
                capture_output=True,
                text=True
            )

            # Print stdout in real-time fashion (after completion)
            if result.stdout:
                print(result.stdout)

            if result.returncode != 0:
                # Check if it was an OOM
                is_oom = self._is_oom_error(result.stderr, result.returncode)

                if is_oom:
                    self._record_oom(job, result.stderr)

                    # Retry with increased VRAM estimate if we haven't exceeded retries
                    if job.retries < job.max_retries:
                        with self.lock:
                            job.retries += 1
                            job.vram_gb = self._get_adjusted_vram(job.model_size, job.batch_size)
                            self.jobs_pending.append(job)
                        print(f">>> REQUEUED {job} for retry #{job.retries} with new VRAM estimate")
                    else:
                        safe_write_to_file(self.fail_log, f"OOM (max retries): {cmd_str}\n")
                        with self.lock:
                            self.failed += 1
                else:
                    # Non-OOM failure
                    safe_write_to_file(self.fail_log, f"FAILED (rc={result.returncode}): {cmd_str}\nStderr: {result.stderr[:200] if result.stderr else 'N/A'}\n")
                    with self.lock:
                        self.failed += 1

                # Print stderr for debugging
                if result.stderr:
                    print(f"STDERR: {result.stderr[:500]}")
            else:
                with self.lock:
                    self.completed += 1

        except Exception as e:
            safe_write_to_file(self.fail_log, f"EXCEPTION: {cmd_str} - {e}\n")
            with self.lock:
                self.failed += 1

        with self.lock:
            self.current_vram -= job.vram_gb
            self.jobs_running = [(j, p, t) for j, p, t in self.jobs_running if j != job]

        print(f"\n<<< FINISHED: {job} | VRAM freed: {job.vram_gb}GB | Remaining: {self.current_vram:.1f}GB")

    def _try_start_jobs(self):
        """Try to start as many pending jobs as VRAM allows."""
        # Check OOM cooldown
        if time.time() < self.oom_cooldown_until:
            return

        with self.lock:
            # Sort by VRAM (largest first) to pack efficiently
            self.jobs_pending.sort(key=lambda j: -j.vram_gb)

            started = []
            for job in self.jobs_pending:
                # Update job's VRAM estimate in case it changed due to OOM
                job.vram_gb = self._get_adjusted_vram(job.model_size, job.batch_size)

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
        print(f"OOM log: {self.oom_log}")
        print(f"{'='*60}\n")

        while self.jobs_pending or self.jobs_running:
            self._try_start_jobs()

            # Wait a bit before checking again
            time.sleep(1)

            # Show status periodically
            with self.lock:
                pending = len(self.jobs_pending)
                running = len(self.jobs_running)
                cooldown_active = time.time() < self.oom_cooldown_until
                if running > 0 or cooldown_active:
                    running_info = ", ".join([f"{j.model_size}{j.batch_size}" for j, _, _ in self.jobs_running])
                    cooldown_str = " [COOLDOWN]" if cooldown_active else ""
                    print(f"\rRunning: [{running_info}] | VRAM: {self.current_vram:.1f}/{self.max_vram_gb}GB | Pending: {pending} | Done: {self.completed} | OOMs: {self.oom_count}{cooldown_str}    ", end="", flush=True)

        # Final summary
        summary = (
            f"\n\n{'='*60}\n"
            f"Queue complete!\n"
            f"  Completed: {self.completed}\n"
            f"  Failed: {self.failed}\n"
            f"  OOM events: {self.oom_count}\n"
            f"  Final max VRAM: {self.max_vram_gb}GB\n"
            f"  VRAM adjustments: {dict(self.vram_adjustments)}\n"
            f"{'='*60}\n"
        )
        print(summary)
        safe_write_to_file(self.oom_log, summary)


def main():
    parser = argparse.ArgumentParser(description='GPU-aware job queue')
    parser.add_argument('--max-vram', type=float, default=75.0,
                        help='Max VRAM to use in GB (default: 75, leaving 5GB headroom on 80GB)')
    parser.add_argument('--fail-log', type=str, default='queue_errors.txt',
                        help='File to log failed jobs')
    parser.add_argument('--oom-log', type=str, default='oom_events.txt',
                        help='File to log OOM events')
    parser.add_argument('--jobs-file', type=str, help='File containing jobs (one per line)')
    parser.add_argument('--model', type=str, choices=['yolov8', 'yolo11'], default='yolov8',
                        help='Model family to use')
    parser.add_argument('--skip-stac', action='store_true', default=True,
                        help='Skip STAC (default: True)')
    parser.add_argument('--safety-margin', type=float, default=5.0,
                        help='VRAM safety margin to subtract after OOM (default: 5GB)')
    args = parser.parse_args()

    queue = GPUQueue(
        max_vram_gb=args.max_vram,
        fail_log=args.fail_log,
        oom_log=args.oom_log,
        safety_margin=args.safety_margin
    )

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
                    'python3', 'dispatcher.py', 'stac.py', 'url_list.txt',
                    f'--model_name={model_name}',
                    f'--skip_stac=True',
                    f'--batch={batch}'
                ]
                queue.add_job(cmd)

    queue.run()


if __name__ == '__main__':
    main()
