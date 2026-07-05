"""Job executor for Makalu API — file-rename based single-job-at-a-time queue."""

import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path

JOBS_DIR = Path("/panda-infra/makalu/jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

_CLI = Path(__file__).parent.parent / "cli.py"


# Job file extensions encode state:
#   {id}.json             → pending
#   {id}.json.processing  → running  (contains pid)
#   {id}.done             → finished (done or failed)
#   {id}.log              → subprocess stdout (written while running)
#   {id}.rc               → exit code (written by executor before finalising)


def _pending_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _processing_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json.processing"


def _done_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.done"


def _log_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.log"


def _rc_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.rc"


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.rename(path)


def _read_log_tail(job_id: str) -> list[str]:
    try:
        return _log_path(job_id).read_text().splitlines()[-200:]
    except OSError:
        return []


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pending_created_at(f: Path) -> float:
    try:
        return json.loads(f.read_text()).get("created_at", 0.0)
    except (json.JSONDecodeError, OSError):
        return 0.0


class JobExecutor:
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._jobs_lock = threading.Lock()
        self._executor_running = False
        self._executor_lock = threading.Lock()
        self._load_jobs()

    def _load_jobs(self) -> None:
        seen: set[str] = set()
        for pattern in ("*.json.processing", "*.json", "*.done"):
            for f in sorted(JOBS_DIR.glob(pattern)):
                try:
                    job = json.loads(f.read_text())
                    jid = job["id"]
                    if jid not in seen:
                        seen.add(jid)
                        self._jobs[jid] = job
                except (json.JSONDecodeError, KeyError, OSError):
                    pass

    def list_jobs(self) -> list[dict]:
        with self._jobs_lock:
            return sorted(
                self._jobs.values(), key=lambda j: j.get("created_at", 0), reverse=True
            )

    def get_job(self, job_id: str) -> dict | None:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def add(self, job_data: dict) -> None:
        """Save job_data to {job_id}.json and start the executor loop if not running."""
        job_id = job_data["id"]
        with self._jobs_lock:
            self._jobs[job_id] = job_data
        _pending_path(job_id).write_text(json.dumps(job_data, indent=2))
        self.kickoff()

    def kickoff(self) -> None:
        """Start the executor loop thread if not already running."""
        with self._executor_lock:
            if self._executor_running:
                return
            self._executor_running = True
        threading.Thread(target=self._executor_loop, daemon=True).start()

    # ── Internal executor ─────────────────────────────────────────────────────

    def _finalize_job(self, job_id: str, rc: int) -> None:
        proc_file = _processing_path(job_id)
        done_file = _done_path(job_id)
        try:
            job = json.loads(proc_file.read_text())
        except (OSError, ValueError):
            with self._jobs_lock:
                job = dict(self._jobs.get(job_id, {"id": job_id}))
        job["returncode"] = rc
        job["status"] = "done" if rc == 0 else "failed"
        job["finished_at"] = time.time()
        job["log"] = _read_log_tail(job_id)
        done_file.write_text(json.dumps(job, indent=2))
        proc_file.unlink(missing_ok=True)
        with self._jobs_lock:
            self._jobs[job_id] = job

    def _monitor_job(
        self, job_id: str, pid: int, proc: subprocess.Popen | None = None
    ) -> None:
        proc_file = _processing_path(job_id)
        rc_file = _rc_path(job_id)

        if proc is not None:
            # We own this process: use proc.poll() which reaps the child.
            # os.kill(pid, 0) returns True for zombies and would deadlock here.
            while proc.poll() is None:
                log = _read_log_tail(job_id)
                with self._jobs_lock:
                    if job_id in self._jobs:
                        self._jobs[job_id]["log"] = log
                try:
                    data = json.loads(proc_file.read_text())
                    data["log"] = log
                    _atomic_write(proc_file, json.dumps(data, indent=2))
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            rc = proc.returncode
            try:
                rc_file.write_text(str(rc))
            except OSError:
                pass
        else:
            # Reattach: not our child — cannot be zombie here.
            while _is_pid_alive(pid) and not rc_file.exists():
                log = _read_log_tail(job_id)
                with self._jobs_lock:
                    if job_id in self._jobs:
                        self._jobs[job_id]["log"] = log
                try:
                    data = json.loads(proc_file.read_text())
                    data["log"] = log
                    _atomic_write(proc_file, json.dumps(data, indent=2))
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            try:
                rc = int(rc_file.read_text().strip())
            except (OSError, ValueError):
                rc = -1

        self._finalize_job(job_id, rc)

    def _cleanup_old_jobs(self) -> None:
        cutoff = time.time() - 86400
        with self._jobs_lock:
            stale = [
                jid
                for jid, j in list(self._jobs.items())
                if j.get("status") in ("done", "failed")
                and j.get("finished_at", j.get("created_at", 0)) < cutoff
            ]
            for jid in stale:
                del self._jobs[jid]
                for path in (_done_path(jid), _log_path(jid), _rc_path(jid)):
                    path.unlink(missing_ok=True)

    def _executor_loop(self) -> None:
        self._cleanup_old_jobs()
        try:
            # Resume any job whose .processing file survived a restart.
            for proc_file in JOBS_DIR.glob("*.json.processing"):
                try:
                    job = json.loads(proc_file.read_text())
                    job_id = job["id"]
                except (json.JSONDecodeError, KeyError, OSError):
                    job_id = proc_file.name.split(".")[0]
                    self._finalize_job(job_id, rc=-1)
                    continue
                pid = job.get("pid")
                if pid and _is_pid_alive(pid):
                    self._monitor_job(job_id, pid)
                else:
                    self._finalize_job(job_id, rc=-1)

            # Pick up pending jobs one at a time.
            while True:
                pending = sorted(JOBS_DIR.glob("*.json"), key=_pending_created_at)
                if not pending:
                    break

                job_file = pending[0]
                try:
                    job = json.loads(job_file.read_text())
                except (json.JSONDecodeError, OSError):
                    job_file.unlink(missing_ok=True)
                    continue

                job_id = job["id"]
                proc_file = _processing_path(job_id)
                job_file.rename(proc_file)  # atomic: pending → processing

                job["status"] = "running"
                job["started_at"] = time.time()

                rc_file = _rc_path(job_id)
                # Wrap in bash so the subprocess writes the .rc file itself;
                # the rc survives an API restart during the job.
                shell_cmd = f"{shlex.join(job['cmd_args'])}; echo $? > {shlex.quote(str(rc_file))}"

                with open(_log_path(job_id), "w") as log_f:
                    proc = subprocess.Popen(
                        ["bash", "-c", shell_cmd],
                        cwd=str(_CLI.parent.parent),
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                    )

                job["pid"] = proc.pid
                _atomic_write(proc_file, json.dumps(job, indent=2))
                with self._jobs_lock:
                    self._jobs[job_id] = job

                self._monitor_job(job_id, proc.pid, proc=proc)

        finally:
            with self._executor_lock:
                self._executor_running = False
            if any(JOBS_DIR.glob("*.json")):
                self.kickoff()
