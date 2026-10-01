"""Background job management for the web application.

Jobs run in a bounded thread pool. Their state is persisted to
``<data_dir>/jobs/<id>/job.json`` (plus the input, output, review PDF and QA
report), so finished jobs survive restarts until they expire.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import pymupdf

from pt2en.config import Settings
from pt2en.errors import InvalidDocumentError, JobCancelled, PipelineError
from pt2en.ingestion.loader import looks_like_pdf
from pt2en.pipeline.options import JobOptions
from pt2en.pipeline.orchestrator import TranslationPipeline
from pt2en.pipeline.progress import ProgressEvent

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


@dataclass
class Job:
    id: str
    filename: str
    created: float
    options: dict
    page_count: int = 0
    status: JobStatus = JobStatus.QUEUED
    progress: dict = field(default_factory=dict)
    error: Optional[str] = None
    error_detail: Optional[str] = None
    summary: Optional[dict] = None
    finished: Optional[float] = None
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data


class JobManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.jobs_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, settings.max_concurrent_jobs),
            thread_name_prefix="pt2en-job",
        )
        self.jobs: dict[str, Job] = {}
        self.cancel_events: dict[str, threading.Event] = {}
        self.subscribers: dict[
            str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]
        ] = {}
        self._lock = threading.RLock()
        self._load_existing()

    # ------------------------------------------------------------ storage
    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def path(self, job_id: str, name: str) -> Path:
        return self.job_dir(job_id) / name

    def _save(self, job: Job) -> None:
        d = self.job_dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "job.json.tmp"
        tmp.write_text(json.dumps(job.to_dict(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(d / "job.json")

    def _load_existing(self) -> None:
        for jf in self.root.glob("*/job.json"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                job = Job(**{**data, "status": JobStatus(data["status"])})
                if job.status not in TERMINAL:
                    # The process stopped while this job was running.
                    job.status = JobStatus.FAILED
                    job.error = (
                        "The server restarted while this document was being processed."
                    )
                self.jobs[job.id] = job
            except Exception as exc:  # pragma: no cover - corrupt state
                log.warning("ignoring job state %s: %s", jf, exc)
        self.cleanup()

    def cleanup(self) -> int:
        ttl = self.settings.job_ttl_hours * 3600
        removed = 0
        now = time.time()
        with self._lock:
            for job_id, job in list(self.jobs.items()):
                if job.status in TERMINAL and now - (job.finished or job.created) > ttl:
                    self.delete(job_id)
                    removed += 1
        return removed

    # ------------------------------------------------------------- create
    def create(self, filename: str, data: bytes, options: JobOptions) -> Job:
        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise InvalidDocumentError(
                f"The file is larger than {self.settings.max_upload_mb} MB."
            )
        if not looks_like_pdf(data):
            raise InvalidDocumentError("The uploaded file is not a PDF document.")
        try:
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                if doc.needs_pass and not doc.authenticate(""):
                    raise InvalidDocumentError("The PDF is password-protected.")
                pages = doc.page_count
        except InvalidDocumentError:
            raise
        except Exception as exc:
            raise InvalidDocumentError(
                "The PDF could not be opened; it may be corrupted."
            ) from exc
        if pages == 0:
            raise InvalidDocumentError("The PDF has no pages.")
        if pages > self.settings.max_pages:
            raise InvalidDocumentError(
                f"The PDF has {pages} pages; the limit is {self.settings.max_pages}."
            )

        job = Job(
            id=uuid.uuid4().hex,
            filename=_safe_name(filename),
            created=time.time(),
            options=options.model_dump(mode="json"),
            page_count=pages,
        )
        d = self.job_dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "input.pdf").write_bytes(data)
        with self._lock:
            self.jobs[job.id] = job
            self.cancel_events[job.id] = threading.Event()
        self._save(job)
        self.executor.submit(self._run, job.id)
        return job

    # ---------------------------------------------------------------- run
    def _run(self, job_id: str) -> None:
        job = self.jobs[job_id]
        cancel = self.cancel_events.setdefault(job_id, threading.Event())
        if cancel.is_set():
            self._finish(job, JobStatus.CANCELLED)
            return
        job.status = JobStatus.RUNNING
        self._save(job)
        options = JobOptions(**job.options)

        def on_progress(ev: ProgressEvent) -> None:
            data = ev.to_dict()
            job.progress = data
            if (
                ev.message
                and (not job.log or job.log[-1] != ev.message)
                and ev.stage_progress in (0.0, 1.0)
            ):
                job.log.append(ev.message)
                job.log = job.log[-50:]
            self._publish(job_id, {"type": "progress", "job": self._public(job)})

        try:
            pipeline = TranslationPipeline(
                self.settings, options, progress=on_progress, cancel_event=cancel
            )
            result = pipeline.run(self.path(job_id, "input.pdf").read_bytes())
            self.path(job_id, "output.pdf").write_bytes(result.output_pdf)
            self.path(job_id, "review.pdf").write_bytes(result.review_pdf)
            self.path(job_id, "report.json").write_text(
                json.dumps(result.report, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            job.summary = {**result.report["summary"], "stats": result.report["stats"]}
            self._finish(job, JobStatus.COMPLETED)
        except JobCancelled:
            self._finish(job, JobStatus.CANCELLED)
        except PipelineError as exc:
            job.error = exc.message
            job.error_detail = exc.detail or None
            self._finish(job, JobStatus.FAILED)
        except Exception as exc:  # pragma: no cover - unexpected failure
            log.exception("job %s failed", job_id)
            job.error = "An unexpected error occurred while processing the document."
            job.error_detail = str(exc)[:500]
            self._finish(job, JobStatus.FAILED)

    def _finish(self, job: Job, status: JobStatus) -> None:
        job.status = status
        job.finished = time.time()
        if status == JobStatus.COMPLETED:
            job.progress = {
                **job.progress,
                "overall": 1.0,
                "stage": "done",
                "stage_label": "Done",
                "eta_seconds": 0,
            }
        self._save(job)
        self._publish(job.id, {"type": "status", "job": self._public(job)})

    # ------------------------------------------------------------ public
    def _public(self, job: Job) -> dict:
        data = job.to_dict()
        data["files"] = {
            name: self.path(job.id, f"{name}.pdf").exists()
            for name in ("output", "review")
        }
        return data

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def public(self, job_id: str) -> Optional[dict]:
        job = self.get(job_id)
        return self._public(job) if job else None

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job or job.status in TERMINAL:
            return False
        self.cancel_events.setdefault(job_id, threading.Event()).set()
        if job.status == JobStatus.QUEUED:
            self._finish(job, JobStatus.CANCELLED)
        return True

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self.jobs.pop(job_id, None)
            if job and job.status not in TERMINAL:
                self.cancel_events.get(job_id, threading.Event()).set()
            self.cancel_events.pop(job_id, None)
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        return job is not None

    # ------------------------------------------------------- subscriptions
    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        loop = asyncio.get_running_loop()
        with self._lock:
            self.subscribers.setdefault(job_id, []).append((loop, queue))
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            subs = self.subscribers.get(job_id, [])
            self.subscribers[job_id] = [(lp, q) for lp, q in subs if q is not queue]

    def _publish(self, job_id: str, message: dict) -> None:
        with self._lock:
            subs = list(self.subscribers.get(job_id, []))
        for loop, queue in subs:
            try:
                loop.call_soon_threadsafe(_offer, queue, message)
            except RuntimeError:  # loop closed
                self.unsubscribe(job_id, queue)

    def shutdown(self) -> None:
        for ev in self.cancel_events.values():
            ev.set()
        self.executor.shutdown(wait=False, cancel_futures=True)


def _offer(queue: asyncio.Queue, message: dict) -> None:
    try:
        queue.put_nowait(message)
    except asyncio.QueueFull:  # pragma: no cover - slow consumer
        pass


def _safe_name(name: str) -> str:
    name = Path(name or "document.pdf").name
    cleaned = (
        "".join(c for c in name if c.isalnum() or c in " ._-()").strip()
        or "document.pdf"
    )
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned[:150]
