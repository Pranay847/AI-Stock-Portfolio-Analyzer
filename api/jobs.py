"""Background portfolio analysis.

analyze_portfolio() makes an LLM round trip per position and is bounded by a
deadline of minutes, too long to hold an HTTP request open behind most hosting
proxies. So the API starts a job and the client polls it; progress comes from
analyze_portfolio's own progress_callback.

Each job gets a *private* analyzer. analyze_portfolio reads self.portfolio and
writes self.last_analysis_notes, so a shared instance would let concurrent jobs
overwrite each other's holdings and results.

Jobs are cancellable and the queue is bounded. A job whose session has moved
on (disconnect, reload, re-sync) is cancelled rather than left running: it
would otherwise hold a worker for minutes, keep sending the user's holdings to
the LLM provider after they disconnected, and make everyone else wait.
"""

import logging
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from api.serialize import jsonable

logger = logging.getLogger(__name__)


class RunnerBusy(Exception):
    """Too many analyses are queued or running; the caller should retry later."""


class _Cancelled(Exception):
    """Raised from the progress callback to stop a cancelled job early."""


@dataclass
class AnalysisJob:
    id: str
    total: int
    status: str = "queued"  # queued -> running -> done | error
    completed: int = 0
    current_symbol: str | None = None
    results: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        """Stop the job as soon as it next checks: before it starts, or after
        any position finishes. Positions not yet started are never analyzed."""
        self._cancel.set()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "status": self.status,
                "total": self.total,
                "completed": self.completed,
                "current_symbol": self.current_symbol,
                "results": list(self.results),
                "notes": list(self.notes),
                "error": self.error,
            }

    def _update(self, **changes) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self, key, value)


class AnalysisRunner:
    def __init__(self, max_concurrent: int, analyzer_factory: Callable[[], Any],
                 max_pending: int | None = None):
        workers = max(1, max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="analysis")
        self._analyzer_factory = analyzer_factory
        # Queued + running. ThreadPoolExecutor bounds workers but not its queue,
        # so without this any client could queue analyses without limit.
        self._max_pending = max_pending or workers * 2
        self._pending = 0
        self._pending_lock = threading.Lock()

    def start(self, positions: list[dict]) -> AnalysisJob:
        """Queue an analysis, or raise RunnerBusy if the queue is full."""
        with self._pending_lock:
            if self._pending >= self._max_pending:
                raise RunnerBusy()
            self._pending += 1
        job = AnalysisJob(id=secrets.token_urlsafe(9), total=len(positions))
        try:
            # Copy the holdings: the session may be cleared while the job runs.
            self._executor.submit(self._run, job, [dict(p) for p in positions])
        except Exception:
            self._release()
            raise
        return job

    def _release(self) -> None:
        with self._pending_lock:
            self._pending -= 1

    def _run(self, job: AnalysisJob, positions: list[dict]) -> None:
        try:
            if job.cancelled:
                return
            job._update(status="running")
            analyzer = self._analyzer_factory()
            if analyzer is None:
                job._update(status="error", error="The analysis engine failed to initialise.")
                return
            if job.cancelled:
                return
            analyzer.portfolio = positions

            def progress(completed: int, total: int, symbol: str) -> None:
                # Raising here makes analyze_portfolio cancel every position
                # that has not started (its finally shuts its pool down with
                # cancel_futures=True).
                if job.cancelled:
                    raise _Cancelled()
                job._update(completed=completed, total=total, current_symbol=symbol)

            results = analyzer.analyze_portfolio(progress_callback=progress)
            notes = [str(n) for n in (getattr(analyzer, "last_analysis_notes", None) or [])]
            job._update(
                status="done",
                results=jsonable(results or []),
                notes=notes,
                completed=job.total,
                current_symbol=None,
            )
        except _Cancelled:
            job._update(status="error", error="Cancelled")
        except Exception as e:
            logger.exception("portfolio analysis failed")
            job._update(status="error", error=f"{type(e).__name__}: {e}")
        finally:
            self._release()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
