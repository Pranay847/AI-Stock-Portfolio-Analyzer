"""Background portfolio analysis.

analyze_portfolio() makes an LLM round trip per position and is bounded by a
deadline of minutes, too long to hold an HTTP request open behind most hosting
proxies. So the API starts a job and the client polls it; progress comes from
analyze_portfolio's own progress_callback.

Each job gets a *private* analyzer. analyze_portfolio reads self.portfolio and
writes self.last_analysis_notes, so a shared instance would let concurrent jobs
overwrite each other's holdings and results.
"""

import logging
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from api.serialize import jsonable

logger = logging.getLogger(__name__)


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

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

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
    def __init__(self, max_concurrent: int, analyzer_factory: Callable[[], Any]):
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_concurrent), thread_name_prefix="analysis"
        )
        self._analyzer_factory = analyzer_factory

    def start(self, positions: list[dict]) -> AnalysisJob:
        job = AnalysisJob(id=secrets.token_urlsafe(9), total=len(positions))
        # Copy the holdings: the session may be cleared while the job runs.
        self._executor.submit(self._run, job, [dict(p) for p in positions])
        return job

    def _run(self, job: AnalysisJob, positions: list[dict]) -> None:
        job._update(status="running")
        try:
            analyzer = self._analyzer_factory()
            if analyzer is None:
                job._update(status="error", error="The analysis engine failed to initialise.")
                return
            analyzer.portfolio = positions

            def progress(completed: int, total: int, symbol: str) -> None:
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
        except Exception as e:
            logger.exception("portfolio analysis failed")
            job._update(status="error", error=f"{type(e).__name__}: {e}")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
