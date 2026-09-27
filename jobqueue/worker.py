"""Phase 2: a process pool that drains a :class:`JobQueue`.

The core (``core.py``) owns *nothing* about execution — it only hands out
claimable jobs and records their outcome. This module supplies the executor:
a fixed set of worker *processes*, each running at most one job at a time, with
graceful (SIGTERM/SIGINT) and hard-deadline shutdown.

Process pool, not thread pool
-----------------------------
``Pool(n)`` starts ``n`` separate processes (start method = ``fork``). This is
deliberate:

* A handler runs in its own process with its own memory — a crashing or
  ``exit()``-calling handler cannot take the worker (or other jobs) down.
* A handler that *hangs* can be force-terminated on shutdown (see
  ``_execute_job``): you cannot kill a thread, but you can ``terminate()`` a
  process. This is what makes "a handler must not block shutdown forever"
  actually true.
* ``core.py`` requires one SQLite connection per process — a shared connection
  is unsafe across processes. Each worker therefore opens its own short-lived
  connection by reconstructing the queue with :meth:`JobQueue._shared_config`.

Each job's *handler execution* happens in a short-lived child of the worker.
The worker ``join()``-s that child; if it does not return before the shutdown
deadline it is terminated and the job is left ``running`` so its lease expires
and another worker retries it (at-most-once replay).

No busy-wait
------------
When a worker finds nothing to claim it must wait for work without spinning the
CPU. SQLite gives us no portable cross-process "job arrived" primitive that
does not require external infrastructure, so an idle worker polls the queue with
a short, configurable backoff (``poll_interval``). That is *not* a spin: between
polls the thread sleeps, so average CPU stays near 0 while idle. The trade-off is
wake-up latency bounded by ``poll_interval``. Note that ``core.Queue.dequeue``
already uses SQLite's ``busy_timeout`` under write contention, so workers do not
hammer the DB with a thundering herd when many contend for the single write lock
— they block inside SQLite instead of looping. A hookable wake event could reduce
latency later without changing the polling fallback.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import time
import traceback
from typing import Callable, Optional

from .core import JobQueue

# handler(job_payload: dict) -> result dict
Handler = Callable[[dict], dict]

__all__ = ["Pool", "Worker"]

logger = None  # replaced in _ensure_logger()


def _ensure_logger():
    global logger
    if logger is None:
        import logging
        logger = logging.getLogger("jobqueue.worker")
    return logger


# ----------------------------------------------------------------------------
# Process entry points (module-level so they work under fork AND spawn)
# ----------------------------------------------------------------------------

def _child_entry(
    db_path: str, worker_id: str, job_id: int, handler: Optional[Handler], payload,
) -> None:
    """Runs inside the short-lived child that actually executes a handler.

    Opens its own connection (must not reuse the parent's sqlite handle).
    Reports the outcome through the queue: success -> complete(), any raise ->
    fail() which drives core's retry/backoff. Never returns an error upward —
    it always reports to the queue, or aborts via terminate() when the hard
    shutdown deadline is exceeded (handled by the parent's join()).
    """
    q = JobQueue(db_path)
    try:
        if handler is None:
            raise RuntimeError(f"no handler registered for job_type={job_id!r}")
        result = handler(payload)
    except BaseException:  # noqa: BLE001 — every failure path must reach fail()
        q.fail(job_id, traceback.format_exc())
    else:
        q.complete(job_id, result)


def _serve_worker(
    db_path: str,
    worker_id: str,
    queue_config: dict,
    handlers: dict[str, Handler],
    opts: dict,
) -> None:
    """Entry point for one worker process. Runs until shut down."""
    log = _ensure_logger()

    # Put ourselves in our own session/process group. We cannot ask multiprocessing
    # to do this (no Process kwarg supports it), so we setsid() in the child.
    # This lets a hard teardown reap this worker's handler children in one shot
    # with os.killpg(worker_pid) — a killed worker must not leave an orphan that
    # simply leaks. (Graceful shutdown never does this; in-flight work finishes.)
    try:
        os.setsid()
    except OSError:  # pragma: no cover - already a group leader / platform edge
        pass

    poll_interval = opts["poll_interval"]
    shutdown_timeout = opts["shutdown_timeout"]

    # One connection per process (never share the parent's sqlite handle).
    queue = JobQueue(db_path, **queue_config)

    stop = threading.Event()
    deadline: dict[str, Optional[float]] = {"t": None}

    def _on_signal(signum: int, frame) -> None:
        # Graceful shutdown: stop claiming *new* jobs. In-flight work is allowed
        # to finish (up to the hard deadline enforced in _execute_job).
        log.info("worker %s got signal %d — graceful shutdown", worker_id, signum)
        stop.set()
        deadline["t"] = time.monotonic()

    # Install after our connection is up, so a handler running inside a child can
    # still complete()/fail() normally. SIGINT == interactive Ctrl-C; SIGTERM ==
    # orchestrator `kill`.
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    log.info("worker %s up (db=%s)", worker_id, db_path)

    while not stop.is_set():
        job = queue.dequeue(worker_id)
        if job is None:
            # Idle: sleep a short backoff instead of spinning the CPU on SQLite.
            # Never sleep past an outstanding shutdown deadline so we still react
            # to `kill` promptly while idle.
            if deadline["t"] is None:
                stop.wait(poll_interval)
            else:
                stop.wait(max(0.0, min(poll_interval, deadline["t"] - time.monotonic())))
            continue
        _execute_job(queue, worker_id, job, handlers.get(job.job_type), shutdown_timeout, deadline)

    log.info("worker %s exiting", worker_id)
    queue.close()


def _execute_job(
    queue: JobQueue,
    worker_id: str,
    job,
    handler: Optional[Handler],
    shutdown_timeout: float,
    deadline: dict[str, Optional[float]],
) -> None:
    """Run one claimed job in a child process and report its outcome.

    * returned normally            -> child called complete()
    * handler raised              -> child called fail()  (core retries per backoff)
    * still running at the hard   -> terminated; job left 'running' so its lease
      shutdown deadline             expires and another worker retries it (logged).
    """
    log = _ensure_logger()
    ctx = mp.get_context("fork")

    child = ctx.Process(
        target=_child_entry,
        args=(queue.db_path, worker_id, job.id, handler, job.payload),
        name=f"{worker_id}-job-{job.id}",
    )
    child.start()

    # How long to wait for this in-flight child: generous while working normally;
    # bounded by the shutdown deadline once a graceful stop has begun. The bound
    # is what makes "a hanging handler can't block shutdown forever" hold.
    if deadline["t"] is None:
        remaining: Optional[float] = None
    else:
        remaining = max(0.0, deadline["t"] + shutdown_timeout - time.monotonic())

    child.join(remaining)
    if child.is_alive():
        # Hard deadline exceeded. Terminate the handler and LEAVE the job running
        # — its lease will expire naturally and another worker will retry it
        # (at most once). We do not call fail(); that is deliberate so nobody has
        # to know this attempt never completed.
        log.warning(
            "worker %s hard-deadline (%gs) exceeded for job %d — terminating handler, leaving lease",
            worker_id, shutdown_timeout, job.id,
        )
        child.terminate()
        child.join(5.0)  # give terminate() a moment to finish killing it
        if child.is_alive():  # pragma: no cover - stubborn process refused to die
            child.kill()
            child.join(2.0)
        return

    # Child finished on its own and already reported complete()/fail(). The only
    # remaining failure mode is the child dying before reporting (e.g. OOM'd
    # mid-handler); surface that so a stuck job isn't silent.
    if child.exitcode not in (0, None):
        log.error(
            "worker %s handler for job %d aborted unexpectedly (exit %s); job left running",
            worker_id, job.id, child.exitcode,
        )


class Worker:
    """A single worker process running one job at a time.

    Create via :meth:`Pool.start`; users normally don't instantiate directly."""

    def __init__(self, queue: JobQueue, worker_id: str, handlers: dict, opts: dict) -> None:
        self._queue = queue
        self.worker_id = worker_id
        self._opts = opts
        self._handlers = handlers
        self.ctx = mp.get_context("fork")
        self._proc: Optional[mp.Process] = None

    def start(self) -> "Worker":
        # Reconstruct the queue from its *public* attributes so we never touch
        # core.py for this. Each child opens a fresh connection (fork inherits the
        # config, not the sqlite handle); clocks are passed through as callables.
        cfg = {
            "max_attempts": getattr(self._queue, "max_attempts", 3),
            "backoff_base": getattr(self._queue, "backoff_base", 0.5),
            "backoff_cap": getattr(self._queue, "backoff_cap", 3600.0),
            "visibility_timeout": getattr(self._queue, "visibility_timeout", 30.0),
            "busy_timeout_ms": getattr(self._queue, "busy_timeout_ms", 5000),
            "clock": self._queue._wall_clock,
            "lease_clock": self._queue._lease_clock,
        }
        self._proc = self.ctx.Process(
            target=_serve_worker,
            args=(
                self._queue.db_path,
                self.worker_id,
                cfg,
                dict(self._handlers),
                self._opts,
            ),
            name=f"worker-{self.worker_id}",
            daemon=False,
        )
        self._proc.start()
        return self

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None

    @property
    def alive(self) -> bool:
        return bool(self._proc and self._proc.is_alive())

    def join(self, timeout: Optional[float] = None) -> bool:
        """Wait up to ``timeout`` for the worker to exit. Returns True if it did."""
        if not self._proc:
            return False
        self._proc.join(timeout)
        return not self._proc.is_alive()

    def stop(self) -> None:
        """Ask this worker to shut down gracefully (SIGTERM)."""
        if self._proc and self._proc.is_alive():
            try:
                self._proc.terminate()  # SIGTERM for the fork child
            except ProcessLookupError:
                pass

    def force_stop(self) -> None:
        """Hard kill this worker AND any handler child it spawned.

        Only used for last-resort teardown (join timed out). A graceful stop
        never reaches here — in-flight work is meant to finish. Putting the
        worker in its own session lets us reap its orphaned children in one shot
        with ``killpg`` instead of leaking them.
        """
        if not self._proc or not self._proc.is_alive():
            return
        try:
            os.killpg(self._proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass


class Pool:
    """A fixed-size pool of worker processes draining a shared :class:`JobQueue`."""

    def __init__(
        self,
        queue: JobQueue,
        handlers: dict[str, Handler],
        n: int = 4,
        *,
        shutdown_timeout: float = 30.0,
        poll_interval: float = 0.05,
    ) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.queue = queue
        self.handlers = dict(handlers)
        self.n = n
        self.shutdown_timeout = shutdown_timeout
        self.poll_interval = poll_interval
        self._opts = {
            "shutdown_timeout": shutdown_timeout,
            "poll_interval": poll_interval,
        }
        self.workers: list[Worker] = [
            Worker(queue, f"w{i}", self.handlers, self._opts) for i in range(n)
        ]

    def start(self) -> "Pool":
        for w in self.workers:
            w.start()
        _ensure_logger().info(
            "pool started with %d workers (db=%s)", self.n, self.queue.db_path
        )
        return self

    def join(self, timeout: Optional[float] = None) -> int:
        """Wait for all workers to exit cleanly. Returns the number that did."""
        deadline = time.monotonic() + timeout if timeout is not None else None
        exited = 0
        for w in self.workers:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if w.join(remaining):
                exited += 1
            elif time.monotonic() >= deadline:  # last one hit the outer deadline
                break
        still = [w for w in self.workers if w.alive]
        if still:
            _ensure_logger().warning(
                "pool join timed out with %d worker(s) alive — force stopping", len(still)
            )
            for w in still:
                w.force_stop()
        return exited

    def shutdown(self, timeout: Optional[float] = None) -> int:
        """Gracefully stop the pool: SIGTERM each worker, then wait."""
        for w in self.workers:
            w.stop()
        return self.join(timeout)

    def __enter__(self) -> "Pool":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.shutdown()
