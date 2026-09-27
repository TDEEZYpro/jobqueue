"""Phase 2 tests: the process worker pool (Pool / Worker).

Deterministic by construction:
* No real wall-clock sleeps longer than 0.05s gate any assertion.
* The lease-expiry reclaim test drives time with a *process-shared* fake clock
  (``SharedLeaseClock``) — advancing it makes an expired lease reclaimable on the
  very next dequeue, so the test is race-free and needs no waiting.

Run: python -m unittest tests.test_worker -v
"""

import os
import signal
import sqlite3
import tempfile
import time
import unittest
from multiprocessing import get_context

from jobqueue import JobQueue, Pool, State

CTX = get_context("fork")


class SharedLeaseClock:
    """A monotonic-ish clock shared across forked processes via shared memory.

    core leases are measured on the lease_clock callable. By backing it with a
    multiprocessing Value (real shared memory), the *test process* can advance
    time deterministically and every worker observes the jump on its next
    dequeue — no sleeping, no race. ``__call__`` is what we hand to JobQueue."""

    def __init__(self, start: float = 1000.0):
        self._v = CTX.Value("d", start)
        self._lock = CTX.Lock()

    def __call__(self) -> float:
        with self._lock:
            return float(self._v.value)

    def advance(self, secs: float) -> None:
        with self._lock:
            self._v.value += secs


def _new_queue(path=None, **kwargs):
    if path is None:
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "queue.db")
    kwargs.setdefault("max_attempts", 3)
    return JobQueue(path, **kwargs)


def _read_job(path, job_id):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            "SELECT state, worker_id, attempts FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
    finally:
        con.close()
    return dict(row) if row else None


def _wait_until(predicate, timeout=5.0, interval=0.01):
    """Busy-free-ish wait with single sleeps <= 0.05s (allowed by the rules)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _nuke(pool):
    """Last-resort teardown: SIGKILL each worker's whole process group to reap
    any orphaned handler children, then reap the worker leaders themselves.

    killpg targets the *group*: each worker calls os.setsid() to own a process
    group, so its in-flight handler children share it and get reaped even after
    the worker itself is gone (a killed worker's child stays in that group until
    signalled)."""
    for w in pool.workers:
        if not w._proc:
            continue
        if w.pid:
            try:
                os.killpg(w.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        w._proc.join(0)  # reap the leader so we don't leave a zombie


class DrainTests(unittest.TestCase):
    """Smoke + exactly-once through the real process pool."""

    def test_pool_drains_every_job_exactly_once(self):
        path = os.path.join(tempfile.mkdtemp(), "queue.db")
        q = _new_queue(path)
        N = 60

        # Cross-process: children are forked, so they can't append to a parent
        # list. Exactly-once is verified from the DB (workers write results via
        # their own connections).
        def handle(payload):
            return {"echo": payload}

        ids = [q.enqueue("j", {"i": i}) for i in range(N)]
        pool = Pool(q, {"j": handle}, n=4, shutdown_timeout=10.0, poll_interval=0.02)
        try:
            pool.start()
            # let the pool drain; then signal a graceful stop
            self.assertTrue(_wait_until(lambda: q.count_by_state().get(State.DONE) == N, 15))
        finally:
            pool.shutdown(timeout=5)

        states = q.count_by_state()
        self.assertEqual(states.get(State.DONE), N)
        self.assertEqual(states.get(State.RUNNING), 0)
        # Completed jobs carry a stored result (jobs.result), proving real work
        # happened in worker processes and each job completed exactly once.
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT id FROM jobs WHERE state='done' AND result IS NOT NULL"
            ).fetchall()
        finally:
            con.close()
        self.assertEqual(len(rows), N)


class GracefulShutdownTests(unittest.TestCase):
    """Required: SIGTERM mid-job completes the in-flight job and exits clean."""

    def test_sigterm_mid_job_completes_and_exits_clean(self):
        started = CTX.Event()   # set by handler when it begins running
        release = CTX.Event()   # handler blocks here until we let it return

        def slow_job(payload):
            started.set()
            self.assertIn("go", payload)
            release.wait(5)
            return {"done": True}

        path = os.path.join(tempfile.mkdtemp(), "queue.db")
        q = _new_queue(path, visibility_timeout=30.0)
        pool = Pool(q, {"slow": slow_job}, n=1, shutdown_timeout=30.0, poll_interval=0.02)
        try:
            pool.start()
            jid = q.enqueue("slow", {"go": True})
            self.assertTrue(started.wait(5), "handler never started")

            # Simulate `kill -TERM <worker>`, but BEFORE the handler has returned.
            w0 = pool.workers[0]
            os.kill(w0.pid, signal.SIGTERM)

            # The in-flight job is still running when SIGTERM arrives. Graceful
            # shutdown must let it FINISH (not abort mid-execution), then exit.
            release.set()

            exited = pool.join(timeout=5)
        finally:
            if pool.workers and pool.workers[0].alive:
                pool.workers[0].force_stop()

        self.assertEqual(exited, 1, "worker did not exit cleanly")
        self.assertEqual(
            pool.workers[0]._proc.exitcode, 0, "worker exited with a non-zero status"
        )

        states = q.count_by_state()
        self.assertEqual(states.get(State.DONE), 1)
        self.assertEqual(states.get(State.RUNNING), 0, "in-flight job was abandoned!")
        self.assertEqual(states.get(State.FAILED), 0)


class ReclaimTests(unittest.TestCase):
    """Required: SIGKILL a worker -> its job is reclaimed via the lease by another
    worker."""

    def test_sigkill_leaves_job_reclaimable(self):
        sc = SharedLeaseClock(start=1000.0)
        path = os.path.join(tempfile.mkdtemp(), "queue.db")
        # Long visibility window; expiry is fully controlled by the fake clock so
        # this test needs no real waiting.
        q = _new_queue(path, visibility_timeout=100.0, lease_clock=sc, max_attempts=3)

        started = CTX.Event()
        never = CTX.Event()             # handler blocks here forever; teardown reaps it

        def hang_job(payload):
            # Runs long enough that we can kill the worker while it's in flight.
            started.set()
            never.wait(10)

        pool = Pool(q, {"hang": hang_job}, n=2, shutdown_timeout=30.0, poll_interval=0.02)
        try:
            pool.start()
            jid = q.enqueue("hang", {})
            self.assertTrue(started.wait(5), "first worker never started")

            # Hard-crash worker w0 (the one holding the job). Its lease is still
            # live -> job stays 'running' with worker_id='w0'.
            w0 = pool.workers[0]
            os.kill(w0.pid, signal.SIGKILL)

            owner = _read_job(path, jid)
            self.assertEqual(owner["state"], State.RUNNING)
            self.assertEqual(owner["worker_id"], "w0")

            # Advance the (shared) lease clock past expiry -> job reclaimable.
            sc.advance(100.5)

            # Another live worker must pick it up: worker_id flips, attempts bumps.
            got_it = _wait_until(
                lambda: _read_job(path, jid)["worker_id"] not in (None, "w0"),
                timeout=5,
            )
            self.assertTrue(got_it, "no other worker reclaimed the abandoned job")
            after = _read_job(path, jid)
            self.assertEqual(after["worker_id"], "w1")
            self.assertEqual(after["attempts"], 2)
        finally:
            _nuke(pool)


class RetryTests(unittest.TestCase):
    """Required: a raising handler retries per core's backoff, then goes dead."""

    def test_raises_retries_then_dies(self):
        path = os.path.join(tempfile.mkdtemp(), "queue.db")
        # Tiny base so real backoff elapses quickly; still exercises the doubling.
        q = _new_queue(path, max_attempts=3, backoff_base=0.01, backoff_cap=0.5)

        def boom(payload):
            raise RuntimeError("kaboom")

        pool = Pool(q, {"boom": boom}, n=2, shutdown_timeout=10.0, poll_interval=0.02)
        try:
            pool.start()
            jid = q.enqueue("boom", {})

            went_dead = _wait_until(
                lambda: _read_job(path, jid)["state"] == State.DEAD, timeout=15
            )
        finally:
            pool.shutdown(timeout=5)

        states = q.count_by_state()
        self.assertTrue(went_dead, "job never reached DEAD after exhausting retries")
        self.assertEqual(states.get(State.DEAD), 1)
        self.assertEqual(states.get(State.PENDING), 0)
        self.assertEqual(states.get(State.RUNNING), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
