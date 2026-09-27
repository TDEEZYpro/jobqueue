"""Durable SQLite-backed job queue — Phase 1 core.

Standard library only (Python 3.12). No external dependencies.

Design notes
------------
* Each public call opens its own short-lived ``sqlite3`` connection. A single
  shared connection would be unsafe to use across separate OS processes, and
  SQLite serialises writes at the database level — so per-connection connections
  are fine as long as we let SQLite enforce ordering (see atomic dequeue below).

* WAL mode is forced on construction (see why in `_configure`). In brief: with
  a single writer and many readers across processes, WAL lets concurrent
  readers see one consistent snapshot without blocking the writer and vice
  versa. Without WAL the very SELECT+UPDATE we do inside a dequeue transaction
  would thrash against writers/other readers with SQLITE_BUSY.

* Concurrency / exactly-once dispatch is guaranteed by SQLite's own locking,
  never a Python lock: ``dequeue`` runs ``BEGIN IMMEDIATE`` which acquires the
  write lock *before* the SELECT, so two processes can never both read the same
  candidate row and both mark it running. The second process blocks (up to
  busy_timeout) until the first commits, then sees the job already claimed.

No sweeper thread is used. Expired leases are reclaimed lazily inside dequeue:
a background UPDATE flips any ``running`` job past its visibility timeout back to
``pending``, and the very same SELECT can then pick it up.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional

__all__ = ["JobQueue", "Job", "State"]


class State:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DEAD = "dead"


# Backoff base (seconds) and cap. Backoff for attempt n is base * 2**(n-1).
DEFAULT_BACKOFF_BASE = 0.5
DEFAULT_BACKOFF_CAP = 3600.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_VISIBILITY_TIMEOUT = 30.0


@dataclass
class Job:
    id: int
    job_type: str
    payload: dict
    priority: int
    attempts: int
    worker_id: Optional[str] = None
    lease_expires: Optional[float] = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Job(id={self.id}, job_type={self.job_type!r}, "
            f"priority={self.priority}, attempts={self.attempts})"
        )


class JobQueue:
    """A durable queue backed by a single SQLite database."""

    def __init__(
        self,
        db_path: str | os.PathLike[str],
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_cap: float = DEFAULT_BACKOFF_CAP,
        visibility_timeout: float = DEFAULT_VISIBILITY_TIMEOUT,
        busy_timeout_ms: int = 5000,
        clock=time.time,            # wall clock → run_at scheduling (evaluated each call)
        lease_clock=time.monotonic,  # monotonic → lease lifetime (evaluated each call)
    ) -> None:
        self.db_path = os.fspath(db_path)
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_cap = min(backoff_cap, backoff_base * 64.0)
        self.visibility_timeout = visibility_timeout
        self.busy_timeout_ms = busy_timeout_ms
        # NOTE: we pass the functions themselves (time.time / time.monotonic),
        # NOT their call results. A bare `time.monotonic` is a callable invoked
        # per-read by `_lease_now`; passing `time.monotonic()` would bind clock
        # to a single frozen value captured at import/construct time.
        self._wall_clock = clock  # time.time(): schedules run_at bookkeeping
        self._lease_clock = lease_clock  # time.monotonic(): bounds lease lifetime
        # A lock only guards our in-process state machine (the single connection
        # we own), never SQLite itself — cross-process safety comes from SQLite.
        self._conn_lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._boot_synced = False  # one-time per-boot lease invalidation (see _sync_boot_ref)
        self._ensure_connected().row_factory = sqlite3.Row
        self._create_schema()
        self._sync_boot_ref()

    # ------------------------------------------------------------------ conn
    def _ensure_connected(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.busy_timeout_ms / 1000.0,
                isolation_level=None,  # autocommit; we drive transactions explicitly
            )
            conn.execute("PRAGMA foreign_keys = ON")
            # --- WHY WAL MATTERS HERE --------------------------------------
            # WAL lets readers and a writer operate concurrently without
            # blocking each other. Our dequeue takes the write lock for the
            # whole SELECT+UPDATE, so under rollback journal mode every polling
            # reader would block on that writer and writers would block on each
            # other, exploding into SQLITE_BUSY. WAL keeps reads fast and makes
            # the busy_timeout we set above meaningful.
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")  # aligned with WAL: safe & faster
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    __enter__ = lambda self: self
    def __exit__(self, *exc) -> None:
        self.close()

    # --------------------------------------------------------------- schema
    def _create_schema(self) -> None:
        self._ensure_connected().executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type        TEXT    NOT NULL,
                payload         BLOB    NOT NULL,
                priority        INTEGER NOT NULL DEFAULT 0,
                state           TEXT    NOT NULL,
                run_at          REAL    NOT NULL,
                attempts        INTEGER NOT NULL DEFAULT 0,
                max_attempts    INTEGER NOT NULL DEFAULT 3,
                worker_id       TEXT,
                lease_expires   REAL,
                error           TEXT,
                result          BLOB,
                created_at      REAL    NOT NULL,
                updated_at      REAL    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_lookup
                ON jobs (state, run_at, priority DESC, id);
            -- One-row store of the current boot's identity (see _sync_boot_ref).
            -- ``value`` is TEXT: it holds a per-boot identifier (boot_id) on Linux.
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    # --------------------------------------------------------------- helpers
    # Two clocks, kept deliberately separate. `run_at` scheduling (enqueue /
    # the run_at<=now gate in dequeue / fail backoff / created_at/updated_at)
    # is genuine wall-clock and uses time.time(). Lease lifetime, by contrast,
    # must be measured against a monotonic clock so NTP/sleep adjustments can
    # neither un-expire an expired lease (backwards jump) nor reclaim a live one
    # early (forward jump). See the comment on `_lease_now` below about restarts.
    def _wall_now(self) -> float:
        return float(self._wall_clock())

    def _lease_now(self) -> float:
        """Monotonic time for lease lifetimes only.

        Monotonic *does not* reset across process restarts on Linux/CPython: it
        is CLOCK_MONOTONIC, a single system-wide baseline (verified — two runs of
        the interpreter read ~6381 -> ~6382, not a fresh per-process zero). That
        is exactly why it is the correct clock here: a lease's expiry survives a
        worker (re)start and other processes share the same reference.

        Defined behaviour at restart: leases survive a worker (re)start because
        CLOCK_MONOTONIC keeps its baseline across process restarts — a job held by
        a re-executed worker is not reclaimed until its window genuinely elapses.
        BUT a full machine reboot / kexec *does* reset the monotonic baseline to
        ~0. A lease stamped on the pre-reboot uptime (e.g. up(216000s)+30 for a
        box that was up 2.5 days) would then be *larger* than every future now,
        so it would never expire by lease expiry: the stuck job could sit in
        `running` for the entire previous uptime (days), not one window. We do
        not accept that, so _sync_boot_ref() detects the reset and invalidates
        stale leases — see that method for why that is safe.
        """
        return float(self._lease_clock())

    def _current_boot_id(self) -> Optional[str]:
        """Identity of the current boot — stable within a boot, changes at reboot.

        Linux exposes /proc/sys/kernel/random/boot_id: a freshly generated 128-bit
        random UUID assigned at every boot and left unchanged for the lifetime of
        that boot. It is the right primitive here because it changes *per boot*
        regardless of uptime, unlike CLOCK_MONOTONIC (which only counts up from
        boot and therefore cannot tell a late-boot second apart from a post-reboot
        second past the old mark — see _sync_boot_ref). Returns None when the
        source is unavailable.

        Platform without boot_id: we fall back to CLOCK_MONOTONIC ordering as a
        best effort (now < stored monotonic baseline), which catches reboots that
        reconnect before their uptime passes the prior mark but still leaves a
        residual gap — exactly the missed case above. Identity comparison is used
        whenever available; only then is this fallback active.
        """
        path = "/proc/sys/kernel/random/boot_id"
        try:
            with open(path) as f:
                return f.read().strip() or None
        except OSError:
            return None

    def _sync_boot_ref(self) -> None:
        """Recover in-flight leases after a reboot by comparing boot identity.

        A lease's expiry is measured on CLOCK_MONOTONIC, which counts up from the
        boot — so its value GROWS through a process's life and *cannot* tell a
        late instant of one boot from an early instant of the next. Detecting a
        reboot therefore needs something that changes per boot, not ordering.
        We persist the current ``boot_id`` in ``meta.ref`` and compare by identity:
        on connect, if the stored id differs from ours we have crossed a boot
        boundary and invalidate stale leases.

        Why identity, not ordering: ordering misses the case where a rebooted box
        reconnects after its uptime has already passed the prior mark (e.g. up 100s,
        reboot, first connect at 200s: 200 >= 100 → no reset signalled). boot_id is
        fresh on every boot so that case cannot slip through.

        Chosen behaviour (and why it is safe): on identity mismatch we flip all
        `running` jobs back to `pending` (clearing worker_id/lease_expires) so the
        next dequeue reclaims them — at-most-once replay. Because every worker dies
        on reboot, any job still `running` was necessarily interrupted, so treating
        the whole set as reclaimable is correct and needs no per-job tag or sweep.
        Latency after reboot is bounded by one dequeue of the earliest-eligible job.

        Does ref get refreshed every connect? No — it is written once per boot and
        only overwritten when a *different* id is observed (i.e. at a reboot).
        Identity does not drift, so there is no decay to chase; refreshing on every
        connect would simply rewrite the same value.
        """
        if self._boot_synced:
            return
        self._boot_synced = True
        conn = self._ensure_connected()
        current_id = self._current_boot_id()
        now = self._lease_now()  # used only by the absent-source fallback below
        # BEGIN IMMEDIATE so the check+reset is serialized with workers writing
        # (and against another process also doing its first-time boot sync).
        conn.execute("BEGIN IMMEDIATE")
        try:
            stored = conn.execute(
                "SELECT value FROM meta WHERE key='ref'"
            ).fetchone()
            prior = stored["value"] if stored else None
            if current_id is not None:
                rebooted = (prior != current_id)
            else:
                # platform without boot_id -> best-effort monotonic ordering
                rebooted = prior is not None and now < float(prior)
            value = current_id if current_id is not None else repr(now)
            if prior is None:
                conn.execute("INSERT INTO meta (key, value) VALUES ('ref', ?)", (value,))
            elif rebooted:
                conn.execute(
                    "UPDATE jobs SET state='pending', worker_id=NULL, "
                    "lease_expires=NULL WHERE state='running'"
                )
                conn.execute("UPDATE meta SET value=? WHERE key='ref'", (value,))
            conn.commit()
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    @staticmethod
    def _dumps(obj) -> bytes:
        return json.dumps(obj, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _loads(blob) -> object:
        if blob is None:
            return None
        return json.loads(bytes(blob))

    def _backoff(self, attempts: int) -> float:
        # attempt >= 1 here; backoff for the *next* retry.
        secs = self.backoff_base * (2 ** (attempts - 1))
        return min(secs, self.backoff_cap)

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            job_type=row["job_type"],
            payload=self._loads(row["payload"]),
            priority=row["priority"],
            attempts=row["attempts"],
            worker_id=row["worker_id"],
            lease_expires=row["lease_expires"],
        )

    # --------------------------------------------------------------- public API
    def enqueue(
        self,
        job_type: str,
        payload: dict,
        priority: int = 0,
        run_at: Optional[float] = None,
        max_attempts: Optional[int] = None,
    ) -> int:
        """Insert a pending job and return its id."""
        now = self._wall_now()
        if run_at is None:
            run_at = now
        rows = self._ensure_connected().execute(
            """
            INSERT INTO jobs (job_type, payload, priority, state, run_at,
                              attempts, max_attempts, created_at, updated_at)
            VALUES (?, ?, ?, 'pending', ?, 0, ?, ?, ?)
            """,
            (
                job_type,
                self._dumps(payload),
                priority,
                run_at,
                self.max_attempts if max_attempts is None else max_attempts,
                now,
                now,
            ),
        )
        self._conn.commit()
        return int(rows.lastrowid)

    def dequeue(self, worker_id: str, types: Optional[Iterable[str]] = None) -> Job | None:
        """Atomically claim and return the next runnable job, or None.

        Runs inside BEGIN IMMEDIATE so SQLite serialises writers before we read;
        two concurrent processes can never be handed the same row.
        """
        conn = self._ensure_connected()
        # Two independent "nows": lease maths against the monotonic clock, and
        # scheduling/bookkeeping against wall-clock.
        t_mon = self._lease_now()   # used only for lease reclaim + lease_expires
        t_wall = self._wall_now()   # used only for run_at gate + updated_at

        allowed = tuple(types) if types else None
        allowed_clause = ""
        params: list = []
        if allowed:
            placeholders = ",".join("?" for _ in allowed)
            allowed_clause = f" AND job_type IN ({placeholders})"
            params.extend(allowed)

        with self._conn_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # 1) Reclaim leases of workers that vanished (lazy, no sweeper).
                conn.execute(
                    """
                    UPDATE jobs
                       SET state = 'pending', worker_id = NULL, lease_expires = NULL
                     WHERE state = 'running' AND lease_expires IS NOT NULL
                       AND lease_expires <= ?
                    """,
                    (t_mon,),
                )

                # 2) Select the next runnable job we are allowed to claim.
                row = conn.execute(
                    f"""
                    SELECT * FROM jobs
                     WHERE state = 'pending'
                       AND run_at <= ?
                       {allowed_clause}
                     ORDER BY priority DESC, run_at ASC, id ASC
                     LIMIT 1
                    """,
                    (t_wall, *params),
                ).fetchone()

                if row is None:
                    conn.execute("ROLLBACK")
                    return None

                # 3) Claim it: flip to running, stamp the lease. attempts
                #    increments atomically in SQL so a concurrent claim cannot
                #    also see this row as eligible (the SELECT excludes 'running').
                new_lease_expires = t_mon + self.visibility_timeout
                conn.execute(
                    """
                    UPDATE jobs
                       SET state = 'running',
                           worker_id = ?,
                           lease_expires = ?,
                           attempts = attempts + 1,
                           error = NULL,
                           updated_at = ?
                     WHERE id = ?
                    """,
                    (worker_id, new_lease_expires, t_wall, row["id"]),
                )
                # Re-read the *updated* row so reported attempts/lease are fresh
                # and consistent with what we just wrote.
                claimed = conn.execute(
                    "SELECT * FROM jobs WHERE id = ?", (row["id"],)
                ).fetchone()
                conn.execute("COMMIT")
                self._conn.commit()
                return self._row_to_job(claimed)
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def complete(self, job_id: int, result: dict) -> None:
        """Mark a running job done and store its result."""
        now = self._wall_now()
        cur = self._ensure_connected().execute(
            """
            UPDATE jobs
               SET state = 'done',
                   worker_id = NULL,
                   lease_expires = NULL,
                   result = ?,
                   error = NULL,
                   updated_at = ?
             WHERE id = ? AND state = 'running'
            """,
            (self._dumps(result), now, job_id),
        )
        self._conn.commit()
        if cur.rowcount != 1:
            raise ValueError(f"no running job with id={job_id}")

    def fail(self, job_id: int, error: str) -> None:
        """Record a failure; retry with backoff until attempts are exhausted."""
        now = self._wall_now()
        row = self._ensure_connected().execute(
            "SELECT attempts, max_attempts FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"no job with id={job_id}")

        # `attempts` is the dispatch count already bumped when the job was claimed.
        # We do not bump it again here: this is still that same attempt, now failing.
        attempts = row["attempts"]
        if attempts >= row["max_attempts"]:
            state, run_at = State.DEAD, None
        else:
            state, run_at = State.PENDING, now + self._backoff(attempts)

        cur = self._ensure_connected().execute(
            """
            UPDATE jobs
               SET state = ?,
                   worker_id = NULL,
                   lease_expires = NULL,
                   error = ?,
                   run_at = COALESCE(?, run_at),
                   updated_at = ?
             WHERE id = ?
            """,
            (state, error, run_at, now, job_id),
        )
        self._conn.commit()
        if cur.rowcount != 1:
            raise ValueError(f"no job with id={job_id}")

    # --------------------------------------------------------------- introspection
    def count_by_state(self) -> dict[str, int]:
        """Return {state: count} across all jobs — handy for tests/debug."""
        rows = self._ensure_connected().execute(
            "SELECT state, COUNT(*) AS n FROM jobs GROUP BY state"
        ).fetchall()
        counts = {s: 0 for s in (
            State.PENDING, State.RUNNING, State.DONE,
            State.FAILED, State.DEAD,
        )}
        counts.update({r["state"]: r["n"] for r in rows})
        return counts
