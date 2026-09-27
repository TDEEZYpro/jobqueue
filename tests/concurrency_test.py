"""Concurrency test: N separate *processes* drain a shared DB.

Each worker owns its own SQLite connection (spawn start method -> fresh child),
so cross-process isolation is exercised for real. No threads, no sleeps.

Exactly-once invariant checked three ways:
  1. the shared completion list has length exactly N_jobs (no duplicate completes)
  2. final DB state: every job is 'done', none left running/dead/failed
  3. union of completed ids == the set of enqueued ids

Run: python -m tests.concurrency_test   (or: pytest tests/concurrency_test.py)
"""

import os
import sys
import tempfile
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobqueue import JobQueue  # noqa: E402

N_PROCESSES = 8
N_JOBS = 500


def drain_worker(db_path: str, worker_id: str, completed) -> int:
    q = JobQueue(db_path)
    n = 0
    try:
        while True:
            job = q.dequeue(worker_id)
            if job is None:
                break
            # simulate work; complete atomically via SQLite
            q.complete(job.id, {"worker": worker_id}, worker_id)
            completed.append(job.id)
            n += 1
    finally:
        q.close()
    return n


def main() -> int:
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "queue.db")

    seed = JobQueue(db_path)
    ids = [seed.enqueue("job", {"seq": i}) for i in range(N_JOBS)]
    seed.close()

    ctx = get_context("spawn")
    mgr = ctx.Manager()
    completed = mgr.list()

    pool = ctx.Pool(processes=N_PROCESSES)
    results = pool.starmap(
        drain_worker,
        [(db_path, f"w{i}", completed) for i in range(N_PROCESSES)],
    )
    pool.close()
    pool.join()

    done = list(completed)
    ok = True

    per_process = sum(results)
    print(f"enqueued          : {N_JOBS}")
    print(f"completed by procs : {per_process} (sum of per-process counts)")
    print(f"unique completed ids: {len(set(done))}")
    print(f"per-process breakdown: {results}")

    if len(done) != N_JOBS:
        print(f"FAIL: expected {N_JOBS} completes, got {len(done)} (duplicate handling?)")
        ok = False
    else:
        print("PASS: every job completed exactly once (no duplicate completes)")

    if set(done) != set(ids):
        print("FAIL: completed id set does not match enqueued id set")
        ok = False

    # third check: re-open the DB and confirm final durable state.
    verify = JobQueue(db_path)
    states = verify.count_by_state()
    verify.close()
    print(f"final db states   : {states}")
    if states.get("done") != N_JOBS or any(
        states.get(s) for s in ("running", "failed", "dead", "pending")
    ):
        print("FAIL: final DB state is not all-done")
        ok = False
    else:
        print("PASS: durable final state is all 'done'")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
