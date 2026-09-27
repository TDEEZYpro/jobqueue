"""Deterministic unit tests. No real time is slept; clocks are injected.

Two injectable clocks exist now: `clock` (wall, time.time) drives run_at
scheduling/bookkeeping and `lease_clock` (monotonic, time.monotonic) bounds
lease lifetimes. They can be advanced independently — that independence is the
subject of test_two_clocks_are_separate.
"""

import os
import tempfile
import unittest
from unittest import mock

from jobqueue import JobQueue, State


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def advance(self, secs):
        self.now += secs

    def __call__(self):
        return self.now


def make_queue(**kwargs):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "queue.db")
    kwargs.setdefault("max_attempts", 3)
    q = JobQueue(path, **kwargs)
    return q, path


class DequeueOrderingTests(unittest.TestCase):
    def test_priority_then_fifo(self):
        q, path = make_queue()
        low = q.enqueue("t", {"i": 1}, priority=0)
        high = q.enqueue("t", {"i": 2}, priority=10)
        mid = q.enqueue("t", {"i": 3}, priority=5)

        self.assertEqual(q.dequeue("w").id, high)
        self.assertEqual(q.dequeue("w").id, mid)
        self.assertEqual(q.dequeue("w").id, low)
        self.assertIsNone(q.dequeue("w"))

    def test_types_filter(self):
        q, _ = make_queue()
        a = q.enqueue("alpha", {})
        b = q.enqueue("beta", {})
        self.assertEqual(q.dequeue("w", types=["beta"]).id, b)
        self.assertIsNone(q.dequeue("w", types=["beta"]))
        # alpha still available under its own filter
        self.assertEqual(q.dequeue("w2", types=["alpha"]).id, a)


class ScheduledTests(unittest.TestCase):
    def test_present_time_dequeued_immediately(self):
        clk = FakeClock()  # wall clock
        q, _ = make_queue(clock=clk)
        now_job = q.enqueue("t", {}, run_at=clk.now)
        self.assertEqual(q.dequeue("w").id, now_job)

    def test_not_early(self):
        clk = FakeClock()
        q, _ = make_queue(clock=clk)
        future = q.enqueue("t", {}, run_at=clk.now + 50)
        self.assertIsNone(q.dequeue("w"))
        clk.advance(51)
        self.assertEqual(q.dequeue("w").id, future)


class LifecycleTests(unittest.TestCase):
    def test_complete(self):
        q, _ = make_queue()
        jid = q.enqueue("t", {"x": 1})
        job = q.dequeue("w")
        self.assertIsNotNone(job)
        q.complete(jid, {"ok": True})
        states = q.count_by_state()
        self.assertEqual(states.get(State.DONE), 1)
        self.assertEqual(states.get(State.RUNNING), 0)

    def test_complete_wrong_state_raises(self):
        q, _ = make_queue()
        jid = q.enqueue("t", {})
        with self.assertRaises(ValueError):
            q.complete(jid, {})  # still pending


class OwnershipTests(unittest.TestCase):
    """Proves complete()/fail() reject a stale worker who lost the job.

    This is finding #1: without an ownership guard, a worker whose lease expired
    mid-job could overwrite the report of the worker who re-claimed it. A new,
    optional `worker_id` argument makes completion/failure match on (id,
    state='running', worker_id); matching zero rows is a silent drop, never a
    retry (finding #3).
    """

    def test_stale_complete_is_rejected_but_live_owner_completes(self):
        lease = FakeClock(5000.0)
        q, _ = make_queue(lease_clock=lease, visibility_timeout=30.0)
        jid = q.enqueue("t", {"i": 1})

        a = q.dequeue("wA")          # worker A claims it
        self.assertEqual(a.id, jid)
        b = q.dequeue("wB")          # still leased -> B cannot grab it yet
        self.assertIsNone(b)

        # lease expires; B re-claims as its own attempt (attempts == 2)
        lease.advance(31.0)
        b = q.dequeue("wB")
        self.assertEqual(b.id, jid)
        self.assertEqual(b.attempts, 2)

        # A's report is now stale: it must be silently dropped, not raise and
        # not overwrite the live owner.
        self.assertIsNone(q.complete(jid, {"stale": True}, worker_id="wA"))

        # B's report wins: job completes exactly once.
        self.assertIsNone(q.complete(jid, {"ok": True}, worker_id="wB"))

        states = q.count_by_state()
        self.assertEqual(states.get(State.DONE), 1)
        self.assertEqual(states.get(State.RUNNING), 0)

    def test_stale_fail_is_not_retried(self):
        lease = FakeClock(5000.0)
        q, _ = make_queue(lease_clock=lease, visibility_timeout=30.0,
                          max_attempts=3)
        jid = q.enqueue("t", {"i": 1})

        q.dequeue("wA")     # A claims (attempt 1)
        lease.advance(31.0)  # lease expires
        b = q.dequeue("wB")  # B re-claims (attempt 2)

        # A's stale fail must be dropped, not retried: job stays owned by B and is
        # NOT resurrected into a fresh pending row or a bumped attempt.
        self.assertIsNone(q.fail(jid, "stale error", worker_id="wA"))

        states = q.count_by_state()
        self.assertEqual(states.get(State.RUNNING), 1)
        # attempts unchanged: still on B's attempt 2, not bumped by A's fail.
        self.assertEqual(b.attempts, 2)


class RetryTests(unittest.TestCase):
    def test_backoff_then_dead(self):
        clk = FakeClock(1000.0)  # wall clock drives run_at/backoff
        q, _ = make_queue(clock=clk, max_attempts=3, backoff_base=1.0, backoff_cap=60.0)
        jid = q.enqueue("t", {})

        # attempt 1 fails -> retry at +1s (base * 2**0)
        job = q.dequeue("w")
        self.assertEqual(job.attempts, 1)
        q.fail(jid, "boom")
        self.assertEqual(q.count_by_state().get(State.PENDING), 1)
        # not runnable yet (run_at in the future on the wall clock)
        self.assertIsNone(q.dequeue("w"))

        # need +1s to become runnable; also exercise lease reclamation implicitly
        clk.advance(1.0)
        job2 = q.dequeue("w")
        self.assertEqual(job2.attempts, 2)
        q.fail(jid, "boom again")  # attempt 2 -> retry at +2s

        clk.advance(2.0)
        job3 = q.dequeue("w")
        self.assertEqual(job3.attempts, 3)
        q.fail(jid, "final")  # attempts==max_attempts -> dead

        states = q.count_by_state()
        self.assertEqual(states.get(State.DEAD), 1)
        self.assertEqual(states.get(State.PENDING), 0)
        self.assertIsNone(q.dequeue("w"))

    def test_success_after_retry(self):
        clk = FakeClock()
        q, _ = make_queue(clock=clk, max_attempts=5, backoff_base=1.0)
        jid = q.enqueue("t", {})
        job = q.dequeue("w")
        q.fail(jid, "transient")
        clk.advance(1.0)
        self.assertEqual(q.dequeue("w").id, jid)  # retried fine
        q.complete(jid, {"done": True})


class LeaseTests(unittest.TestCase):
    def test_lease_prevents_reclaim_until_timeout(self):
        wall = FakeClock(100.0)   # stays fixed; run_at gating lives on this clock
        lease = FakeClock(5000.0)  # the monotonic clock that bounds the lease
        q, _ = make_queue(clock=wall, lease_clock=lease,
                          visibility_timeout=30.0, max_attempts=3)
        jid = q.enqueue("t", {})

        a = q.dequeue("w1")
        self.assertEqual(a.id, jid)
        # second worker within the visibility window cannot grab it
        self.assertIsNone(q.dequeue("w2"))

        # before timeout: still held (and not reclaimable as pending)
        lease.advance(29.0)
        self.assertEqual(q.count_by_state().get(State.RUNNING), 1)
        self.assertEqual(q.count_by_state().get(State.PENDING), 0)

        # after the lease expires, another worker can reclaim it
        lease.advance(1.0)
        reclaimed = q.dequeue("w2")
        self.assertEqual(reclaimed.id, jid)
        self.assertEqual(reclaimed.attempts, 2)


class ClockSeparationTests(unittest.TestCase):
    """Proves the two clocks are genuinely independent."""

    def test_wall_jump_does_not_reclaim_live_lease(self):
        # A large forward wall-clock jump (as NTP could cause) must not expire a
        # lease, because leases ride on the monotonic clock, not the wall one.
        wall = FakeClock(0.0)
        lease = FakeClock(1000.0)
        q, _ = make_queue(clock=wall, lease_clock=lease,
                          visibility_timeout=30.0, max_attempts=3)
        jid = q.enqueue("t", {})

        self.assertEqual(q.dequeue("w1").id, jid)  # leased at lease-clock 1000 -> expires 1030
        wall.advance(10_000)                       # wall clock surges forward massively
        self.assertIsNone(q.dequeue("w2"))         # lease still live on monotonic clock

    def test_lease_advances_independent_of_run_at(self):
        # Advancing the lease clock expires a job while run_at gating (wall clock)
        # is untouched — the job that was already runnable stays claimable.
        wall = FakeClock(0.0)
        lease = FakeClock(1000.0)
        q, _ = make_queue(clock=wall, lease_clock=lease,
                          visibility_timeout=30.0, max_attempts=3)
        # run_at in the past on the wall clock => immediately runnable
        jid = q.enqueue("t", {}, run_at=-100.0)

        self.assertEqual(q.dequeue("w1").id, jid)
        lease.advance(31.0)  # visibility window elapses
        self.assertEqual(q.dequeue("w2").id, jid)  # reclaimed purely on lease clock


class RebootTests(unittest.TestCase):
    """Reboot detection is by boot identity (boot_id), not monotonic ordering,
    so a box that reconnects after its uptime passes the old mark is still caught.
    """

    def test_reboot_past_old_mark_is_detected(self):
        # Up 100s, job running under BOOT-A. Reboot; new boot already at 200s uptime
        # — ABOVE the prior monotonic mark. Ordering would miss this; identity does not.
        boot_a = FakeClock(100.0)
        q_a, path = make_queue(clock=FakeClock(0.0), lease_clock=boot_a,
                               visibility_timeout=30.0, max_attempts=3)
        with mock.patch.object(JobQueue, "_current_boot_id", return_value="BOOT-A"):
            jid = q_a.enqueue("t", {})
            self.assertEqual(q_a.dequeue("w1").id, jid)  # running under BOOT-A

        boot_b = FakeClock(200.0)  # new boot, uptime already past the old mark (200 > 100)
        with mock.patch.object(JobQueue, "_current_boot_id", return_value="BOOT-B"):
            q_b = JobQueue(path, clock=FakeClock(0.0), lease_clock=boot_b,
                           visibility_timeout=30.0, max_attempts=3)
        states = q_b.count_by_state()
        self.assertEqual(states.get(State.RUNNING), 0)   # stale lease invalidated
        self.assertEqual(states.get(State.PENDING), 1)   # back to pending
        self.assertEqual(q_b.dequeue("w2").id, jid)

    def test_same_boot_no_false_reboot(self):
        # Same identity across processes -> never invalidates, regardless of how far
        # the monotonic clock has advanced within the boot.
        boot_a = FakeClock(100.0)
        with mock.patch.object(JobQueue, "_current_boot_id", return_value="BOOT-SAME"):
            q_a, path = make_queue(clock=FakeClock(0.0), lease_clock=boot_a,
                                   visibility_timeout=30.0, max_attempts=3)
            jid = q_a.enqueue("t", {})
            self.assertEqual(q_a.dequeue("w1").id, jid)  # running

            boot_b = FakeClock(120.0)  # advanced within the same boot (lease not yet expired)
            q_b = JobQueue(path, clock=FakeClock(0.0), lease_clock=boot_b,
                           visibility_timeout=30.0, max_attempts=3)
        self.assertEqual(q_b.count_by_state().get(State.RUNNING), 1)   # not falsely reclaimed
        self.assertEqual(q_b.dequeue("w2"), None)                       # still held
    def test_absent_boot_id_falls_back_to_monotonic_ordering(self):
        # Platform without boot_id: fall back to monotonic ordering. This catches a
        # reboot that reconnects before uptime passes the prior mark (documented gap).
        with mock.patch.object(JobQueue, "_current_boot_id", return_value=None):
            boot_a = FakeClock(300.0)
            q_a, path = make_queue(clock=FakeClock(0.0), lease_clock=boot_a,
                                   visibility_timeout=30.0, max_attempts=3)
            jid = q_a.enqueue("t", {})
            self.assertEqual(q_a.dequeue("w1").id, jid)  # running

            boot_b = FakeClock(250.0)  # reboot, uptime still below the prior 300
            q_b = JobQueue(path, clock=FakeClock(0.0), lease_clock=boot_b,
                           visibility_timeout=30.0, max_attempts=3)
        states = q_b.count_by_state()
        self.assertEqual(states.get(State.RUNNING), 0)   # detected by ordering
        self.assertEqual(q_b.dequeue("w2").id, jid)

    def test_absent_boot_id_misses_late_reconnect(self):
        """The documented residual gap: without boot_id a reboot reconnecting after
        the old mark is missed (monotonic ordering cannot tell apart boots). This
        test records that behaviour so it is explicit, not accidental."""
        with mock.patch.object(JobQueue, "_current_boot_id", return_value=None):
            boot_a = FakeClock(100.0)  # ref stored at first connect = 100; job claimed up~100 -> lease_expires ~130
            q_a, path = make_queue(clock=FakeClock(0.0), lease_clock=boot_a,
                                   visibility_timeout=30.0, max_attempts=3)
            jid = q_a.enqueue("t", {})
            self.assertEqual(q_a.dequeue("w1").id, jid)  # running under prior ref 100

            boot_b = FakeClock(120.0)  # reboot: uptime (120) > prior ref (100) -> ordering MISSES
            q_b = JobQueue(path, clock=FakeClock(0.0), lease_clock=boot_b,
                           visibility_timeout=30.0, max_attempts=3)
        self.assertEqual(q_b.count_by_state().get(State.RUNNING), 1)   # still held by stale lease
        self.assertIsNone(q_b.dequeue("w2"))  # not reclaimable until new uptime >= ~130 (self-expiry)



if __name__ == "__main__":
    unittest.main()
