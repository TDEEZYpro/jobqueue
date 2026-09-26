"""Deterministic unit tests. No real time is slept; the clock is injected."""

import os
import tempfile
import unittest

from jobqueue import JobQueue, State


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def advance(self, secs):
        self.now += secs

    def __call__(self):
        return self.now


def make_queue(**kw):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "queue.db")
    kw.setdefault("max_attempts", 3)
    q = JobQueue(path, **kw)
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
        self.assertEqual(q.dequeue("w", types=["beta"]) is None, True)
        # alpha still available under its own filter
        self.assertEqual(q.dequeue("w2", types=["alpha"]).id, a)


class ScheduledTests(unittest.TestCase):
    def test_present_time_dequeued_immediately(self):
        clk = FakeClock()
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


class RetryTests(unittest.TestCase):
    def test_backoff_then_dead(self):
        clk = FakeClock(1000.0)
        q, _ = make_queue(clock=clk, max_attempts=3, backoff_base=1.0, backoff_cap=60.0)
        jid = q.enqueue("t", {})

        # attempt 1 fails -> retry at +1s (base * 2**0)
        job = q.dequeue("w")
        self.assertEqual(job.attempts, 1)
        q.fail(jid, "boom")
        self.assertEqual(q.count_by_state().get(State.PENDING), 1)
        # not runnable yet
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
        clk = FakeClock()
        q, _ = make_queue(clock=clk, visibility_timeout=30.0)
        jid = q.enqueue("t", {})

        a = q.dequeue("w1")
        self.assertEqual(a.id, jid)
        # second worker within the visibility window cannot grab it
        self.assertIsNone(q.dequeue("w2"))

        # before timeout: still held (and not reclaimable as pending)
        clk.advance(29.0)
        self.assertEqual(q.count_by_state().get(State.RUNNING), 1)
        self.assertEqual(q.count_by_state().get(State.PENDING), 0)

        # after the lease expires, another worker can reclaim it
        clk.advance(1.0)
        reclaimed = q.dequeue("w2")
        self.assertEqual(reclaimed.id, jid)
        self.assertEqual(reclaimed.attempts, 2)


if __name__ == "__main__":
    unittest.main()
