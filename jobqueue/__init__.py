"""Durable SQLite-backed job queue. Standard library only.

Phase 1: ``JobQueue`` (durable enqueue/dequeue with leases + retry).
Phase 2: ``Pool`` / ``Worker`` (process pool that drains the queue).
"""

from .core import Job, JobQueue, State
from .worker import Handler, Pool, Worker

__all__ = ["Job", "JobQueue", "State", "Pool", "Worker", "Handler"]
