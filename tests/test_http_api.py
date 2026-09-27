"""Integration tests for the stdlib HTTP admin API.

These exercise the *real* :class:`ThreadingHTTPServer` end to end over an actual
TCP socket (via ``http.client``) — no handler mocking, no in-process stubbing.
The server runs in a daemon thread bound to ``127.0.0.1:0`` so the OS hands us an
ephemeral port; every request is an honest HTTP round trip through the socket.

State we need outside the HTTP surface (driving a job to `dead` / `running`, or
reading row-level state for assertions) is done through a plain :class:`JobQueue`
pointed at the same database file — that is not mocking the handler, it is just
setting up and verifying a shared SQLite store the API operates on.
"""

import http.client
import json
import os
import shutil
import tempfile
import threading
import time
import unittest

from jobqueue import JobQueue
from jobqueue.admin import make_admin_server


class _Client:
    """Minimal HTTP/1.1 client around ``http.client`` returning (status, body)."""

    def __init__(self, host, port):
        self.host = host
        self.port = port

    def request(self, method, path, body=None, raw=False):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        headers = {}
        payload = None
        if body is not None:
            payload = body if raw else json.dumps(body)
            headers["Content-Type"] = "application/json"
        try:
            conn.request(method, path, body=payload, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            parsed = json.loads(data.decode("utf-8")) if data else None
            return resp.status, parsed
        finally:
            conn.close()


class AdminAPITest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="adminapi_")
        self.db_path = os.path.join(self.dir, "queue.db")
        # Fresh core queue for seeding / inspecting state (shared SQLite store).
        self.q = JobQueue(self.db_path)

        self.server = make_admin_server(self.db_path, host="127.0.0.1", port=0)
        self.port = self.server.server_address[1]  # OS-assigned ephemeral port
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.q.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    @property
    def c(self):
        return _Client("127.0.0.1", self.port)

    # -- helpers ----------------------------------------------------------
    def enqueue(self, body=None):
        status, data = self.c.request(
            "POST", "/jobs", {"job_type": "t", "payload": {}, **(body or {})}
        )
        return status, data

    # -- happy paths ------------------------------------------------------
    def test_enqueue_returns_201_and_list_lists_it(self):
        status, data = self.enqueue()
        self.assertEqual(status, 201)
        self.assertIn("job_id", data)
        jid = data["job_id"]

        status, data = self.c.request("GET", "/jobs")
        self.assertEqual(status, 200)
        ids = [j["id"] for j in data["jobs"]]
        self.assertIn(jid, ids)

    def test_list_default_limit(self):
        for _ in range(60):
            self.enqueue()
        status, data = self.c.request("GET", "/jobs")
        # Default page is 50; a next cursor points at the remaining 10.
        self.assertEqual(len(data["jobs"]), 50)
        self.assertIsNotNone(data["cursor"])

    def test_get_detail_includes_attempts_error_and_metadata(self):
        jid = self.q.enqueue("slow", {"size": 4}, 0, None, 1)  # max_attempts=1
        self.q.dequeue("w")
        self.q.fail(jid, "crash", worker_id="w")  # -> DEAD

        status, data = self.c.request("GET", f"/jobs/{jid}")
        self.assertEqual(status, 200)
        job = data["job"]
        self.assertEqual(job["state"], "dead")
        self.assertEqual(job["job_type"], "slow")
        self.assertEqual(job["attempts"], 1)
        self.assertEqual(job["error"], "crash")
        self.assertEqual(job["max_attempts"], 1)
        self.assertIn("run_at", job)

    # -- retry ------------------------------------------------------------
    def test_retry_revives_dead_resets_attempts(self):
        jid = self.q.enqueue("slow", {}, 0, None, 1)
        self.q.dequeue("w")
        self.q.fail(jid, "boom", worker_id="w")

        status, data = self.c.request("POST", f"/jobs/{jid}/retry")
        self.assertEqual(status, 200)
        job = data["job"]
        self.assertEqual(job["state"], "pending")
        self.assertEqual(job["attempts"], 0)  # revived attempt count reset

        status, after = self.c.request("GET", f"/jobs/{jid}")
        self.assertEqual(after["job"]["state"], "pending")

    def test_retry_rejects_non_dead(self):
        jid = self.q.enqueue("plain", {})  # still pending
        status, data = self.c.request("POST", f"/jobs/{jid}/retry")
        self.assertEqual(status, 409)
        self.assertIn("error", data)
        # Not resurrected; job stays as it was (pending), never dead.
        self.assertEqual(self.q.count_by_state().get("dead"), 0)

    def test_retry_unknown_id_404(self):
        status, data = self.c.request("POST", "/jobs/987654321/retry")
        self.assertEqual(status, 404)
        self.assertIn("error", data)

    # -- delete -----------------------------------------------------------
    def test_delete_refuses_running_then_deletes_pending(self):
        running_id = self.q.enqueue("r", {})
        self.q.dequeue("w")  # -> RUNNING

        status, data = self.c.request("DELETE", f"/jobs/{running_id}")
        self.assertEqual(status, 409)
        self.assertIn("error", data)

        pending_id = self.q.enqueue("p", {})
        status, data = self.c.request("DELETE", f"/jobs/{pending_id}")
        self.assertEqual(status, 200)
        self.assertTrue(data["deleted"])

        status, _ = self.c.request("GET", f"/jobs/{pending_id}")
        self.assertEqual(status, 404)

    def test_delete_unknown_id_404(self):
        status, data = self.c.request("DELETE", "/jobs/123987")
        self.assertEqual(status, 404)
        self.assertIn("error", data)

    # -- error handling (client errors must be 4xx, never 5xx) -----------
    def test_malformed_json_is_400(self):
        status, body = self.c.request(
            "POST", "/jobs", raw=True, body=b"{not json,"
        )
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_missing_job_type_is_400_not_keyerror(self):
        # The whole point: a missing required field must be a tidy 400, never a
        # KeyError that leaks as a 500.
        status, body = self.c.request("POST", "/jobs", {"payload": {}})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_unknown_field_is_400(self):
        status, _ = self.c.request(
            "POST", "/jobs", {"job_type": "t", "payload": {}, "nope": 1}
        )
        self.assertEqual(status, 400)

    def test_payload_must_be_object_is_400(self):
        status, _ = self.c.request(
            "POST", "/jobs", {"job_type": "t", "payload": "not-object"}
        )
        self.assertEqual(status, 400)

    def test_priority_must_be_integer_is_400(self):
        status, _ = self.c.request(
            "POST", "/jobs", {"job_type": "t", "payload": {}, "priority": 1.5}
        )
        self.assertEqual(status, 400)

    def test_bad_state_filter_is_400(self):
        status, _ = self.c.request("GET", "/jobs?state=banana")
        self.assertEqual(status, 400)

    def test_non_integer_limit_is_400(self):
        status, _ = self.c.request("GET", "/jobs?limit=abc")
        self.assertEqual(status, 400)

    # -- cursor pagination -----------------------------------------------
    def test_cursor_pagination_pages_exactly_once(self):
        # 8 jobs, page size 3 -> pages of [3, 3, 2]. Following cursors to
        # exhaustion must surface every job exactly once with no overlap.
        ids = [self.q.enqueue("t", {"i": i}) for i in range(8)]
        all_ids: list = []
        prev_id = None
        cursor = None
        for _ in range(5):  # cap iterations; well under the real page count
            status, page = self.c.request(
                "GET", f"/jobs?limit=3&cursor={cursor}" if cursor else "/jobs?limit=3"
            )
            self.assertEqual(status, 200)
            jobs = page["jobs"]
            all_ids += [j["id"] for j in jobs]
            # Ordered ascending by (created_at ASC, id ASC) within and across pages.
            for j in jobs:
                if prev_id is not None:
                    self.assertGreaterEqual(j["id"], prev_id)
                prev_id = j["id"]
            cursor = page.get("cursor")
            if cursor is None:  # last page
                break
        # Every job surfaced exactly once, in the correct order, no dupes.
        self.assertEqual(all_ids, ids)
        self.assertEqual(sorted(all_ids), sorted(ids))

    def test_bad_cursor_is_400_not_500(self):
        status, body = self.c.request("GET", "/jobs?cursor=@@@not-base64@@@")
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    # -- filtering + stats ------------------------------------------------
    def test_state_filter(self):
        jid = self.q.enqueue("filtered", {}, 0, None, 1)
        self.q.dequeue("w")
        self.q.fail(jid, "x", worker_id="w")  # DEAD

        status, data = self.c.request("GET", "/jobs?state=dead")
        self.assertEqual(status, 200)
        self.assertTrue(all(j["state"] == "dead" for j in data["jobs"]))
        self.assertIn(jid, [j["id"] for j in data["jobs"]])

    def test_stats_reflects_live_state(self):
        jid = self.q.enqueue("s", {}, 0, None, 1)
        self.q.dequeue("w")
        self.q.fail(jid, "x", worker_id="w")  # -> DEAD

        status, data = self.c.request("GET", "/stats")
        self.assertEqual(status, 200)
        # stats() returns the tally unwrapped at the top level of the body.
        self.assertIn("counts", data)
        self.assertIn("dead", data)
        # Counts agree with the core's own tally over the same file.
        live = self.q.count_by_state()
        for key in ("pending", "running", "done", "failed", "dead"):
            self.assertEqual(data["counts"].get(key, 0), live.get(key, 0))

    def test_stats_reports_dead_count(self):
        jid = self.q.enqueue("s", {}, 0, None, 1)
        self.q.dequeue("w")
        self.q.fail(jid, "x", worker_id="w")
        status, body = self.c.request("GET", "/stats")
        self.assertEqual(body["dead"], 1)


if __name__ == "__main__":
    unittest.main()
