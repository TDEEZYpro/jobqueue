"""Phase 3: a thin HTTP admin API over :class:`JobQueue`.

Stdlib only (``http.server.ThreadingHTTPServer``). JSON in, JSON out.

Endpoints
---------
POST   /jobs            enqueue -> {"job_id": id}
GET    /jobs            list with ?state= ?types=& limit= & cursor=
GET    /jobs/<id>       full detail (attempts + last error)
POST   /jobs/<id>/retry revive a DEAD job to PENDING, attempts reset -> 0
DELETE /jobs/<id>       remove a job (refused while running)
GET    /stats           counts by state, oldest-pending age, dead count

Design constraints (the ones the caller cares about)
----------------------------------------------------
* Own connection per request. Each request opens its own short-lived
  ``JobQueue`` for the whole handler body and closes it at the end. The server
  therefore holds *no* shared queue object with the worker pool — an admin
  request can never hold a worker's write transaction open, and two requests on
  different threads never share one sqlite connection (which JobQueue does not
  protect against). This is why we reconstruct from db_path + config per
  request rather than passing a live queue in.

* Read paths never take the write lock. Every read here is an autocommit
  SELECT; under WAL (enforced by JobQueue) readers do not block the single
  SQLite writer, and writes stay short one-statement commits. There is no long-
  held transaction anywhere. Pagination is cursor-based, not offset (see
  JobQueue.list_jobs).

* Fail fast, fail quiet. Malformed JSON, unknown fields and bad types are a 400
  with a human message — never a 500. Unknown routes/ids map to 404; reviving a
  non-dead job or deleting a running one is a 409. Anything else (e.g. a SQLite
  I/O error) is a genuine 500.
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qsl, urlsplit

from .core import JobNotFound, JobQueue, OwnershipError, QueueError, State

__all__ = ["make_admin_server", "AdminAPI"]

_KNOWN_STATES = (State.PENDING, State.RUNNING, State.DONE, State.FAILED, State.DEAD)


class APIError(Exception):
    """An error that maps to a specific HTTP status + JSON body."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ----------------------------------------------------------------------------
# request / response helpers
# ----------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    """One handler class; routing dispatches on (method, path)."""

    server_version = "JobQueueAdmin/1.0"
    protocol_version = "HTTP/1.1"  # keep-alive; we always send Content-Length

    api: "AdminAPI"  # assigned by make_admin_server on the server instance

    # --- low-level response helpers ---------------------------------------
    def _send(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        # Never leak a stack trace; always a tidy JSON error body.
        self._send(status, {"error": message})

    def log_message(self, fmt: str, *args) -> None:  # pragma: no cover - noisy default
        # Keep the stdlib access log quiet; operators can pipe stderr elsewhere.
        pass

    def _body(self) -> bytes:
        length = self.headers.get("Content-Length")
        if length is None:
            raise APIError(400, "request must include a Content-Length header")
        try:
            n = int(length)
        except ValueError:
            raise APIError(400, "invalid Content-Length") from None
        if n < 0:
            raise APIError(400, "Content-Length must not be negative")
        return self.rfile.read(n)

    def _json_body(self) -> dict:
        raw = self._body()
        if not raw:
            raise APIError(400, "request body is required")
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise APIError(400, "body is not valid JSON") from None
        if not isinstance(obj, dict):
            raise APIError(400, "request body must be a JSON object")
        return obj

    # --- routing ------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        path = urlsplit(self.path).path.rstrip("/") or "/"
        try:
            if path == "/jobs":
                return self._handle_enqueue()
            m = re.fullmatch(r"/jobs/(?P<id>\d+)/retry", path)
            if m:
                return self._handle_retry(int(m.group("id")))
            raise APIError(404, f"no such route: POST {self.path}")
        except APIError as exc:
            return self._error(exc.status, exc.message)
        except (QueueError, ValueError) as exc:
            # Missing job / stale ownership reach here. A recover() of a non-dead
            # job raises ValueError and must be a 409, not a 500.
            if isinstance(exc, JobNotFound):
                return self._error(404, str(exc))
            return self._error(409, str(exc))

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path.rstrip("/") or "/"
        try:
            if path == "/jobs":
                return self._handle_list()
            if path == "/stats":
                return self._handle_stats()
            m = re.fullmatch(r"/jobs/(?P<id>\d+)", path)
            if m:
                return self._handle_detail(int(m.group("id")))
            raise APIError(404, f"no such route: GET {self.path}")
        except APIError as exc:
            return self._error(exc.status, exc.message)
        except (QueueError, ValueError) as exc:
            if isinstance(exc, JobNotFound):
                # e.g. an unparseable pagination cursor, or a job read past expiry.
                return self._error(404, str(exc))
            if isinstance(exc, ValueError):
                # An opaque/invalid cursor is a client error -> 400, never 500.
                return self._error(400, str(exc))
            return self._error(500, "internal error")

    def do_DELETE(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path.rstrip("/") or "/"
        try:
            m = re.fullmatch(r"/jobs/(?P<id>\d+)", path)
            if not m:
                raise APIError(404, f"no such route: DELETE {self.path}")
            return self._handle_delete(int(m.group("id")))
        except APIError as exc:
            return self._error(exc.status, exc.message)
        except ValueError as exc:
            # "cannot delete running job ..." -> 409 Conflict.
            return self._error(409, str(exc))
        except QueueError as exc:
            if isinstance(exc, JobNotFound):
                return self._error(404, str(exc))
            raise

    # --- endpoint handlers --------------------------------------------------
    def _handle_enqueue(self) -> None:
        data = self._json_body()
        job_id = self.api.enqueue(data)
        self._send(201, {"job_id": job_id})

    def _handle_list(self) -> None:
        params = dict(parse_qsl(urlsplit(self.path).query))
        jobs, cursor = self.api.list_jobs(params)
        self._send(200, {"jobs": jobs, "cursor": cursor})

    def _handle_detail(self, job_id: int) -> None:
        detail = self.api.fetch_job_detail(job_id)
        if detail is None:
            raise APIError(404, f"no such job: {job_id}")
        self._send(200, {"job": detail})

    def _handle_retry(self, job_id: int) -> None:
        detail = self.api.recover(job_id)
        self._send(200, {"job": detail, "id": job_id})

    def _handle_delete(self, job_id: int) -> None:
        self.api.delete_job(job_id)
        self._send(200, {"deleted": True, "id": job_id})

    def _handle_stats(self) -> None:
        self._send(200, self.api.stats())


# ----------------------------------------------------------------------------
# API + server construction
# ----------------------------------------------------------------------------

_ADMIN_ENQUEUE_FIELDS = ("job_type", "payload", "priority", "run_at", "max_attempts")


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or value == "":
        raise APIError(400, f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, field: str, *, minimum: Optional[int] = None) -> int:
    # bool is a subclass of int; reject it so True/False don't sneak through.
    if isinstance(value, bool) or not isinstance(value, int):
        raise APIError(400, f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise APIError(400, f"{field} must be >= {minimum}")
    return value


def _require_number_or_none(value: Any, field: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise APIError(400, f"{field} must be a number or null")
    return float(value)


class AdminAPI:
    """The business logic behind the HTTP endpoints. Thin: delegates to core.

    Constructed from a db_path (+ optional queue config) so every request can
    open an isolated connection — never sharing the worker pool's connection.
    """

    def __init__(self, db_path: str, queue_config: Optional[dict] = None) -> None:
        self.db_path = db_path
        self.queue_config = dict(queue_config or {})

    def _queue(self) -> JobQueue:
        # Fresh connection per call; caller holds it in a `with` for the whole
        # request so nothing outlives the handler (see module docstring).
        return JobQueue(self.db_path, **self.queue_config)

    def enqueue(self, data: dict) -> int:
        unknown = set(data) - set(_ADMIN_ENQUEUE_FIELDS)
        if unknown:
            raise APIError(400, f"unknown field(s): {', '.join(sorted(unknown))}")
        job_type = _require_str(data.get("job_type"), "job_type")
        payload = data.get("payload")
        if not isinstance(payload, dict):
            raise APIError(400, "payload must be a JSON object")
        priority = _require_int(
            data.get("priority", 0), "priority"
        )
        run_at = _require_number_or_none(data.get("run_at"), "run_at")
        max_attempts = (
            None
            if "max_attempts" not in data
            else _require_int(data["max_attempts"], "max_attempts", minimum=1)
        )
        with self._queue() as q:
            return q.enqueue(job_type, payload, priority, run_at, max_attempts)

    def list_jobs(self, params: dict):
        state = params.get("state")
        if state is not None and state not in _KNOWN_STATES:
            raise APIError(400, f"unknown state '{state}' (allowed: {', '.join(_KNOWN_STATES)})")
        types_raw = params.get("types")
        types = None
        if types_raw is not None:
            types = [t.strip() for t in types_raw.split(",")]
            if any(t == "" for t in types):
                raise APIError(400, "types must be a comma-separated list of non-empty strings")
        limit_raw = params.get("limit", "50")
        try:
            limit = int(limit_raw)
        except ValueError:
            raise APIError(400, "limit must be an integer") from None
        cursor = params.get("cursor")
        with self._queue() as q:
            return q.list_jobs(state=state, types=types, limit=limit, cursor=cursor)

    def fetch_job_detail(self, job_id: int) -> Optional[dict]:
        with self._queue() as q:
            return q.fetch_job_detail(job_id)

    def recover(self, job_id: int) -> dict:
        with self._queue() as q:
            return q.recover(job_id)

    def delete_job(self, job_id: int) -> None:
        with self._queue() as q:
            q.delete_job(job_id)

    def stats(self) -> dict:
        with self._queue() as q:
            counts = q.count_by_state()
            age = q.oldest_pending_age()
        return {
            "counts": counts,
            "oldest_pending_age_seconds": age,
            "dead": counts[State.DEAD],
            "running": counts[State.RUNNING],
            "pending": counts[State.PENDING],
        }


def make_admin_server(
    queue, *, host: str = "127.0.0.1", port: int = 0
) -> ThreadingHTTPServer:
    """Build a ThreadingHTTPServer backed by an :class:`AdminAPI`.

    `queue` is either a live :class:`JobQueue` (its db_path + config are copied)
    or a bare db_path string. In both cases the server opens its own connection
    per request, so it never shares a handle with the worker pool and never holds
    a write transaction open against a running queue.

    The returned object is a ready-to-serve ``ThreadingHTTPServer``; start it in
    a background thread (as tests do) or pass it to ``serve_forever``.
    """
    if isinstance(queue, JobQueue):
        db_path = queue.db_path
        queue_config = {
            "max_attempts": getattr(queue, "max_attempts", 3),
            "backoff_base": getattr(queue, "backoff_base", 0.5),
            "backoff_cap": getattr(queue, "backoff_cap", 3600.0),
            "visibility_timeout": getattr(queue, "visibility_timeout", 30.0),
            "busy_timeout_ms": getattr(queue, "busy_timeout_ms", 5000),
        }
    else:
        db_path = queue
        queue_config = {}

    api = AdminAPI(db_path, queue_config)
    handler = _bound_handler(api)

    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.api = api  # type: ignore[attr-defined]
    return server


def _bound_handler(api: AdminAPI) -> "_Handler":
    # Bind this request's API onto the class attribute so BaseHTTPRequestHandler
    # can reach it without per-instance plumbing.
    class _BoundHandler(_Handler):
        pass

    _BoundHandler.api = api  # type: ignore[attr-defined]
    return _BoundHandler
