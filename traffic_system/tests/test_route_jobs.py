"""Job lifecycle contracts. Run with unittest; no live Redis/worker is needed."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.route_jobs import RouteJobNotFound, RouteJobs, RouteJobServiceUnavailable
from app.schemas_route import RoutePlanRequest


class RouteJobsTests(unittest.TestCase):
    def setUp(self):
        self.queue = Mock()
        self.registry = Mock()
        self.jobs = RouteJobs(self.queue, self.registry, ttl_seconds=86400)
        self.payload = RoutePlanRequest(
            origin="PUP",
            origin_lat=14.599,
            origin_lng=121.01,
            destinations=[{"name": "Destination", "lat": 14.6, "lng": 121.02}],
            model="baseline",
        )
        self.registry.get.return_value = json.dumps({"model": "baseline"})

    def test_submit_returns_without_waiting_for_baseline(self):
        response = self.jobs.submit(self.payload)
        self.assertEqual(response.status, "queued")
        self.assertIsNone(response.result)
        self.queue.AsyncResult.assert_not_called()
        arguments = self.queue.send_task.call_args.kwargs
        self.assertEqual(arguments["queue"], "routes-baseline")
        self.assertEqual(arguments["task_id"], response.job_id)
        self.assertEqual(arguments["args"][0], self.payload.model_dump(mode="json"))
        self.assertFalse(arguments["retry"])
        self.assertEqual(self.registry.set.call_args.kwargs["ex"], 86400)

    def test_antroute_uses_a_separate_queue(self):
        self.payload.model = "antroute"
        self.jobs.submit(self.payload)
        self.assertEqual(self.queue.send_task.call_args.kwargs["queue"], "routes-primary")

    def test_publishing_failure_removes_registration(self):
        self.queue.send_task.side_effect = ConnectionError("broker unavailable")
        with self.assertRaises(RouteJobServiceUnavailable):
            self.jobs.submit(self.payload)
        self.registry.delete.assert_called_once()

    def test_registration_failure_does_not_publish(self):
        self.registry.set.side_effect = ConnectionError("Redis unavailable")
        with self.assertRaises(RouteJobServiceUnavailable):
            self.jobs.submit(self.payload)
        self.queue.send_task.assert_not_called()

    def test_unknown_or_expired_job_is_not_pending_forever(self):
        self.registry.get.return_value = None
        with self.assertRaises(RouteJobNotFound):
            self.jobs.status("unknown-id")
        self.queue.AsyncResult.assert_not_called()

    def test_poll_never_waits_for_task_completion(self):
        task = Mock(state="STARTED")
        task.get.side_effect = AssertionError("blocking result.get must not be called")
        self.queue.AsyncResult.return_value = task
        response = self.jobs.status("registered-id")
        self.assertEqual(response.status, "running")
        self.assertIsNone(response.result)
        task.get.assert_not_called()

    def test_success_preserves_actual_algorithm_and_geometry(self):
        result = {"routes": [{
            "label": "Baseline", "via": "Road", "duration_min": 8,
            "distance_km": 3.2, "congestion_level": "moderate", "algorithm": "iaco",
            "path": [{"lat": 14.599, "lng": 121.01}, {"lat": 14.6, "lng": 121.02}],
        }], "notice": None}
        self.queue.AsyncResult.return_value = SimpleNamespace(state="SUCCESS", result=result)
        response = self.jobs.status("registered-id")
        self.assertEqual(response.status, "completed")
        self.assertEqual(response.result.routes[0].algorithm, "iaco")
        self.assertEqual(response.result.routes[0].path[1].lng, 121.02)

    def test_no_baseline_route_is_a_completed_result_with_notice(self):
        self.queue.AsyncResult.return_value = SimpleNamespace(
            state="SUCCESS", result={"routes": [], "notice": "No IACO route found."}
        )
        response = self.jobs.status("registered-id")
        self.assertEqual(response.status, "completed")
        self.assertEqual(response.result.notice, "No IACO route found.")

    def test_worker_failure_is_not_a_successful_empty_baseline(self):
        self.queue.AsyncResult.return_value = SimpleNamespace(state="FAILURE")
        response = self.jobs.status("registered-id")
        self.assertEqual(response.status, "failed")
        self.assertIsNone(response.result)
        self.assertTrue(response.error)

    def test_status_backend_failure_is_distinct_from_not_found(self):
        self.registry.get.side_effect = ConnectionError("Redis unavailable")
        with self.assertRaises(RouteJobServiceUnavailable):
            self.jobs.status("registered-id")


if __name__ == "__main__":
    unittest.main()
