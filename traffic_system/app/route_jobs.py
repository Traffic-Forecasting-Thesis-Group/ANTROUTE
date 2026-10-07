"""Submit and inspect jobs without waiting for their calculations."""

import json
from contextlib import suppress
from uuid import uuid4

from app.schemas_route import RouteJobResponse, RoutePlanRequest, RoutePlanResponse

TASK_NAME = "app.route_tasks.compute_route"


class RouteJobNotFound(Exception):
    pass


class RouteJobServiceUnavailable(Exception):
    pass


class RouteJobs:
    def __init__(self, queue, registry, ttl_seconds: int):
        self.queue = queue
        self.registry = registry
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _key(job_id: str) -> str:
        return f"antroute:route-job:{job_id}"

    def submit(self, payload: RoutePlanRequest) -> RouteJobResponse:
        job_id = str(uuid4())
        key = self._key(job_id)
        try:
            self.registry.set(key, json.dumps({"model": payload.model}), ex=self.ttl_seconds)
            self.queue.send_task(
                TASK_NAME,
                args=[payload.model_dump(mode="json")],
                task_id=job_id,
                queue="routes-baseline" if payload.model == "baseline" else "routes-primary",
                expires=self.ttl_seconds,
                # Retrying an ambiguous publish could create duplicate computations.
                retry=False,
            )
        except Exception as exc:
            with suppress(Exception):
                self.registry.delete(key)
            raise RouteJobServiceUnavailable("The route job service is unavailable. Try again.") from exc
        return RouteJobResponse(job_id=job_id, model=payload.model, status="queued")

    def status(self, job_id: str) -> RouteJobResponse:
        try:
            metadata = self.registry.get(self._key(job_id))
            if metadata is None:
                raise RouteJobNotFound("This route job does not exist or has expired.")
            model = json.loads(metadata)["model"]
            task = self.queue.AsyncResult(job_id)
            state = task.state
            if state == "SUCCESS":
                result = RoutePlanResponse.model_validate(task.result)
                return RouteJobResponse(job_id=job_id, model=model, status="completed", result=result)
            if state in {"FAILURE", "REVOKED"}:
                return RouteJobResponse(
                    job_id=job_id,
                    model=model,
                    status="failed",
                    error="Route calculation failed or exceeded its execution limit. Check the worker logs.",
                )
            return RouteJobResponse(
                job_id=job_id,
                model=model,
                status="running" if state == "STARTED" else "queued",
            )
        except RouteJobNotFound:
            raise
        except Exception as exc:
            raise RouteJobServiceUnavailable("Could not read route job status. Try again.") from exc
