"""Executed by process workers, never by the API's job-submission endpoint."""

import asyncio
import logging
from time import perf_counter

from app.route_plan import compute_route_plan
from app.schemas_route import RoutePlanRequest
from app.task_queue import celery_app

logger = logging.getLogger(__name__)


async def _compute(payload: RoutePlanRequest):
    try:
        return await compute_route_plan(payload, offload=False)
    finally:
        # asyncio.run creates a new loop per job; don't reuse an OSRM Redis connection
        # attached to a closed loop on the next job in this worker process.
        from app.cache import redis_client

        await redis_client.aclose()


@celery_app.task(name="app.route_tasks.compute_route", bind=True)
def compute_route(self, payload: dict) -> dict:
    request = RoutePlanRequest.model_validate(payload)
    started = perf_counter()
    try:
        result = asyncio.run(_compute(request))
        return result.model_dump(mode="json")
    finally:
        logger.info(
            "route_job id=%s model=%s elapsed_seconds=%.3f",
            self.request.id,
            request.model,
            perf_counter() - started,
        )
