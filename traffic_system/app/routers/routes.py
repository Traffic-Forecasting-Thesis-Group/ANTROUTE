from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.evaluation import get_comparison_metrics
from app.config import settings
from app.route_jobs import RouteJobNotFound, RouteJobs, RouteJobServiceUnavailable
from app.route_plan import compute_route_plan
from app.schemas_route import ComparisonMetricsResponse, RouteJobResponse, RoutePlanRequest, RoutePlanResponse
from app.task_queue import celery_app, job_registry

router = APIRouter(prefix="/routes", tags=["routes"])
jobs = RouteJobs(celery_app, job_registry, settings.route_job_ttl_seconds)


@router.post("/plan", response_model=RoutePlanResponse)
async def plan(payload: RoutePlanRequest):
    # Compatibility endpoint. Updated clients use jobs for both models.
    return await compute_route_plan(payload)


@router.post("/jobs", response_model=RouteJobResponse, status_code=202)
def create_route_job(payload: RoutePlanRequest):
    # Sync endpoints run in FastAPI's thread pool: Redis/publishing cannot block
    # the API event loop, and no road-graph calculation happens here.
    try:
        return jobs.submit(payload)
    except RouteJobServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=RouteJobResponse)
def route_job_status(job_id: UUID):
    try:
        return jobs.status(str(job_id))
    except RouteJobNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RouteJobServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/comparison-metrics", response_model=ComparisonMetricsResponse)
async def comparison_metrics():
    return get_comparison_metrics()
