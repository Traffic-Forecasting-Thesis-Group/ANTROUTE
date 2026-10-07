"""Lightweight API/worker configuration; importing this must not load the road graph."""

from celery import Celery
from redis import Redis

from app.config import settings

broker_url = settings.celery_broker_url or settings.redis_url
result_url = settings.celery_result_backend or settings.redis_url
visibility_timeout = settings.route_job_time_limit_seconds + 300

celery_app = Celery(
    "antroute",
    broker=broker_url,
    backend=result_url,
    include=["app.route_tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_default_queue="routes-primary",
    task_track_started=True,
    task_time_limit=settings.route_job_time_limit_seconds,
    result_expires=settings.route_job_ttl_seconds,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    # A killed child process must produce FAILURE, rather than endlessly requeuing
    # a trip that cannot finish within the execution limit.
    task_reject_on_worker_lost=False,
    broker_connection_timeout=3,
    broker_transport_options={
        "socket_connect_timeout": 3,
        "socket_timeout": 3,
        "max_retries": 0,
        "visibility_timeout": visibility_timeout,
    },
    redis_socket_connect_timeout=3,
    redis_socket_timeout=3,
    result_backend_transport_options={
        "retry_policy": {"max_retries": 0},
        "visibility_timeout": visibility_timeout,
    },
    visibility_timeout=visibility_timeout,
)

# Registration distinguishes unknown/expired IDs from Celery's generic PENDING state.
job_registry = Redis.from_url(
    result_url,
    decode_responses=True,
    socket_connect_timeout=3,
    socket_timeout=3,
)
