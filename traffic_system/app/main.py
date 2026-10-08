import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import risk_routing
from app.database import Base, engine
from app.routers import auth, places, routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    # MVP-only table creation. Switch to Alembic migrations once your schema
    # stabilizes and you add the spatial (GeoAlchemy2) tables for routes/events.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Load the road network and incident feed now (~40s) rather than inside the first
    # route request; a request that arrives before it finishes waits for it.
    threading.Thread(target=risk_routing.available, daemon=True).start()
    yield


app = FastAPI(title="Traffic API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(routes.router)
app.include_router(places.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/model")
async def model_health():
    """
    Which scored artefact this server is serving routes from, so a demo can answer "is this
    really the new model?" from the system rather than from someone's recollection.

    Loading the network is the expensive part of a first request, so a failure here is
    returned as a readable message instead of a 500 -- "the file is missing" is exactly the
    answer being asked for.
    """
    from app.risk_routing import model_provenance

    try:
        return model_provenance()
    except FileNotFoundError as exc:
        return {"status": "no risk_edges.csv", "expected_at": str(exc)}
    except Exception as exc:  # noqa: BLE001 -- a diagnostic endpoint must not itself 500
        return {"status": "could not read the scored artefact", "error": f"{type(exc).__name__}: {exc}"}