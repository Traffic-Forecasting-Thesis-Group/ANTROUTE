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