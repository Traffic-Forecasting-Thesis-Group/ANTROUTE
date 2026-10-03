import uuid

from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

# Spatial models (routes, congestion segments, event locations) will use
# GeoAlchemy2's Geometry column type once you get to that part of the thesis
# pipeline — e.g. `from geoalchemy2 import Geometry` and
# `Column(Geometry("POINT", srid=4326))`. Not needed for auth, so left out here.