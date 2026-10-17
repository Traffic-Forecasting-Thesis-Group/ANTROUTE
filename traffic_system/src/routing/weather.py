"""Observed weather collection and its explicitly estimated routing effects."""
import os
from datetime import datetime, timezone

import requests
from shapely.geometry import shape

from .network import WeatherEffect


def weather_condition(observation: dict) -> str:
    current = observation.get("current", {})
    precipitation = float(current.get("precip") or 0)
    description = " ".join(current.get("weather_descriptions", [])).lower()
    if precipitation >= 7.5 or "heavy rain" in description or "torrential" in description:
        return "heavy_rain"
    if precipitation > 0 or any(word in description for word in ("rain", "drizzle", "thunder")):
        return "rain"
    return "clear"


def observed_effect(observation: dict, scenarios: dict) -> WeatherEffect:
    return WeatherEffect.model_validate(scenarios[weather_condition(observation)])


def fetch_current_weather(boundaries: dict) -> dict:
    key = os.getenv("WEATHERSTACK_API_KEY", "")
    if not key:
        raise ValueError("WEATHERSTACK_API_KEY is not configured")
    observations = {}
    for city, geometry in boundaries.items():
        point = shape(geometry).representative_point()
        try:
            response = requests.get("https://api.weatherstack.com/current", params={
                "access_key": key, "query": f"{point.y},{point.x}", "units": "m",
            }, timeout=10)
        except requests.RequestException:
            raise RuntimeError(f"Weather connection failed for {city}") from None
        if response.status_code != 200:
            raise RuntimeError(f"Weather request failed for {city}: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(f"Invalid weather response for {city}") from None
        if "error" in data:
            error = data["error"]
            # The numeric code is safe to report; provider messages can reflect credentials.
            code = error.get("code")
            code = code if isinstance(code, int) else "unknown"
            raise RuntimeError(f"Weather provider rejected request: code {code}")
        if not data.get("current"):
            raise RuntimeError(f"Weather observation missing for {city}")
        observations[city] = {
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "query_coordinates": [point.x, point.y],
            "location": data.get("location", {}), "current": data["current"],
        }
    return {"source": "Weatherstack", "cities": observations}
