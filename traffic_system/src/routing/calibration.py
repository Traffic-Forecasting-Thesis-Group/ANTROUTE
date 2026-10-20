"""Development-only calibration of contract version 2: split, mappings, time profiles, CRITIC weights, maxima.

Nothing here reads tuning or test dates. Every fitted value (class severities, time profiles, hourly CRITIC
weights, development maxima, Cheng references) comes from the declared development period only.

Hourly CRITIC (primary estimator, not pooled):
  for each eligible development sample hour h, w[., h] = CRITIC over that hour's road segments, with constant
  criteria at 0; hours where CRITIC is undefined are excluded whole and logged;
  fixed_weight[j] = sum_{h in H} w[j, h] / |H| over the common valid-hour set H, zeros kept.
Within-hour min-max normalization belongs to CRITIC only; routing uses the separate development maxima.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

from .critic import CriticAccumulator, UndefinedCriticError
from .criteria import CRITERIA

SPLIT_NAMES = ("development", "tuning", "test")
STUDY_START, STUDY_END = date(2025, 6, 1), date(2026, 5, 31)


class CalibrationBlocked(RuntimeError):
    """Calibration needs inputs that are not declared (for example, the development/tuning/test boundaries)."""


class CalibrationError(RuntimeError):
    """Calibration ran but cannot produce a valid result (for example, every sampled hour was undefined)."""


def load_split(policy: dict) -> dict[str, tuple[date, date]]:
    """Declared, chronological, non-overlapping split inside the study period; blocked when any boundary is unset."""
    declared = policy.get("calibration_split") or {}
    missing = [name for name in SPLIT_NAMES if not (declared.get(name) or {}).get("start") or not (declared.get(name) or {}).get("end")]
    if missing:
        raise CalibrationBlocked(
            "Development/tuning/test boundaries are not declared (configs/historical_study.json calibration_split: "
            f"{', '.join(missing)} unset). Full-data calibration is blocked; no dates are invented.")
    split = {name: (date.fromisoformat(declared[name]["start"]), date.fromisoformat(declared[name]["end"])) for name in SPLIT_NAMES}
    previous_end = None
    for name in SPLIT_NAMES:
        start, end = split[name]
        if not STUDY_START <= start <= end <= STUDY_END:
            raise ValueError(f"{name} period must lie within June 1, 2025 - May 31, 2026")
        if previous_end is not None and start <= previous_end:
            raise ValueError("Split periods must be chronological and must not overlap")
        previous_end = end
    return split


def days(period: tuple[date, date], step: int = 1):
    day = period[0]
    while day <= period[1]:
        yield day
        day += timedelta(days=step)


def class_severity(speeds_by_class: dict[str, list[float]]) -> dict[str, float]:
    """p_k = 1 - median free-flow speed of class k / the largest class median (an assumed preference mapping)."""
    medians = {name: statistics.median(values) for name, values in speeds_by_class.items() if values}
    if not medians:
        raise CalibrationError("No road speeds to derive class severity")
    top = max(medians.values())
    return {name: round(1 - median / top, 6) for name, median in sorted(medians.items())}


@dataclass
class ProfileFit:
    """Mean congestion by road group, day category and local hour, accumulated over development hours."""
    groups: dict[str, str]  # road_id -> group key "city|road_type"
    road_types: dict[str, str]  # road_id -> road type
    sums: dict = field(default_factory=lambda: defaultdict(float))
    counts: dict = field(default_factory=lambda: defaultdict(int))

    def add(self, category: str, hour: int, congestion: dict[str, float]) -> None:
        for road_id, value in congestion.items():
            for key in ((self.groups[road_id], category, hour), ("type:" + self.road_types[road_id], category, hour),
                        ("all", category, hour)):
                self.sums[key] += value
                self.counts[key] += 1

    def profiles(self) -> tuple[dict[str, dict[str, list[float]]], dict[str, int]]:
        """Per-road {category: 24 values}, falling back group -> road type -> all roads; and fallback counts."""
        fallbacks, cache, result = defaultdict(int), {}, {}
        for road_id, group in self.groups.items():
            if group not in cache:
                profile = {}
                for category in ("weekday", "weekend"):
                    values = []
                    for hour in range(24):
                        for level, key in (("group", group), ("road_type", "type:" + self.road_types[road_id]), ("all", "all")):
                            if self.counts[(key, category, hour)]:
                                values.append(round(self.sums[(key, category, hour)] / self.counts[(key, category, hour)], 6))
                                fallbacks[level] += 1
                                break
                        else:
                            raise CalibrationError(f"No development observations for {category} hour {hour}")
                    profile[category] = values
                cache[group] = profile
            result[road_id] = cache[group]
        return result, dict(fallbacks)


def hourly_critic(hours) -> dict:
    """hours: iterable of (hour_id, rows | None, reason). rows: [roads x ten criteria]; None marks an ineligible hour.

    Returns the fixed weights (mean over the common valid-hour set H, zeros kept), the complete hourly vectors,
    active masks, valid hour ids and excluded hours with reasons. Raises CalibrationError when H is empty.
    """
    vectors, masks, valid, excluded = {}, {}, [], {}
    for hour_id, rows, reason in hours:
        if rows is None:
            excluded[hour_id] = reason
            continue
        accumulator = CriticAccumulator(CRITERIA)
        accumulator.add(rows)
        try:
            result = accumulator.result()
        except UndefinedCriticError as error:
            excluded[hour_id] = f"undefined: {error}"
            continue
        vectors[hour_id] = [result["weights"][name] for name in CRITERIA]
        masks[hour_id] = [result["active"][name] for name in CRITERIA]
        valid.append(hour_id)
    if not valid:
        raise CalibrationError("CRITIC is undefined in every sampled development hour")
    weights = {name: sum(vectors[hour][j] for hour in valid) / len(valid) for j, name in enumerate(CRITERIA)}
    return {"weights": weights, "valid_hours": valid, "excluded_hours": excluded,
            "hourly_weights": vectors, "active_masks": masks, "criteria": list(CRITERIA)}
