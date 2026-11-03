from dataclasses import dataclass
from datetime import datetime

import numpy as np

from cup_model.contracts import Snapshot, digest
from cup_model.store import Dataset

FEATURES = (
    "weight_kg",
    "draw_percentile",
    "effective_draw_missing",
    "age_years",
    "weight_vs_field",
    "age_vs_field",
    "history_count",
    "days_since_run",
    "starts_60d",
    "form_120d",
    "form_sd",
    "last_form",
    "staying_form",
    "ground_form",
    "rating_120d",
    "last_rating",
    "rating_trend",
    "rating_sd",
    "rating_missing",
    "distance_extrapolation",
    "last_distance_change",
    "last_final_600",
    "sectional_missing",
    "distance_age",
    "ground_weight",
    "win_rate_120d",
    "market_surplus_120d",
    "weight_missing",
    "age_missing",
    "draw_missing",
    "going_missing",
)


@dataclass
class Race:
    snapshot: Snapshot
    horse_ids: tuple[str, ...]
    names: tuple[str, ...]
    x: np.ndarray
    market: np.ndarray | None
    y: np.ndarray | None
    data_hash: str


def market_probabilities(odds) -> np.ndarray:
    values = np.asarray(odds, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("market needs finite decimal odds for every runner")
    if np.any(values <= 1):
        raise ValueError("decimal odds must exceed one")
    inverse = 1 / values
    return inverse / inverse.sum()


def build_race(dataset: Dataset, snapshot: Snapshot, labels_as_of: datetime | None = None) -> Race:
    active = sorted((r for r in snapshot.runners if r.status == "active"), key=lambda r: r.horse_id)
    weights = np.array([r.weight_kg for r in active], dtype=float)
    ages = np.array([r.age_years for r in active], dtype=float)
    weight_mean = np.nanmean(weights) if np.isfinite(weights).any() else np.nan
    age_mean = np.nanmean(ages) if np.isfinite(ages).any() else np.nan
    barriers = [r.barrier_effective or r.barrier_drawn for r in active]
    draw_max = max((b for b in barriers if b is not None), default=1)
    rows = []
    used = {}
    for runner in active:
        past = [
            (result, finish)
            for result, finish in dataset.horse_history(runner.horse_id, snapshot.cutoff)
            if result.race_id != snapshot.race_id
        ]
        for result, _ in past:
            if result.synthetic != snapshot.synthetic:
                raise ValueError("synthetic and real histories cannot be mixed")
            used[result.race_id] = result
        days = np.array([(snapshot.cutoff - r.race_start).total_seconds() / 86400 for r, _ in past])
        decay = np.exp2(-days / 120)
        form = np.array(
            [
                np.nan
                if f.status == "unplaced_unknown"
                else 1 - ((f.position or len(r.finishes)) - 1) / (len(r.finishes) - 1)
                for r, f in past
            ]
        )
        wins = np.array([float(f.position == 1) for _, f in past])
        surplus = np.array(
            [
                float(f.position == 1) - f.prior_market_probability
                if f.prior_market_probability is not None
                else np.nan
                for _, f in past
            ]
        )
        rating = np.array([f.rating if f.rating is not None else np.nan for _, f in past])

        def average(values, mask=None, decay=decay):
            valid = np.isfinite(values)
            if mask is not None:
                valid &= mask
            return float(np.average(values[valid], weights=decay[valid])) if valid.any() else np.nan

        ratings_valid = rating[np.isfinite(rating)]
        last_rating = ratings_valid[-1] if len(ratings_valid) else np.nan
        last = past[-1] if past else None
        staying = np.array([r.distance_m >= 2400 for r, _ in past], dtype=bool)
        ground = np.array(
            [
                r.going is not None
                and snapshot.going is not None
                and abs(r.going - snapshot.going) <= 1
                for r, _ in past
            ],
            dtype=bool,
        )
        barrier = runner.barrier_effective or runner.barrier_drawn
        age = runner.age_years if runner.age_years is not None else np.nan
        weight = runner.weight_kg if runner.weight_kg is not None else np.nan
        rows.append(
            [
                weight,
                barrier / draw_max if barrier is not None else np.nan,
                float(runner.barrier_effective is None),
                age,
                weight - weight_mean,
                age - age_mean,
                len(past),
                min(days) if len(days) else np.nan,
                float(np.sum(days <= 60)),
                average(form),
                float(np.nanstd(form)) if np.isfinite(form).any() else np.nan,
                form[-1] if len(form) else np.nan,
                average(form, staying),
                average(form, ground),
                average(rating),
                last_rating,
                last_rating - average(rating),
                float(np.std(ratings_valid)) if len(ratings_valid) else np.nan,
                float(not len(ratings_valid)),
                max(0, snapshot.distance_m - max(r.distance_m for r, _ in past)) / 1000
                if past
                else np.nan,
                (snapshot.distance_m - last[0].distance_m) / 1000 if last else np.nan,
                last[1].final_600_s if last and last[1].final_600_s is not None else np.nan,
                float(not last or last[1].final_600_s is None),
                snapshot.distance_m / 1000 * age,
                snapshot.going * (weight - weight_mean) if snapshot.going is not None else np.nan,
                average(wins),
                average(surplus),
                float(runner.weight_kg is None),
                float(runner.age_years is None),
                float(barrier is None),
                float(snapshot.going is None),
            ]
        )
    market = None
    if all(r.decimal_odds is not None for r in active):
        market = market_probabilities([r.decimal_odds for r in active])
    target = None
    if labels_as_of is not None:
        result = dataset.result_as_of(snapshot.race_id, labels_as_of)
        if result is None:
            raise ValueError(f"result unavailable for {snapshot.race_id} at label cutoff")
        if result.synthetic != snapshot.synthetic:
            raise ValueError("synthetic flag mismatch")
        if {r.horse_id for r in result.finishes} != {r.horse_id for r in active}:
            raise ValueError("official starters do not match snapshot; refresh scratchings")
        if snapshot.cutoff >= result.race_start:
            raise ValueError("snapshot cutoff is after actual start")
        winners = {f.horse_id for f in result.finishes if f.position == 1}
        target = np.array([float(r.horse_id in winners) / len(winners) for r in active])
    provenance = [snapshot.model_dump(mode="json")]
    provenance.extend(dataset.result_hash(used[k]) for k in sorted(used))
    return Race(
        snapshot,
        tuple(r.horse_id for r in active),
        tuple(r.name for r in active),
        np.array(rows, dtype=float),
        market,
        target,
        digest(provenance),
    )
