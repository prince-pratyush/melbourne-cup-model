from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.special import softmax

from cup_model.contracts import Finish, Result, Runner, Snapshot


def generate(path: str | Path, days: int = 40, seed: int = 42):
    if days < 10:
        raise ValueError("demo needs at least ten days")
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, 24)
    base = datetime(2018, 1, 1, 4, tzinfo=UTC)
    records = []
    for day in range(days + 1):
        start = base + timedelta(days=day)
        horses = sorted(rng.choice(len(ability), 8, replace=False).tolist())
        weights = rng.uniform(51, 58, 8)
        strength = ability[horses] - 0.08 * (weights - 54)
        market = softmax(strength + rng.normal(0, 0.5, 8))
        race_id = f"synthetic-{day:04d}"
        runners = tuple(
            Runner(
                horse_id=f"synthetic-horse-{horse:02d}",
                name=f"Synthetic Horse {horse}",
                weight_kg=float(weights[i]),
                barrier_drawn=i + 1,
                barrier_effective=i + 1,
                age_years=4 + horse % 5,
                decimal_odds=float(1 / (market[i] * 1.02)),
            )
            for i, horse in enumerate(horses)
        )
        snapshot = Snapshot(
            race_id=race_id,
            scheduled_start=start,
            cutoff=start - timedelta(minutes=5),
            available_at=start - timedelta(minutes=6),
            source="synthetic-generator-v1",
            distance_m=3200 if day % 5 == 0 else 2400,
            going=4 + day % 3,
            venue="Synthetic Park",
            market_operator="synthetic",
            market_at=start - timedelta(minutes=5),
            expected_starters=8,
            runners=runners,
            synthetic=True,
        )
        records.append(snapshot)
        if day == days:
            continue
        ranking = np.argsort(-(strength + rng.gumbel(size=8)))
        positions = {int(index): rank + 1 for rank, index in enumerate(ranking)}
        records.append(
            Result(
                race_id=race_id,
                race_start=start,
                available_at=start + timedelta(hours=1),
                source="synthetic-generator-v1",
                distance_m=snapshot.distance_m,
                going=snapshot.going,
                finishes=tuple(
                    Finish(
                        horse_id=runner.horse_id,
                        position=positions[i],
                        rating=float(80 + ability[horses[i]] * 10 + rng.normal(0, 2)),
                        final_600_s=float(36 - strength[i] * 0.2 + rng.normal(0, 0.5)),
                    )
                    for i, runner in enumerate(runners)
                ),
                synthetic=True,
            )
        )
    Path(path).write_text("".join(r.model_dump_json() + "\n" for r in records))
    return {
        "synthetic": True,
        "records": len(records),
        "target_race": records[-1].race_id,
        "target_cutoff": records[-1].cutoff.isoformat(),
    }
