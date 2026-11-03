from datetime import datetime

import numpy as np

from cup_model.evaluation import paired_interval, race_metrics
from cup_model.features import FEATURES, build_race
from cup_model.pipeline import session_date, train
from cup_model.store import Dataset


def summarize(rows):
    if not rows:
        return {"races": 0, "models": {}}
    names = set.intersection(*(set(row["metrics"]) for row in rows))
    return {
        "races": len(rows),
        "models": {
            name: {
                "wins_selected": int(sum(row["metrics"][name]["top_one"] for row in rows)),
                **{
                    metric: float(np.mean([row["metrics"][name][metric] for row in rows]))
                    for metric in rows[0]["metrics"][name]
                },
            }
            for name in sorted(names)
        },
    }


def historical_audit(
    dataset: Dataset,
    frozen_at: datetime,
    score_as_of: datetime,
    *,
    trees=False,
    tree_iterations=500,
    allow_reconstructed=False,
    l2=1,
    seed=42,
):
    if frozen_at >= score_as_of:
        raise ValueError("model freeze must precede evaluation cutoff")
    model = train(
        dataset,
        frozen_at,
        trees=trees,
        tree_iterations=tree_iterations,
        allow_reconstructed=allow_reconstructed,
        l2=l2,
        seed=seed,
    )
    results = dataset.results_as_of(score_as_of)
    targets = [
        s
        for s in dataset.latest_snapshots(before=score_as_of)
        if s.cutoff >= frozen_at and s.race_id in results
    ]
    rows = []
    missing = np.zeros(len(FEATURES), dtype=int)
    runner_count = 0
    for snapshot in targets:
        race = build_race(dataset, snapshot, score_as_of)
        missing += np.sum(~np.isfinite(race.x), axis=0)
        runner_count += len(race.horse_ids)
        parts = model.components(race)
        parts["model"] = model.probabilities(race)
        parts["uniform"] = np.full(len(race.horse_ids), 1 / len(race.horse_ids))
        rows.append(
            {
                "race_id": snapshot.race_id,
                "session": session_date(snapshot).isoformat(),
                "venue": snapshot.venue,
                "distance_m": snapshot.distance_m,
                "is_melbourne_cup": snapshot.venue == "Flemington" and snapshot.distance_m == 3200,
                "prediction": model.predict(race),
                "metrics": {
                    name: race_metrics(race, p, market_tiebreak=name != "uniform")
                    for name, p in parts.items()
                },
                "winner_ids": [race.horse_ids[i] for i in np.flatnonzero(race.y)],
                "official_result_hash": dataset.result_hash(results[snapshot.race_id]),
            }
        )
    if not rows:
        raise ValueError("no completed races after model freeze")
    years = sorted({r["session"][:4] for r in rows})
    report = {
        "evaluation": "retrospective_fixed_model_chronological_holdout",
        "frozen_at": frozen_at.isoformat(),
        "score_as_of": score_as_of.isoformat(),
        "model_id": model.manifest["model_id"],
        "synthetic": model.manifest["synthetic"],
        "timing_provenance": model.manifest["timing_provenance"],
        "result_availability_bases": model.manifest["result_availability_bases"],
        "training_windows": {
            name: {key: value for key, value in data.items() if key not in ("inputs", "targets")}
            for name, data in model.manifest["windows"].items()
        },
        "ensemble_weights": model.manifest["weights"],
        "temperature": model.manifest["temperature"],
        "overall": summarize(rows),
        "by_year": {
            year: summarize([r for r in rows if r["session"].startswith(year)]) for year in years
        },
        "melbourne_cup": summarize([r for r in rows if r["is_melbourne_cup"]]),
        "paired_interval": paired_interval(rows, seed=seed),
        "feature_missing_fraction": dict(
            zip(FEATURES, (missing / runner_count).tolist(), strict=True)
        ),
        "races": rows,
    }
    return model, report
