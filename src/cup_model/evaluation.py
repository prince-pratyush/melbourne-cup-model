from datetime import datetime

import numpy as np

from cup_model.contracts import digest
from cup_model.features import Race, build_race
from cup_model.models import select
from cup_model.pipeline import session_date, train
from cup_model.store import Dataset


def race_metrics(race: Race, probabilities: np.ndarray):
    if race.y is None:
        raise ValueError("evaluation needs an official result")
    chosen = select(race, probabilities)
    rank = sorted(
        range(len(probabilities)),
        key=lambda i: (
            -probabilities[i],
            -race.market[i] if race.market is not None else 0,
            race.horse_ids[i],
        ),
    )
    winner_rank = min(rank.index(i) + 1 for i in np.flatnonzero(race.y))
    return {
        "top_one": float(race.y[chosen] > 0),
        "log_loss": float(-np.dot(race.y, np.log(np.clip(probabilities, 1e-15, 1)))),
        "brier": float(np.sum((probabilities - race.y) ** 2)),
        "winner_rank": winner_rank,
        "top_three": float(winner_rank <= 3),
    }


def paired_interval(rows: list[dict], seed: int = 42, samples: int = 2000):
    groups = sorted({r["session"] for r in rows})
    if len(groups) < 2 or not all("market" in r["metrics"] for r in rows):
        return None
    blocks = [
        np.array(
            [
                r["metrics"]["model"]["log_loss"] - r["metrics"]["market"]["log_loss"]
                for r in rows
                if r["session"] == group
            ]
        )
        for group in groups
    ]
    rng = np.random.default_rng(seed)
    estimates = [
        np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]).mean()
        for _ in range(samples)
    ]
    return {
        "metric": "model_minus_market_log_loss",
        "method": "paired_race_day_bootstrap",
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "race_days": len(groups),
        "samples": samples,
    }


def backtest(
    dataset: Dataset,
    start: datetime,
    score_as_of: datetime,
    *,
    trees: bool = False,
    l2: float = 1,
    seed: int = 42,
    tree_iterations: int = 500,
):
    if start >= score_as_of:
        raise ValueError("backtest start must precede score cutoff")
    results = dataset.results_as_of(score_as_of)
    snapshots = [
        s
        for s in dataset.latest_snapshots(before=score_as_of)
        if s.cutoff >= start and s.race_id in results
    ]
    days = sorted({session_date(s) for s in snapshots})
    rows = []
    reliability = [[] for _ in range(10)]
    for day in days:
        targets = [s for s in snapshots if session_date(s) == day]
        frozen_at = min(s.cutoff for s in targets)
        forecaster = train(
            dataset,
            frozen_at,
            trees=trees,
            l2=l2,
            seed=seed,
            tree_iterations=tree_iterations,
        )
        for snapshot in targets:
            race = build_race(dataset, snapshot, score_as_of)
            p = forecaster.probabilities(race)
            baseline = np.full(len(p), 1 / len(p))
            metrics = {"model": race_metrics(race, p), "uniform": race_metrics(race, baseline)}
            if race.market is not None:
                metrics["market"] = race_metrics(race, race.market)
            for estimate, observed in zip(p, race.y, strict=True):
                reliability[min(9, int(estimate * 10))].append((float(estimate), float(observed)))
            rows.append(
                {
                    "race_id": snapshot.race_id,
                    "session": day.isoformat(),
                    "prediction": forecaster.predict(race),
                    "metrics": metrics,
                    "official_result_hash": digest(
                        results[snapshot.race_id].model_dump(mode="json")
                    ),
                }
            )
    if not rows:
        raise ValueError("no completed races in backtest window")
    summary = {
        name: {
            metric: float(np.mean([r["metrics"][name][metric] for r in rows]))
            for metric in rows[0]["metrics"][name]
        }
        for name in rows[0]["metrics"]
        if all(name in r["metrics"] for r in rows)
    }
    return {
        "evaluation": "retrospective_walk_forward",
        "race_count": len(rows),
        "start": start.isoformat(),
        "score_as_of": score_as_of.isoformat(),
        "synthetic": rows[0]["prediction"]["synthetic"],
        "summary": summary,
        "paired_interval": paired_interval(rows, seed),
        "reliability": [
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "runners": len(bin_rows),
                "mean_probability": float(np.mean([p for p, _ in bin_rows])) if bin_rows else None,
                "observed_share": float(np.mean([y for _, y in bin_rows])) if bin_rows else None,
            }
            for i, bin_rows in enumerate(reliability)
        ],
        "races": rows,
    }
