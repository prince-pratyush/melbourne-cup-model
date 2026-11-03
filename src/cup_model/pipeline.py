import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from cup_model import __version__
from cup_model.contracts import canonical, digest, timestamp
from cup_model.features import FEATURES, Race, build_race
from cup_model.models import (
    ConditionalLogit,
    TreeModel,
    calibrate,
    fit_temperature,
    fit_weights,
    select,
)
from cup_model.store import Dataset


def session_date(snapshot):
    return snapshot.scheduled_start.astimezone(ZoneInfo("Australia/Sydney")).date()


class Forecaster:
    def __init__(self, manifest: dict):
        body = {k: v for k, v in manifest.items() if k != "model_id"}
        if manifest.get("model_id") != digest(body):
            raise ValueError("model artifact checksum mismatch")
        if tuple(manifest["features"]) != FEATURES or manifest["format_version"] != 2:
            raise ValueError("unsupported feature schema or artifact version")
        self.manifest = manifest
        self.models = {
            name: TreeModel.from_dict(data) if name == "tree" else ConditionalLogit.from_dict(data)
            for name, data in manifest["models"].items()
        }
        weights = np.asarray(list(manifest["weights"].values()))
        if (
            not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1)
        ):
            raise ValueError("invalid ensemble weights")
        calibrate(np.array([0.5, 0.5]), manifest["temperature"])

    def components(self, race: Race):
        result = {name: model.predict(race) for name, model in self.models.items()}
        if "market" in self.manifest["weights"]:
            if race.market is None:
                raise ValueError("this model requires a coherent market snapshot")
            result["market"] = race.market
        return result

    def probabilities(self, race: Race):
        if timestamp(self.manifest["frozen_at"]) > race.snapshot.cutoff:
            raise ValueError("model was frozen after prediction cutoff")
        if race.snapshot.race_id in self.manifest["fit_race_ids"]:
            raise ValueError("cannot issue a forecast for a race used to fit this artifact")
        if race.snapshot.synthetic != self.manifest["synthetic"]:
            raise ValueError("synthetic and real data cannot be mixed")
        parts = self.components(race)
        mixed = sum(self.manifest["weights"][name] * p for name, p in parts.items())
        return calibrate(mixed, self.manifest["temperature"])

    def predict(self, race: Race):
        probability = self.probabilities(race)
        winner = select(race, probability)
        ranks = sorted(
            range(len(probability)),
            key=lambda i: (
                -probability[i],
                -race.market[i] if race.market is not None else 0,
                race.horse_ids[i],
            ),
        )
        runner_up = ranks[1]
        logit = self.models["logit"]
        transformed = logit.transform(race.x)
        contributions = (transformed[winner] - transformed.mean(axis=0)) * logit.beta
        factors = sorted(range(len(FEATURES)), key=lambda i: -abs(contributions[i]))[:3]
        card = {
            "race_id": race.snapshot.race_id,
            "cutoff": race.snapshot.cutoff.isoformat(),
            "model_id": self.manifest["model_id"],
            "input_hash": race.data_hash,
            "synthetic": race.snapshot.synthetic,
            "timing_provenance": race.snapshot.timing_provenance,
            "source": race.snapshot.source,
            "market_at": race.snapshot.market_at.isoformat() if race.market is not None else None,
            "selected_runner_components": {
                name: float(p[winner]) for name, p in self.components(race).items()
            },
            "logit_score_factors_vs_field": [
                {"feature": FEATURES[i], "score_contribution": float(contributions[i])}
                for i in factors
            ],
            "selection": {
                "horse_id": race.horse_ids[winner],
                "name": race.names[winner],
                "probability": float(probability[winner]),
            },
            "closest_rival": race.horse_ids[runner_up],
            "probability_gap": float(probability[winner] - probability[runner_up]),
            "runners": [
                {
                    "horse_id": race.horse_ids[i],
                    "name": race.names[i],
                    "probability": float(probability[i]),
                    "rank": rank + 1,
                }
                for rank, i in enumerate(ranks)
            ],
        }
        card["prediction_id"] = digest(card)
        return card

    def save(self, path: str | Path):
        Path(path).write_text(canonical(self.manifest) + "\n")

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text()))


def train(
    dataset: Dataset,
    frozen_at: datetime,
    *,
    trees: bool = False,
    l2: float = 1,
    seed: int = 42,
    tree_iterations: int = 500,
    allow_reconstructed: bool = False,
) -> Forecaster:
    snapshots = dataset.latest_snapshots(before=frozen_at)
    results = dataset.results_as_of(frozen_at)
    snapshots = [s for s in snapshots if s.race_id in results]
    timing = sorted({s.timing_provenance for s in snapshots})
    if timing != ["recorded"] and not allow_reconstructed:
        raise ValueError("historical reconstruction requires --allow-reconstructed")
    days = sorted({session_date(s) for s in snapshots})
    if len(days) < 8:
        raise ValueError("need at least eight completed race days for separate fitting windows")
    blend_start = days[max(4, int(len(days) * 0.6))]
    calibration_start = days[max(6, int(len(days) * 0.8))]
    blocks = [
        [s for s in snapshots if session_date(s) < blend_start],
        [s for s in snapshots if blend_start <= session_date(s) < calibration_start],
        [s for s in snapshots if session_date(s) >= calibration_start],
    ]
    boundaries = [min(s.cutoff for s in blocks[1]), min(s.cutoff for s in blocks[2]), frozen_at]
    races = []
    for block, boundary in zip(blocks, boundaries, strict=True):
        if max(s.cutoff for s in block) >= boundary:
            raise ValueError("snapshot editions overlap chronological fitting windows")
        available = dataset.results_as_of(boundary)
        block_races = [build_race(dataset, s, boundary) for s in block if s.race_id in available]
        if len(block_races) < 2:
            raise ValueError("each chronological fitting window needs two available results")
        races.append(block_races)
    fitting, blending, calibration = races
    flags = {r.snapshot.synthetic for block in races for r in block}
    if len(flags) != 1:
        raise ValueError("refusing to train on mixed synthetic and real records")
    market_flags = {r.market is not None for block in races for r in block}
    if len(market_flags) != 1:
        raise ValueError("market coverage must be consistent across fitting windows")
    market = market_flags.pop()
    models = {"logit": ConditionalLogit(l2=l2).fit(fitting)}
    if market:
        models["residual"] = ConditionalLogit(l2=l2, market_offset=True).fit(fitting)
    if trees:
        models["tree"] = TreeModel().fit(fitting, iterations=tree_iterations, seed=seed)
    names = (["market"] if market else []) + list(models)

    def components(block):
        return [
            [r.market if name == "market" else models[name].predict(r) for r in block]
            for name in names
        ]

    weights = fit_weights(blending, components(blending))
    probabilities = [
        sum(w * p for w, p in zip(weights, row, strict=True))
        for row in zip(*components(calibration), strict=True)
    ]
    temperature = fit_temperature(calibration, probabilities)
    packages = ["numpy", "scipy", "pydantic"] + (["catboost"] if trees else [])
    manifest = {
        "format_version": 2,
        "code_version": __version__,
        "features": list(FEATURES),
        "frozen_at": frozen_at.isoformat(),
        "synthetic": flags.pop(),
        "timing_provenance": timing,
        "result_availability_bases": sorted(
            {results[s.race_id].availability_basis for s in snapshots}
        ),
        "seed": seed,
        "python": platform.python_version(),
        "dependencies": {p: version(p) for p in packages},
        "config": {"l2": l2, "trees": trees, "tree_iterations": tree_iterations},
        "models": {name: model.to_dict() for name, model in models.items()},
        "weights": dict(zip(names, weights.tolist(), strict=True)),
        "temperature": temperature,
        "fit_race_ids": [r.snapshot.race_id for block in races for r in block],
        "windows": {
            name: {
                "races": len(block),
                "first_cutoff": block[0].snapshot.cutoff.isoformat(),
                "last_cutoff": block[-1].snapshot.cutoff.isoformat(),
                "label_cutoff": boundary.isoformat(),
                "inputs": {r.snapshot.race_id: r.data_hash for r in block},
                "targets": {r.snapshot.race_id: r.y.tolist() for r in block},
            }
            for name, block, boundary in zip(
                ("base", "blend", "calibration"),
                races,
                boundaries,
                strict=True,
            )
        },
    }
    manifest["model_id"] = digest(manifest)
    return Forecaster(manifest)
