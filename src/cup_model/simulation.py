from typing import Annotated

import numpy as np
from pydantic import Field, model_validator

from cup_model.contracts import Number, Record


class SimRunner(Record):
    horse_id: str = Field(min_length=1)
    mean_time_s: Number = Field(gt=0)
    execution_sd_s: Number = Field(gt=0)
    pace_response_s: Number = 0
    ground_response_s: Number = 0
    exposed_wind_s: Number = 0
    traffic_risk: Number = Field(default=0, ge=0, le=1)
    traffic_cost_s: Number = Field(default=0, ge=0)


class Scenario(Record):
    name: str = Field(min_length=1)
    probability: Number = Field(ge=0, le=1)
    pace: Number = 0
    ground: Number = 0
    wind: Number = 0


class Simulation(Record):
    runners: tuple[SimRunner, ...]
    scenarios: tuple[Scenario, ...]
    draws: Annotated[int, Field(ge=100, le=1_000_000)] = 20000
    seed: int = 42
    parameter_source: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_simulation(self):
        if len(self.runners) < 2:
            raise ValueError("simulation needs at least two runners")
        if len({r.horse_id for r in self.runners}) != len(self.runners):
            raise ValueError("duplicate simulated runner")
        if not self.scenarios or not np.isclose(sum(s.probability for s in self.scenarios), 1):
            raise ValueError("scenario probabilities must sum to one")
        if len({s.name for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("scenario names must be unique")
        return self


def simulate(config: Simulation):
    rng = np.random.default_rng(config.seed)
    runners = sorted(config.runners, key=lambda r: r.horse_id)
    n = len(runners)
    scenario_ids = rng.choice(
        len(config.scenarios),
        config.draws,
        p=[s.probability for s in config.scenarios],
    )
    times = np.zeros((config.draws, n))
    for i, runner in enumerate(runners):
        response = np.array(
            [
                s.pace * runner.pace_response_s
                + s.ground * runner.ground_response_s
                + s.wind * runner.exposed_wind_s
                for s in config.scenarios
            ]
        )
        times[:, i] = runner.mean_time_s + response[scenario_ids]
        times[:, i] += rng.normal(0, runner.execution_sd_s, config.draws)
    # Shared congestion links trip risk across runners.
    congestion = rng.uniform(size=config.draws)
    for i, runner in enumerate(runners):
        times[:, i] += (congestion < runner.traffic_risk) * runner.traffic_cost_s
    if np.any(times <= 0):
        raise ValueError("parameters generated impossible finish times")
    wins = np.argmin(times, axis=1)
    probabilities = np.bincount(wins, minlength=n) / config.draws
    selected = int(np.argmax(probabilities))
    return {
        "experimental": True,
        "ensemble_weight": 0,
        "parameter_source": config.parameter_source,
        "draws": config.draws,
        "seed": config.seed,
        "selection": runners[selected].horse_id,
        "runners": [
            {
                "horse_id": r.horse_id,
                "probability": float(p),
                "monte_carlo_se": float(np.sqrt(p * (1 - p) / config.draws)),
            }
            for r, p in zip(runners, probabilities, strict=True)
        ],
        "scenarios": [
            {
                "name": scenario.name,
                "draws": int(np.sum(scenario_ids == i)),
                "probabilities": (
                    np.bincount(wins[scenario_ids == i], minlength=n) / np.sum(scenario_ids == i)
                ).tolist()
                if np.any(scenario_ids == i)
                else None,
            }
            for i, scenario in enumerate(config.scenarios)
        ],
    }
