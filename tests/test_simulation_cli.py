import json

import pytest

from cup_model.cli import main
from cup_model.simulation import Simulation, simulate


def test_shared_scenarios_change_winner_and_are_reproducible():
    data = {
        "runners": [
            {"horse_id": "a", "mean_time_s": 200, "execution_sd_s": 0.01, "pace_response_s": 2},
            {"horse_id": "b", "mean_time_s": 201, "execution_sd_s": 0.01, "pace_response_s": -2},
        ],
        "scenarios": [
            {"name": "slow", "probability": 0.5, "pace": -1},
            {"name": "fast", "probability": 0.5, "pace": 1},
        ],
        "draws": 2000,
        "parameter_source": "synthetic-test",
    }
    config = Simulation.model_validate(data)
    result = simulate(config)
    assert result == simulate(config)
    assert result["experimental"] and result["ensemble_weight"] == 0
    assert result["scenarios"][0]["probabilities"] == [1, 0]
    assert result["scenarios"][1]["probabilities"] == [0, 1]
    assert sum(r["probability"] for r in result["runners"]) == pytest.approx(1)


def test_cli_end_to_end(tmp_path):
    raw = str(tmp_path / "demo.jsonl")
    db = str(tmp_path / "races.db")
    artifact = str(tmp_path / "model.json")
    card = str(tmp_path / "prediction.json")
    main(["demo", "--output", raw, "--days", "12"])
    target = json.loads(open(raw).readlines()[-1])
    main(["ingest", "--db", db, "--input", raw])
    main(["train", "--db", db, "--frozen-at", target["cutoff"], "--output", artifact])
    main(
        [
            "predict",
            "--db",
            db,
            "--model",
            artifact,
            "--race-id",
            target["race_id"],
            "--output",
            card,
        ]
    )
    prediction = json.loads(open(card).read())
    assert prediction["synthetic"] is True
    assert prediction["selection"]["horse_id"].startswith("synthetic-")
    assert "recorded_at" in prediction
