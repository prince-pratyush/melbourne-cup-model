from datetime import timedelta

import pytest

from cup_model.contracts import Snapshot


def test_rejects_late_or_incomplete_market(dataset):
    snapshot = dataset.latest_snapshots()[-1].model_dump(mode="json")
    snapshot["market_at"] = "2030-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="market must"):
        Snapshot.model_validate(snapshot)
    snapshot = dataset.latest_snapshots()[-1].model_dump(mode="json")
    snapshot["runners"][0]["decimal_odds"] = None
    with pytest.raises(ValueError, match="complete prices"):
        Snapshot.model_validate(snapshot)


@pytest.mark.parametrize("mutation", ["duplicate", "unresolved", "late", "naive", "incomplete"])
def test_rejects_invalid_fields(dataset, mutation):
    original = dataset.latest_snapshots()[-1]
    data = original.model_dump(mode="json")
    if mutation == "duplicate":
        data["runners"][0]["horse_id"] = data["runners"][1]["horse_id"]
    elif mutation == "unresolved":
        data["runners"][0]["status"] = "unresolved"
    elif mutation == "late":
        data["available_at"] = (original.cutoff + timedelta(seconds=1)).isoformat()
    elif mutation == "naive":
        data["cutoff"] = "2018-01-01T03:55:00"
    elif mutation == "incomplete":
        data["expected_starters"] += 1
    with pytest.raises(ValueError):
        Snapshot.model_validate(data)
