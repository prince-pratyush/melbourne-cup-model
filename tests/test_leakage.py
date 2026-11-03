from datetime import timedelta

import numpy as np
import pytest

from cup_model.contracts import Result
from cup_model.features import FEATURES, build_race
from cup_model.store import Dataset, Store


def test_future_result_changes_do_not_change_past_features(dataset):
    target = dataset.latest_snapshots()[12]
    before = build_race(dataset, target)
    records = dataset.snapshots.copy()
    for result in dataset.results:
        data = result.model_dump(mode="json")
        if result.available_at > target.cutoff:
            for finish in data["finishes"]:
                finish["rating"] = 99999
        records.append(Result.model_validate(data))
    after = build_race(Dataset(records), target)
    np.testing.assert_equal(before.x, after.x)
    assert before.data_hash == after.data_hash


def test_result_revision_uses_publication_time(dataset):
    target = dataset.latest_snapshots()[12]
    prior = dataset.results_as_of(target.cutoff)
    old = max(prior.values(), key=lambda r: r.race_start)
    revision = old.model_dump(mode="json")
    revision["available_at"] = (target.cutoff + timedelta(days=1)).isoformat()
    for finish in revision["finishes"]:
        finish["rating"] = 10000
    revised = Dataset(dataset.snapshots + dataset.results + [Result.model_validate(revision)])
    np.testing.assert_equal(build_race(dataset, target).x, build_race(revised, target).x)


def test_last_run_and_feature_calculation(dataset):
    target = dataset.latest_snapshots()[10]
    race = build_race(dataset, target)
    for i, horse_id in enumerate(race.horse_ids):
        history = [
            result
            for result in dataset.results_as_of(target.cutoff).values()
            if any(f.horse_id == horse_id for f in result.finishes)
        ]
        assert race.x[i, FEATURES.index("history_count")] == len(history)
        if history:
            last = max(history, key=lambda r: r.race_start)
            expected = (target.cutoff - last.race_start).total_seconds() / 86400
            assert race.x[i, FEATURES.index("days_since_run")] == pytest.approx(expected)


def test_import_is_atomic_immutable_and_idempotent(dataset, tmp_path):
    store = Store(":memory:")
    data = dataset.latest_snapshots()[0].model_dump(mode="json")
    path = tmp_path / "records.jsonl"
    import json

    path.write_text(json.dumps(data) + "\n")
    assert store.import_jsonl(path) == 1
    assert store.import_jsonl(path) == 1
    assert len(store.records()) == 1
    changed = dict(data, going=7)
    second = dataset.latest_snapshots()[1].model_dump(mode="json")
    path.write_text(json.dumps(second) + "\n" + json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="immutable"):
        store.import_jsonl(path)
    assert len(store.records()) == 1
    store.close()


def test_target_labels_cannot_be_used_early(dataset):
    target = dataset.latest_snapshots()[5]
    with pytest.raises(ValueError, match="result unavailable"):
        build_race(dataset, target, target.cutoff)
