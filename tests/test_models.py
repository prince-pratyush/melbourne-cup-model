import importlib.util
from datetime import timedelta

import numpy as np
import pytest

from cup_model.contracts import Result, Snapshot
from cup_model.evaluation import backtest, race_metrics
from cup_model.features import build_race, market_probabilities
from cup_model.models import ConditionalLogit, TreeModel, calibrate, select
from cup_model.pipeline import Forecaster, train
from cup_model.store import Dataset, Store


def test_logit_learns_race_level_signal(dataset):
    cutoff = dataset.latest_snapshots()[-1].cutoff
    races = [build_race(dataset, s, cutoff) for s in dataset.latest_snapshots()[:-1]]
    for race in races:
        race.x[:] = 0
        race.x[:, 0] = race.y
    model = ConditionalLogit(l2=0.01).fit(races)
    assert model.beta[0] > 0
    assert all(r.y[np.argmax(model.predict(r))] > 0 for r in races)


def test_market_offset_zero_correction(dataset):
    race = build_race(dataset, dataset.latest_snapshots()[-1])
    model = ConditionalLogit(market_offset=True)
    model.median = model.mean = model.beta = np.zeros(race.x.shape[1])
    model.scale = np.ones(race.x.shape[1])
    np.testing.assert_allclose(model.predict(race), race.market)
    np.testing.assert_allclose(market_probabilities([2, 4, 4]), [0.5, 0.25, 0.25])


def test_uniform_audit_does_not_use_market_to_choose_a_favourite(dataset):
    race = build_race(dataset, dataset.latest_snapshots()[5])
    race.y = np.zeros(len(race.horse_ids))
    race.y[0] = 1
    race.market = np.arange(1, len(race.horse_ids) + 1, dtype=float)
    race.market /= race.market.sum()
    uniform = np.ones(len(race.horse_ids)) / len(race.horse_ids)
    assert race_metrics(race, uniform, market_tiebreak=False)["top_one"] == 1
    assert race_metrics(race, uniform)["top_one"] == 0


def test_pipeline_roundtrip_reproducibility_and_ledger(dataset, tmp_path):
    target = dataset.latest_snapshots()[-1]
    model = train(dataset, target.cutoff)
    repeated = train(dataset, target.cutoff)
    assert model.manifest["model_id"] == repeated.manifest["model_id"]
    path = tmp_path / "model.json"
    model.save(path)
    loaded = Forecaster.load(path)
    race = build_race(dataset, target)
    card = model.predict(race)
    assert card == loaded.predict(race)
    assert sum(r["probability"] for r in card["runners"]) == pytest.approx(1)
    assert len([r for r in card["runners"] if r["rank"] == 1]) == 1
    store = Store(":memory:")
    saved = store.save_prediction(card)
    assert saved == store.save_prediction(card)
    assert "recorded_at" in saved
    store.close()
    path.write_text(path.read_text().replace('"temperature":', '"temperature":999,"unused":'))
    with pytest.raises(ValueError, match="checksum"):
        Forecaster.load(path)


def test_future_data_does_not_change_fitted_model(dataset):
    target = dataset.latest_snapshots()[22]
    model = train(dataset, target.cutoff)
    changed = []
    for result in dataset.results:
        data = result.model_dump(mode="json")
        if result.available_at > target.cutoff:
            for finish in data["finishes"]:
                finish["rating"] = -99999
        changed.append(Result.model_validate(data))
    revised = train(Dataset(dataset.snapshots + changed), target.cutoff)
    assert model.manifest["model_id"] == revised.manifest["model_id"]


def test_scratching_requires_rerun_and_cannot_be_selected(dataset):
    target = dataset.latest_snapshots()[-1]
    model = train(dataset, target.cutoff)
    original = model.predict(build_race(dataset, target))
    data = target.model_dump(mode="json")
    removed = original["selection"]["horse_id"]
    for runner in data["runners"]:
        if runner["horse_id"] == removed:
            runner["status"] = "scratched"
    data["expected_starters"] -= 1
    revised = build_race(dataset, Snapshot.model_validate(data))
    prediction = model.predict(revised)
    assert prediction["selection"]["horse_id"] != removed
    assert len(prediction["runners"]) == len(original["runners"]) - 1
    assert prediction["prediction_id"] != original["prediction_id"]


def test_dead_heat_targets_and_single_selection(dataset):
    snapshot = dataset.latest_snapshots()[10]
    original = next(r for r in dataset.results if r.race_id == snapshot.race_id)
    data = original.model_dump(mode="json")
    for i, finish in enumerate(data["finishes"]):
        finish["position"] = 1 if i < 2 else i + 1
    result = Result.model_validate(data)
    revised = Dataset(dataset.snapshots + [r for r in dataset.results if r != original] + [result])
    race = build_race(revised, snapshot, result.available_at)
    assert sorted(race.y[race.y > 0]) == [0.5, 0.5]
    p = np.full(len(race.y), 0.1 / (len(race.y) - 1))
    winner = np.flatnonzero(race.y)[0]
    p[winner] = 0.9
    assert race_metrics(race, p)["top_one"] == 1
    assert select(race, p) == winner


def test_rejects_in_sample_and_late_model(dataset):
    target = dataset.latest_snapshots()[-1]
    model = train(dataset, target.cutoff)
    earlier = build_race(dataset, dataset.latest_snapshots()[0])
    with pytest.raises(ValueError, match="after prediction"):
        model.predict(earlier)
    card = target.model_dump(mode="json")
    card["race_id"] = model.manifest["fit_race_ids"][0]
    with pytest.raises(ValueError, match="used to fit"):
        model.predict(build_race(dataset, Snapshot.model_validate(card)))


def test_temperatures_preserve_order_and_exact_tie_rule(dataset):
    race = build_race(dataset, dataset.latest_snapshots()[-1])
    p = np.arange(1, len(race.horse_ids) + 1, dtype=float)
    p /= p.sum()
    for temperature in (0.25, 1, 4):
        result = calibrate(p, temperature)
        assert np.argmax(result) == np.argmax(p)
        assert result.sum() == pytest.approx(1)
    uniform = np.ones(len(p)) / len(p)
    assert select(race, uniform) == int(np.argmax(race.market))


def test_walk_forward_audit_keeps_groups_and_scores_every_race(dataset):
    snapshots = dataset.latest_snapshots()
    report = backtest(dataset, snapshots[27].cutoff, snapshots[-1].cutoff + timedelta(days=1))
    assert report["race_count"] == 3
    assert report["synthetic"] is True
    assert set(report["summary"]) == {"model", "market", "uniform"}
    assert report["paired_interval"]["race_days"] == 3
    assert all(r["prediction"]["cutoff"] for r in report["races"])


@pytest.mark.skipif(importlib.util.find_spec("catboost") is None, reason="optional tree dependency")
def test_catboost_query_groups_and_serialization(dataset):
    cutoff = dataset.latest_snapshots()[-1].cutoff
    races = [build_race(dataset, s, cutoff) for s in dataset.latest_snapshots()[:12]]
    tree = TreeModel().fit(races, iterations=20)
    target = build_race(dataset, dataset.latest_snapshots()[-1])
    p = tree.predict(target)
    np.testing.assert_allclose(TreeModel.from_dict(tree.to_dict()).predict(target), p)
    assert p.sum() == pytest.approx(1)
    ensemble = train(dataset, cutoff, trees=True, tree_iterations=20)
    assert ensemble.predict(target)["selection"]["horse_id"] in target.horse_ids
