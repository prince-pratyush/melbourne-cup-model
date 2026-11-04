from copy import deepcopy
from datetime import timedelta

import numpy as np
import pytest

from cup_model.acquisition import (
    convert_market,
    enrich_cup,
    parse_clock,
    parse_cup,
    parse_official_history,
)
from cup_model.contracts import Snapshot
from cup_model.features import FEATURES, build_race
from cup_model.pipeline import train
from cup_model.store import Dataset


@pytest.fixture
def market_rows():
    common = {
        "STATE_CODE": "VIC",
        "LOCAL_MEETING_DATE": "2023-11-07",
        "SCHEDULED_RACE_TIME": "15:00:00.000",
        "ACTUAL_OFF_TIME": "15:04:00.000",
        "WIN_MARKET_ID": "test-market",
        "TRACK": "Synthetic Test Track",
        "DISTANCE": "3200",
        "RACING_TYPE": "Thoroughbred",
        "RACE_TYPE": "Hcap",
        "WIN_MARKET_NAME": "R7 3200m Hcap",
    }
    return [
        {
            **common,
            "SELECTION_ID": "a",
            "SELECTION_NAME": "Synthetic A",
            "WIN_RESULT": "WINNER",
            "BEST_AVAIL_BACK_AT_SCHEDULED_OFF": "2",
        },
        {
            **common,
            "SELECTION_ID": "b",
            "SELECTION_NAME": "Synthetic B",
            "WIN_RESULT": "LOSER",
            "BEST_AVAIL_BACK_AT_SCHEDULED_OFF": "2",
        },
    ]


SOURCE = {"url": "https://example.invalid/test.csv", "retrieved_at": "2026-11-04T00:00:00Z"}


def test_archive_clock_is_eastern_and_supports_april_2025_format():
    assert parse_clock("2025-01-01", "18:37:00", "WA").isoformat() == "2025-01-01T07:37:00+00:00"
    assert parse_clock("1/04/2025", "1:18:00 PM", "VIC").isoformat() == "2025-04-01T02:18:00+00:00"


def test_accepts_integral_decimal_distance_without_rounding(market_rows):
    for row in market_rows:
        row["DISTANCE"] = "3200.0"
    assert convert_market(market_rows, SOURCE)[0][0].distance_m == 3200
    for row in market_rows:
        row["DISTANCE"] = "3200.5"
    assert convert_market(market_rows, SOURCE)[1] == "invalid_distance"


def test_missing_fundamentals_and_loser_ranks_remain_unknown(market_rows):
    (snapshot, result), reason = convert_market(market_rows, SOURCE)
    assert reason is None
    assert all(r.weight_kg is None and r.age_years is None for r in snapshot.runners)
    assert result.finishes[1].status == "unplaced_unknown"
    assert result.finishes[1].position is None
    assert result.availability_basis == "two_day_embargo_assumption"
    assert snapshot.timing_provenance == "scheduled_off_reconstruction"
    assert snapshot.cutoff < result.race_start
    race = build_race(Dataset([snapshot, result]), snapshot, result.available_at)
    assert np.all(np.isnan(race.x[:, FEATURES.index("weight_kg")]))
    np.testing.assert_equal(race.y, [1, 0])
    later = snapshot.model_copy(
        update={
            "race_id": "future",
            "scheduled_start": snapshot.scheduled_start + timedelta(days=7),
            "cutoff": snapshot.cutoff + timedelta(days=7),
            "available_at": snapshot.available_at + timedelta(days=7),
        }
    )
    future_race = build_race(Dataset([snapshot, result]), later)
    assert np.isnan(future_race.x[1, FEATURES.index("form_120d")])
    assert future_race.x[1, FEATURES.index("win_rate_120d")] == 0


def test_reconstructed_training_needs_explicit_opt_in(dataset):
    snapshots = []
    for original in dataset.snapshots:
        data = original.model_dump(mode="json")
        data["scheduled_start"] = data["available_at"] = data["cutoff"]
        data["timing_provenance"] = "scheduled_off_reconstruction"
        data["retrieved_at"] = "2026-11-04T00:00:00Z"
        snapshots.append(Snapshot.model_validate(data))
    reconstructed = Dataset(snapshots + dataset.results)
    with pytest.raises(ValueError, match="allow-reconstructed"):
        train(reconstructed, snapshots[-1].cutoff)


def test_never_substitutes_bsp_or_inplay_for_scheduled_quotes(market_rows):
    changed = deepcopy(market_rows)
    for row in changed:
        row["WIN_BSP"] = "999"
        row["WIN_PREPLAY_LAST_PRICE_TAKEN"] = "1.01"
        row["WIN_INPLAY_MIN_PRICE_TAKEN"] = "1.01"
    expected, _ = convert_market(market_rows, SOURCE)
    observed, _ = convert_market(changed, SOURCE)
    assert observed == expected
    del changed[0]["BEST_AVAIL_BACK_AT_SCHEDULED_OFF"]
    records, reason = convert_market(changed, SOURCE)
    assert records is None and reason == "missing_scheduled_off_quotes"


def test_rejects_quotes_at_or_after_actual_start(market_rows):
    for row in market_rows:
        row["ACTUAL_OFF_TIME"] = row["SCHEDULED_RACE_TIME"]
    records, reason = convert_market(market_rows, SOURCE)
    assert records is None and reason == "quote_not_strictly_before_actual_start"


def test_rejects_harness_jumps_and_incomplete_books(market_rows):
    for row in market_rows:
        row["RACING_TYPE"] = "Harness"
    assert convert_market(market_rows, SOURCE)[1] == "not_thoroughbred"
    for row in market_rows:
        row["RACING_TYPE"] = "Thoroughbred"
        row["RACE_TYPE"] = "Hrdl"
    assert convert_market(market_rows, SOURCE)[1] == "jumps_race"
    for row in market_rows:
        row["RACE_TYPE"] = "Hcap"
        row["BEST_AVAIL_BACK_AT_SCHEDULED_OFF"] = "20"
    assert convert_market(market_rows, SOURCE)[1] == "implausible_or_incomplete_book"


def test_cup_parser_distinguishes_starting_price_bsp_and_scratchings():
    html = """
    <h2>Melbourne Cup 2006 · 2 runners</h2>
    <table class="ff-table ff-table--results"><tr>
    <th>Fin</th><th>Horse</th><th>Barrier?</th><th>Weight?</th><th>Age</th>
    <th>Sex</th><th>SP?</th><th>BSP?</th><th>Trainer</th><th>Jockey</th></tr>
    <tr><td>1st</td><td>Synthetic A</td><td>1</td><td>53kg</td><td>4</td>
    <td>Gelding</td><td>$5</td><td>$6</td><td>Trainer A</td><td>Rider A</td></tr>
    <tr><td>DNF</td><td>Synthetic B</td><td>2</td><td>54kg</td><td>5</td>
    <td>Gelding</td><td>$2</td><td>$3</td><td>Trainer B</td><td>Rider B</td></tr>
    <tr><td>Scratched</td><td>Synthetic C</td><td>–</td><td>–</td><td>–</td>
    <td></td><td>–</td><td>–</td><td></td><td></td></tr></table>
    """
    cup = parse_cup(html, 2006)
    assert cup["starters"] == 2 and cup["scratchings"] == ["Synthetic C"]
    assert cup["runners"][0]["starting_price"] == 5
    assert cup["runners"][1]["finish"] == "DNF"


def test_official_history_parser_requires_correct_table():
    html = """<table class="melbournecup-history-table"><tr><td>2006</td><td>Synthetic A</td>
    <td>Rider</td><td>Trainer</td><td>4G</td><td>1</td><td>53</td><td>$5</td>
    <td>3:20.00</td></tr></table>"""
    assert parse_official_history(html)[2006]["weight_kg"] == "53"
    with pytest.raises(ValueError):
        parse_official_history("<html></html>")


def cup_archive(first="Synthetic A", second="Synthetic B", second_finish="DNF"):
    return {
        "source": {"url": "https://example.invalid/cup"},
        "runners": [
            {
                "name": first,
                "finish": "1st",
                "weight_kg": 53.0,
                "barrier_effective": 4,
                "age_reported": 5,
            },
            {
                "name": second,
                "finish": second_finish,
                "weight_kg": 57.5,
                "barrier_effective": 1,
                "age_reported": 7,
            },
        ],
    }


def test_cup_archive_fills_fields_and_finish_order(market_rows):
    (snapshot, result), _ = convert_market(market_rows, SOURCE)
    snapshot, result, reason = enrich_cup(snapshot, result, cup_archive())
    assert reason is None
    assert [(r.weight_kg, r.barrier_effective, r.age_years) for r in snapshot.runners] == [
        (53, 4, 5),
        (57.5, 1, 7),
    ]
    assert [(f.position, f.status) for f in result.finishes] == [
        (1, "finished"),
        (None, "non_finisher"),
    ]
    race = build_race(Dataset([snapshot, result]), snapshot, result.available_at)
    np.testing.assert_equal(race.x[:, FEATURES.index("weight_kg")], [53, 57.5])


def test_cup_archive_requires_matching_field_and_winner(market_rows):
    (snapshot, result), _ = convert_market(market_rows, SOURCE)
    unchanged = enrich_cup(snapshot, result, cup_archive(second="Other"))
    assert unchanged == (snapshot, result, "field_mismatch")
    swapped = enrich_cup(snapshot, result, cup_archive("Synthetic B", "Synthetic A", "2nd"))
    assert swapped == (snapshot, result, "winner_mismatch")


def test_market_uses_back_lay_midpoint_with_back_fallback(market_rows):
    market_rows[0]["BEST_AVAIL_LAY_AT_SCHEDULED_OFF"] = "2.5"
    market_rows[1]["BEST_AVAIL_LAY_AT_SCHEDULED_OFF"] = "1.5"
    (snapshot, _), _ = convert_market(market_rows, SOURCE)
    assert snapshot.runners[0].decimal_odds == pytest.approx(2 / (1 / 2 + 1 / 2.5))
    assert snapshot.runners[1].decimal_odds == 2
    market_rows[0]["BEST_AVAIL_LAY_AT_SCHEDULED_OFF"] = "1000"
    (snapshot, _), _ = convert_market(market_rows, SOURCE)
    assert snapshot.runners[0].decimal_odds == 2
