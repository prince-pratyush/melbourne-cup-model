import csv
import hashlib
import io
import itertools
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

from cup_model.contracts import Finish, Result, Runner, Snapshot
from cup_model.features import market_probabilities

BETFAIR_INDEX = "https://betfair-datascientists.github.io/data/dataListing/"
OFFICIAL_HISTORY = "https://www.racingaustralia.horse/arb/melbourne-cup.aspx"
CUP_INDEX = "https://www.furlongfiles.com/melbourne-cup/results/"
ZONES = {
    "VIC": "Australia/Melbourne",
    "NSW": "Australia/Sydney",
    "ACT": "Australia/Sydney",
    "QLD": "Australia/Brisbane",
    "SA": "Australia/Adelaide",
    "WA": "Australia/Perth",
    "TAS": "Australia/Hobart",
    "NT": "Australia/Darwin",
}


def file_hash(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url: str, directory: Path, name: str):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    metadata = target.with_suffix(target.suffix + ".source.json")
    if target.exists() and metadata.exists():
        prior = json.loads(metadata.read_text())
        if prior["url"] == url and prior["sha256"] == file_hash(target):
            return prior
        raise ValueError(f"cached source checksum conflict: {target}")
    request = urllib.request.Request(url, headers={"User-Agent": "MelbourneCupResearch/0.2"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                temporary = target.with_suffix(target.suffix + ".partial")
                size = 0
                with temporary.open("wb") as stream:
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > 200_000_000:
                            raise ValueError("source exceeds 200 MB download limit")
                        stream.write(block)
                temporary.replace(target)
                info = {
                    "url": url,
                    "file": str(target),
                    "bytes": size,
                    "sha256": file_hash(target),
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "last_modified": response.headers.get("Last-Modified"),
                }
                metadata.write_text(json.dumps(info, sort_keys=True, indent=2) + "\n")
                return info
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("download failed")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and "href" in values:
            self.links.append(values["href"])


class Table(HTMLParser):
    def __init__(self, table_class):
        super().__init__()
        self.table_class = table_class
        self.active = False
        self.cell = None
        self.row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == "table" and self.table_class in dict(attrs).get("class", "").split():
            self.active = True
        if not self.active:
            return
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if not self.active:
            return
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row:
            self.rows.append(self.row)
        elif tag == "table":
            self.active = False


def normalize_name(value):
    value = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", value)
    return re.sub(r"[^a-z0-9]", "", value.lower())


def parse_cup(html: str, year: int):
    table = Table("ff-table--results")
    table.feed(html)
    count = re.search(rf"Melbourne Cup {year}\s*[·&].*?(\d+) runners", html, re.S)
    if not count:
        raise ValueError(f"missing declared runner count for {year}")
    runners = []
    scratchings = []
    headers = [value.removesuffix("?") for value in table.rows[0]] if table.rows else []
    required = ["Fin", "Horse", "Barrier", "Weight", "Age", "Sex", "SP", "Trainer", "Jockey"]
    if not set(required).issubset(headers):
        raise ValueError(f"Cup table headers missing for {year}: {headers}")
    for row in table.rows[1:]:
        if len(row) != len(headers):
            raise ValueError(f"unexpected Cup table columns for {year}: {row}")
        values = dict(zip(headers, row, strict=True))
        finish, name, barrier, weight, age, sex, sp, trainer, jockey = [values[k] for k in required]
        if finish.lower() in ("scratched", "scr", "withdrawn"):
            scratchings.append(name)
            continue
        runners.append(
            {
                "finish": finish,
                "name": name,
                "barrier_effective": int(barrier) if barrier.isdigit() else None,
                "weight_kg": float(weight.removesuffix("kg"))
                if re.fullmatch(r"\d+(\.\d+)?kg", weight)
                else None,
                "age_reported": int(age) if age.isdigit() else None,
                "sex": sex,
                "starting_price": float(sp.removeprefix("$"))
                if re.fullmatch(r"\$\d+(\.\d+)?", sp)
                else None,
                "trainer": trainer,
                "jockey": jockey,
            }
        )
    if len(runners) != int(count.group(1)) or not any(r["finish"] == "1st" for r in runners):
        raise ValueError(f"Cup field count/winner mismatch for {year}")
    return {"year": year, "starters": len(runners), "runners": runners, "scratchings": scratchings}


def enrich_cup(snapshot: Snapshot, result: Result, cup: dict):
    rows = {normalize_name(r["name"]): r for r in cup["runners"]}
    names = {r.horse_id: normalize_name(r.name) for r in snapshot.runners}
    if set(names.values()) != set(rows):
        return snapshot, result, "field_mismatch"
    places = {}
    for horse_id, name in names.items():
        place = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", rows[name]["finish"])
        places[horse_id] = int(place.group(1)) if place else None
    winners = {f.horse_id for f in result.finishes if f.position == 1}
    if winners != {h for h, p in places.items() if p == 1}:
        return snapshot, result, "winner_mismatch"
    data = snapshot.model_dump(mode="json")
    data["source"] += " " + cup["source"]["url"]
    for runner in data["runners"]:
        row = rows[normalize_name(runner["name"])]
        runner["weight_kg"] = row["weight_kg"]
        runner["barrier_effective"] = row["barrier_effective"]
        runner["age_years"] = row["age_reported"]
    finishes = result.model_dump(mode="json")
    finishes["source"] += " " + cup["source"]["url"]
    for finish in finishes["finishes"]:
        position = places[finish["horse_id"]]
        finish["position"] = position
        finish["status"] = "finished" if position else "non_finisher"
    return Snapshot.model_validate(data), Result.model_validate(finishes), None


def parse_official_history(html: str):
    table = Table("melbournecup-history-table")
    table.feed(html)
    rows = {}
    for row in table.rows:
        if len(row) == 9 and row[0].isdigit():
            rows[int(row[0])] = {
                "name": row[1],
                "barrier_reported": row[5],
                "weight_kg": row[6],
                "starting_price": row[7],
                "time": row[8],
            }
    if not rows:
        raise ValueError("official winner history table not found")
    return rows


def acquire(destination: str | Path, through_year=2026, cup_start=2006, cup_end=2025):
    root = Path(destination)
    raw = root / "raw"
    index = download(BETFAIR_INDEX, raw, "betfair-listing.html")
    parser = Links()
    parser.feed(Path(index["file"]).read_text())
    urls = []
    for href in parser.links:
        match = re.search(r"ANZ_Thoroughbreds_(\d{4})(?:_\d{2})?\.(zip|csv)$", href)
        if match and int(match.group(1)) <= through_year:
            urls.append(urllib.parse.urljoin(BETFAIR_INDEX, href))
    if not urls:
        raise ValueError("archive links not found in Betfair's published index")

    def get_archive(url):
        return download(url, raw, Path(urllib.parse.urlparse(url).path).name)

    with ThreadPoolExecutor(max_workers=3) as pool:
        archives = list(pool.map(get_archive, sorted(set(urls))))
    official = download(OFFICIAL_HISTORY, raw, "racingaustralia-history.html")
    reference = parse_official_history(Path(official["file"]).read_text())
    cups = []
    discrepancies = []
    for year in range(cup_start, cup_end + 1):
        url = f"{CUP_INDEX}?cup={year}"
        source = download(url, raw, f"melbourne-cup-{year}.html")
        cup = parse_cup(Path(source["file"]).read_text(), year)
        cup["source"] = source
        winner = next(r for r in cup["runners"] if r["finish"] == "1st")
        primary = reference.get(year)
        if not primary or normalize_name(primary["name"]) != normalize_name(winner["name"]):
            raise ValueError(f"official winner cross-check failed for {year}")
        cup["official_winner_crosscheck"] = primary
        for field, a, b in (
            ("winner_weight", primary["weight_kg"], winner["weight_kg"]),
            ("barrier_definition", primary["barrier_reported"], winner["barrier_effective"]),
        ):
            if str(a) != str(b) and not (field == "winner_weight" and float(a) == b):
                discrepancies.append({"year": year, "field": field, "official": a, "secondary": b})
        cups.append(cup)
    catalog = {
        "betfair_index": index,
        "archives": archives,
        "official_history": official,
        "cups": cups,
        "discrepancies": discrepancies,
        "excluded_sources": [
            {
                "url": "https://www.kaggle.com/datasets/hassankamran011/historic-australian-horse-racing-dataset",
                "reason": "field.csv describes harness trotters, drivers and mobile starts",
            }
        ],
    }
    (root / "source-catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    return catalog


def archive_rows(path: Path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith(".csv"):
                    with archive.open(name) as source:
                        yield from csv.DictReader(io.TextIOWrapper(source, encoding="utf-8-sig"))
    else:
        with path.open(encoding="utf-8-sig") as source:
            yield from csv.DictReader(source)


def parse_clock(day: str, clock: str, state: str):
    if state not in ZONES:
        raise ValueError("unsupported archive jurisdiction")
    if "/" in day:
        day = datetime.strptime(day, "%d/%m/%Y").date().isoformat()
    if clock.endswith(("AM", "PM")):
        pattern = "%I:%M:%S %p" if clock.count(":") == 2 else "%I:%M %p"
        clock = datetime.strptime(clock, pattern).time().isoformat()
    return (
        datetime.fromisoformat(f"{day}T{clock}")
        .replace(tzinfo=ZoneInfo("Australia/Sydney"))
        .astimezone(UTC)
    )


def convert_market(rows: list[dict], source: dict, minimum_distance=2000, maximum_distance=3600):
    first = rows[0]
    state = first["STATE_CODE"]
    if state not in ZONES:
        return None, "non_australian"
    if first.get("RACING_TYPE", "").lower() != "thoroughbred":
        return None, "not_thoroughbred"
    distance_value = float(first["DISTANCE"])
    if not distance_value.is_integer():
        return None, "invalid_distance"
    distance = int(distance_value)
    if not minimum_distance <= distance <= maximum_distance:
        return None, "outside_distance_range"
    race_type = first["WIN_MARKET_NAME"] + " " + first.get("RACE_TYPE", "")
    if re.search(r"hurd|hrdl|\bhrd\b|stpl|steeple|chase", race_type, re.I):
        return None, "jumps_race"
    keys = (
        "STATE_CODE",
        "LOCAL_MEETING_DATE",
        "SCHEDULED_RACE_TIME",
        "ACTUAL_OFF_TIME",
        "WIN_MARKET_ID",
        "TRACK",
        "DISTANCE",
    )
    if any(any(r[k] != first[k] for k in keys) for r in rows):
        return None, "inconsistent_race_metadata"
    if any(r["WIN_RESULT"] not in ("WINNER", "LOSER") for r in rows):
        return None, "unresolved_runner_status"
    if len(rows) < 2 or len({r["SELECTION_ID"] for r in rows}) != len(rows):
        return None, "duplicate_or_small_field"
    if not any(r["WIN_RESULT"] == "WINNER" for r in rows):
        return None, "no_winner"
    quote_column = "BEST_AVAIL_BACK_AT_SCHEDULED_OFF"
    try:
        prices = [float(r[quote_column]) for r in rows]
        q = market_probabilities(prices)
    except (KeyError, ValueError):
        return None, "missing_scheduled_off_quotes"
    book = sum(1 / price for price in prices)
    if not 0.85 <= book <= 1.20:
        return None, "implausible_or_incomplete_book"
    start = parse_clock(first["LOCAL_MEETING_DATE"], first["SCHEDULED_RACE_TIME"], state)
    actual = parse_clock(first["LOCAL_MEETING_DATE"], first["ACTUAL_OFF_TIME"], state)
    if actual <= start or actual - start > timedelta(hours=2):
        return None, "quote_not_strictly_before_actual_start"
    race_id = f"betfair:{first['WIN_MARKET_ID']}"
    source_url = source["url"] + "#market=" + first["WIN_MARKET_ID"]
    runners = tuple(
        Runner(
            horse_id=f"betfair:{r['SELECTION_ID']}", name=r["SELECTION_NAME"], decimal_odds=price
        )
        for r, price in zip(rows, prices, strict=True)
    )
    snapshot = Snapshot(
        race_id=race_id,
        scheduled_start=start,
        cutoff=start,
        available_at=start,
        source=source_url,
        distance_m=distance,
        venue=first["TRACK"],
        market_operator="Betfair scheduled-off back",
        market_at=start,
        expected_starters=len(rows),
        runners=runners,
        timing_provenance="scheduled_off_reconstruction",
        retrieved_at=source["retrieved_at"],
    )
    result = Result(
        race_id=race_id,
        race_start=actual,
        available_at=actual + timedelta(days=2),
        source=source_url,
        distance_m=distance,
        availability_basis="two_day_embargo_assumption",
        finishes=tuple(
            Finish(
                horse_id=f"betfair:{r['SELECTION_ID']}",
                position=1 if r["WIN_RESULT"] == "WINNER" else None,
                status="finished" if r["WIN_RESULT"] == "WINNER" else "unplaced_unknown",
                prior_market_probability=float(p),
            )
            for r, p in zip(rows, q, strict=True)
        ),
    )
    return (snapshot, result), None


def prepare(destination: str | Path, minimum_distance=2000, maximum_distance=3600):
    root = Path(destination)
    catalog = json.loads((root / "source-catalog.json").read_text())
    counts = Counter()
    rejection_examples = {}
    years = {}
    seen = set()
    cup_rows = []
    events = root / "racing-history.jsonl"
    with events.open("w") as output:
        for source in catalog["archives"]:
            path = Path(source["file"])
            if file_hash(path) != source["sha256"]:
                raise ValueError(f"source checksum mismatch: {path}")
            for market, group in itertools.groupby(
                archive_rows(path), key=lambda r: r["WIN_MARKET_ID"]
            ):
                rows = list(group)
                counts["downloaded_runner_rows"] += len(rows)
                if market in seen:
                    raise ValueError(f"noncontiguous or duplicate source market: {market}")
                seen.add(market)
                try:
                    records, reason = convert_market(
                        rows, source, min(800, minimum_distance), maximum_distance
                    )
                except (KeyError, ValueError) as error:
                    reason = f"schema_or_value_error:{type(error).__name__}"
                    records = None
                if records is None:
                    counts["excluded:" + reason] += 1
                    if reason not in rejection_examples:
                        rejection_examples[reason] = {"market_id": market, "source": source["url"]}
                    continue
                snapshot, result = records
                year = str(snapshot.cutoff.astimezone(ZoneInfo("Australia/Sydney")).year)
                info = years.setdefault(year, {"races": 0, "runners": 0, "history_races": 0})
                counts["history_races"] += 1
                counts["history_runner_rows"] += len(rows)
                info["history_races"] += 1
                output.write(result.model_dump_json() + "\n")
                if snapshot.distance_m < minimum_distance:
                    counts["shorter_history_only_races"] += 1
                    continue
                counts["accepted_races"] += 1
                counts["accepted_runner_rows"] += len(rows)
                info["races"] += 1
                info["runners"] += len(rows)
                output.write(snapshot.model_dump_json() + "\n")
                if snapshot.venue == "Flemington" and snapshot.distance_m == 3200:
                    secondary = next((c for c in catalog["cups"] if c["year"] == int(year)), None)
                    if secondary:
                        names = {normalize_name(r["SELECTION_NAME"]) for r in rows}
                        other = {normalize_name(r["name"]) for r in secondary["runners"]}
                        winner = [r["SELECTION_NAME"] for r in rows if r["WIN_RESULT"] == "WINNER"]
                        cup_rows.append(
                            {
                                "year": int(year),
                                "race_id": snapshot.race_id,
                                "runners": len(rows),
                                "winner": winner,
                                "full_field_matches_secondary": names == other,
                                "names_only_in_market": sorted(names - other),
                                "names_only_in_secondary": sorted(other - names),
                            }
                        )
    report = {
        "counts": dict(counts),
        "rejection_examples": rejection_examples,
        "years": years,
        "events_file": str(events),
        "events_sha256": file_hash(events),
        "source_archive_count": len(catalog["archives"]),
        "cup_archive_years": [c["year"] for c in catalog["cups"]],
        "cup_archive_starters": sum(c["starters"] for c in catalog["cups"]),
        "cup_market_crosschecks": cup_rows,
        "missing_inputs": [
            "weights",
            "draws",
            "ages",
            "going",
            "sectionals",
            "full_finish_order",
            "five_minutes_before_start_odds",
            "verified_result_publication_timestamps",
        ],
        "timing": {
            "prediction_edition": "scheduled_off_reconstruction",
            "price_column": "BEST_AVAIL_BACK_AT_SCHEDULED_OFF",
            "excluded_columns": ["WIN_BSP", "WIN_PREPLAY_LAST_PRICE_TAKEN", "WIN_INPLAY_*"],
            "result_availability": "actual off plus two days; explicit conservative assumption",
            "timezone_basis": "Australia/Sydney inferred from cross-checks; not venue local time",
            "timezone_crosscheck": {
                "race": "Ascot 2025-01-01 R1",
                "archive_clock": "18:37",
                "official_WA_clock": "15:37",
                "source": "https://mdata.racingnsw.com.au/InteractiveForm/Meeting.aspx?racecode=MTY0OTA2MzA3MDA%3D",
            },
        },
    }
    (root / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
