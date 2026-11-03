import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from cup_model.contracts import Result, Snapshot, canonical, digest, parse_record


class Store:
    def __init__(self, path: str | Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS records (
                key TEXT PRIMARY KEY, kind TEXT NOT NULL, race_id TEXT NOT NULL,
                payload TEXT NOT NULL, sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY, payload TEXT NOT NULL
            );
        """)

    def close(self):
        self.connection.close()

    def import_jsonl(self, path: str | Path) -> int:
        records = []
        with Path(path).open() as stream:
            for line, text in enumerate(stream, 1):
                if text.strip():
                    try:
                        records.append(parse_record(json.loads(text)))
                    except ValueError as error:
                        raise ValueError(f"line {line}: {error}") from error
        with self.connection:
            for record in records:
                self._insert(record)
        return len(records)

    def _insert(self, record: Snapshot | Result):
        data = record.model_dump(mode="json")
        version = record.cutoff if isinstance(record, Snapshot) else record.available_at
        key = f"{record.kind}:{record.race_id}:{version.isoformat()}"
        sha = digest(data)
        prior = self.connection.execute("SELECT sha256 FROM records WHERE key=?", (key,)).fetchone()
        if prior and prior[0] != sha:
            raise ValueError(f"immutable record conflict: {key}")
        self.connection.execute(
            "INSERT OR IGNORE INTO records VALUES (?, ?, ?, ?, ?)",
            (key, record.kind, record.race_id, canonical(data), sha),
        )

    def records(self) -> list[Snapshot | Result]:
        return [
            parse_record(json.loads(row[0]))
            for row in self.connection.execute("SELECT payload FROM records ORDER BY key")
        ]

    def save_prediction(self, prediction: dict):
        key = prediction["prediction_id"]
        with self.connection:
            prior = self.connection.execute(
                "SELECT payload FROM predictions WHERE prediction_id=?", (key,)
            ).fetchone()
            if prior:
                existing = json.loads(prior[0])
                if {k: v for k, v in existing.items() if k != "recorded_at"} != prediction:
                    raise ValueError("prediction id already has different contents")
                return existing
            recorded = {**prediction, "recorded_at": datetime.now(UTC).isoformat()}
            self.connection.execute(
                "INSERT INTO predictions VALUES (?, ?)", (key, canonical(recorded))
            )
        return recorded


class Dataset:
    def __init__(self, records: list[Snapshot | Result]):
        self.snapshots = [r for r in records if isinstance(r, Snapshot)]
        self.results = [r for r in records if isinstance(r, Result)]

    def results_as_of(self, cutoff: datetime) -> dict[str, Result]:
        latest = {}
        for result in sorted(self.results, key=lambda r: r.available_at):
            if result.available_at <= cutoff and result.race_start < cutoff:
                latest[result.race_id] = result
        return latest

    def latest_snapshots(self, before: datetime | None = None) -> list[Snapshot]:
        latest = {}
        for snapshot in sorted(self.snapshots, key=lambda r: r.cutoff):
            if before is None or snapshot.cutoff < before:
                latest[snapshot.race_id] = snapshot
        return sorted(latest.values(), key=lambda r: (r.cutoff, r.race_id))

    def snapshot(self, race_id: str, cutoff: datetime | None = None) -> Snapshot:
        candidates = [s for s in self.snapshots if s.race_id == race_id]
        if cutoff is not None:
            candidates = [s for s in candidates if s.cutoff == cutoff]
        if not candidates:
            raise ValueError(f"no snapshot for {race_id} at requested cutoff")
        return max(candidates, key=lambda s: s.cutoff)
