import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Number = Annotated[float, Field(allow_inf_nan=False)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Runner(Record):
    horse_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    status: Literal["active", "scratched", "unresolved"] = "active"
    weight_kg: Number = Field(ge=40, le=80)
    barrier_drawn: int = Field(ge=1, le=50)
    barrier_effective: int | None = Field(default=None, ge=1, le=50)
    age_years: Number = Field(ge=2, le=20)
    decimal_odds: Number | None = Field(default=None, gt=1)


class Snapshot(Record):
    kind: Literal["snapshot"] = "snapshot"
    race_id: str = Field(min_length=1)
    scheduled_start: AwareDatetime
    cutoff: AwareDatetime
    available_at: AwareDatetime
    source: str = Field(min_length=1)
    distance_m: int = Field(ge=800, le=5000)
    going: Number = Field(ge=1, le=10)
    venue: str = Field(min_length=1)
    market_operator: str | None = None
    market_at: AwareDatetime | None = None
    expected_starters: int = Field(ge=2, le=50)
    runners: tuple[Runner, ...]
    synthetic: bool = False

    @model_validator(mode="after")
    def validate_field(self):
        if not self.available_at <= self.cutoff < self.scheduled_start:
            raise ValueError("require available_at <= cutoff < scheduled_start")
        ids = [r.horse_id for r in self.runners]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate horse identity")
        if any(r.status == "unresolved" for r in self.runners):
            raise ValueError("unresolved runner status")
        active = [r for r in self.runners if r.status == "active"]
        if len(active) != self.expected_starters:
            raise ValueError("active field does not match expected_starters")
        drawn = [r.barrier_drawn for r in self.runners]
        if len(set(drawn)) != len(drawn):
            raise ValueError("duplicate drawn barrier")
        effective = [r.barrier_effective for r in active if r.barrier_effective is not None]
        if len(set(effective)) != len(effective):
            raise ValueError("duplicate effective barrier")
        priced = [r.decimal_odds is not None for r in active]
        if any(priced):
            if not all(priced) or not self.market_operator or self.market_at is None:
                raise ValueError("market needs complete prices, operator and timestamp")
            age = (self.cutoff - self.market_at).total_seconds()
            if not 0 <= age <= 900:
                raise ValueError("market must be no more than 15 minutes old at cutoff")
        return self


class Finish(Record):
    horse_id: str = Field(min_length=1)
    position: int | None = Field(default=None, ge=1)
    status: Literal["finished", "non_finisher", "disqualified"] = "finished"
    rating: Number | None = None
    final_600_s: Number | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_finish(self):
        if (self.status == "finished") != (self.position is not None):
            raise ValueError("only finishers must have a position")
        return self


class Result(Record):
    kind: Literal["result"] = "result"
    race_id: str = Field(min_length=1)
    race_start: AwareDatetime
    available_at: AwareDatetime
    source: str = Field(min_length=1)
    distance_m: int = Field(ge=800, le=5000)
    going: Number = Field(ge=1, le=10)
    finishes: tuple[Finish, ...]
    synthetic: bool = False

    @model_validator(mode="after")
    def validate_result(self):
        if self.available_at <= self.race_start:
            raise ValueError("result cannot be available before race completion")
        ids = [r.horse_id for r in self.finishes]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("results need a complete unique field")
        if not any(r.position == 1 for r in self.finishes):
            raise ValueError("official result needs a winner")
        if any(r.position is not None and r.position > len(ids) for r in self.finishes):
            raise ValueError("position exceeds field size")
        return self


def parse_record(data: dict) -> Snapshot | Result:
    if data.get("kind") == "snapshot":
        return Snapshot.model_validate(data)
    if data.get("kind") == "result":
        return Result.model_validate(data)
    raise ValueError("kind must be snapshot or result")


def canonical(data) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data) -> str:
    return hashlib.sha256(canonical(data).encode()).hexdigest()


def timestamp(value: str) -> datetime:
    from pydantic import TypeAdapter

    return TypeAdapter(AwareDatetime).validate_python(value)
