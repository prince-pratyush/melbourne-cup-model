import json
from datetime import UTC, datetime

import pytest

from cup_model.betfair import BETTING_URL, LOGIN_URL, Client, record


class Transport:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, body, headers):
        self.calls.append((url, body, headers))
        return self.replies.pop(0)


def test_login_returns_session_and_hides_password_on_failure():
    transport = Transport({"status": "SUCCESS", "token": "session"})
    client = Client.login("key", "user", "secret", transport)
    assert client.token == "session"
    assert transport.calls[0][0] == LOGIN_URL
    assert transport.calls[0][2]["X-Application"] == "key"
    failing = Transport({"status": "FAIL", "error": "INVALID_USERNAME_OR_PASSWORD"})
    with pytest.raises(ValueError, match="INVALID_USERNAME_OR_PASSWORD") as error:
        Client.login("key", "user", "secret", failing)
    assert "secret" not in str(error.value)


def test_from_env_reports_missing_variables(monkeypatch):
    for name in ("BETFAIR_APP_KEY", "BETFAIR_USERNAME", "BETFAIR_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="BETFAIR_APP_KEY, BETFAIR_USERNAME"):
        Client.from_env(Transport())


def test_call_wraps_json_rpc_and_raises_api_errors():
    transport = Transport({"result": [1]}, {"error": {"code": -32099}})
    client = Client("key", "session", transport)
    assert client.call("listMarketCatalogue", {"maxResults": 1}) == [1]
    url, body, headers = transport.calls[0]
    assert url == BETTING_URL and headers["X-Authentication"] == "session"
    assert json.loads(body)["method"] == "SportsAPING/v1.0/listMarketCatalogue"
    with pytest.raises(ValueError, match="listMarketBook"):
        client.call("listMarketBook", {})


def test_record_appends_catalogue_and_books_with_capture_time(tmp_path):
    markets = [{"marketId": f"1.{i}"} for i in range(41)]
    transport = Transport(
        {"result": markets},
        {"result": [{"marketId": f"1.{i}", "runners": []} for i in range(40)]},
        {"result": [{"marketId": "1.40", "runners": []}]},
    )
    times = iter([datetime(2026, 11, 4, 1, tzinfo=UTC), datetime(2026, 11, 4, 1, 0, 5, tzinfo=UTC)])
    summary = record(Client("key", "session", transport), tmp_path, 30, lambda: next(times))
    assert summary["markets"] == 41
    request = json.loads(transport.calls[0][1])["params"]
    assert request["filter"]["marketStartTime"] == {
        "from": "2026-11-04T01:00:00Z",
        "to": "2026-11-04T01:30:00Z",
    }
    assert "RUNNER_METADATA" in request["marketProjection"]
    assert [len(json.loads(c[1])["params"]["marketIds"]) for c in transport.calls[1:]] == [40, 1]
    lines = [json.loads(x) for x in (tmp_path / "2026-11-04.jsonl").read_text().splitlines()]
    assert len(lines) == 41 and lines[40]["book"]["marketId"] == "1.40"
    assert lines[0]["recorded_at"] == "2026-11-04T01:00:05+00:00"


def test_record_refuses_truncated_catalogue(tmp_path):
    transport = Transport({"result": [{"marketId": str(i)} for i in range(100)]})
    with pytest.raises(ValueError, match="truncated"):
        record(Client("key", "session", transport), tmp_path)
