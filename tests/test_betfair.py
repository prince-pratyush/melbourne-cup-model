import json

import pytest

from cup_model.betfair import BETTING_URL, LOGIN_URL, Client


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
