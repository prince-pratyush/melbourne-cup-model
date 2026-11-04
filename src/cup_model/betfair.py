import json
import os
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

LOGIN_URL = "https://identitysso.betfair.com.au/api/login"
BETTING_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"
ENVIRONMENT = ("BETFAIR_APP_KEY", "BETFAIR_USERNAME", "BETFAIR_PASSWORD")
PROJECTION = [
    "EVENT",
    "MARKET_START_TIME",
    "MARKET_DESCRIPTION",
    "RUNNER_DESCRIPTION",
    "RUNNER_METADATA",
]
STAMP = "%Y-%m-%dT%H:%M:%SZ"


def post(url: str, body: bytes, headers: dict) -> dict:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


class Client:
    def __init__(self, app_key: str, token: str, transport=post):
        self.app_key = app_key
        self.token = token
        self.transport = transport

    @classmethod
    def login(cls, app_key: str, username: str, password: str, transport=post):
        body = urllib.parse.urlencode({"username": username, "password": password}).encode()
        reply = transport(
            LOGIN_URL,
            body,
            {
                "X-Application": app_key,
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        if reply.get("status") != "SUCCESS":
            raise ValueError(f"Betfair login failed: {reply.get('error', 'unknown error')}")
        return cls(app_key, reply["token"], transport)

    @classmethod
    def from_env(cls, transport=post):
        missing = [name for name in ENVIRONMENT if not os.environ.get(name)]
        if missing:
            raise ValueError("missing environment variables: " + ", ".join(missing))
        return cls.login(*(os.environ[name] for name in ENVIRONMENT), transport=transport)

    def call(self, method: str, params: dict):
        body = json.dumps(
            {"jsonrpc": "2.0", "method": f"SportsAPING/v1.0/{method}", "params": params, "id": 1}
        ).encode()
        reply = self.transport(
            BETTING_URL,
            body,
            {
                "X-Application": self.app_key,
                "X-Authentication": self.token,
                "Content-Type": "application/json",
            },
        )
        if "error" in reply:
            raise ValueError(f"Betfair {method} failed: {reply['error']}")
        return reply["result"]


def record(client: Client, destination, minutes_ahead=30, clock=lambda: datetime.now(UTC)):
    start = clock()
    window = {
        "from": start.strftime(STAMP),
        "to": (start + timedelta(minutes=minutes_ahead)).strftime(STAMP),
    }
    catalogue = client.call(
        "listMarketCatalogue",
        {
            "filter": {
                "eventTypeIds": ["7"],
                "marketCountries": ["AU"],
                "marketTypeCodes": ["WIN"],
                "marketStartTime": window,
            },
            "marketProjection": PROJECTION,
            "maxResults": 100,
        },
    )
    if len(catalogue) >= 100:
        raise ValueError("catalogue may be truncated; lower --minutes-ahead")
    ids = [market["marketId"] for market in catalogue]
    books = {}
    for i in range(0, len(ids), 40):
        for book in client.call(
            "listMarketBook",
            {"marketIds": ids[i : i + 40], "priceProjection": {"priceData": ["EX_BEST_OFFERS"]}},
        ):
            books[book["marketId"]] = book
    recorded_at = clock().isoformat()
    path = Path(destination) / f"{start.date().isoformat()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        for market in catalogue:
            line = {
                "recorded_at": recorded_at,
                "catalogue": market,
                "book": books.get(market["marketId"]),
            }
            stream.write(json.dumps(line, sort_keys=True) + "\n")
    return {"markets": len(catalogue), "file": str(path), "recorded_at": recorded_at}
