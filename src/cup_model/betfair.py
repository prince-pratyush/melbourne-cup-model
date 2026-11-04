import json
import os
import urllib.parse
import urllib.request

LOGIN_URL = "https://identitysso.betfair.com.au/api/login"
BETTING_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"
ENVIRONMENT = ("BETFAIR_APP_KEY", "BETFAIR_USERNAME", "BETFAIR_PASSWORD")


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
