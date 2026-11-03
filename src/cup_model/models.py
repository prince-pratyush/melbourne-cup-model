import base64
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax

from cup_model.features import FEATURES, Race


def require_labels(races: list[Race]):
    if not races or any(r.y is None for r in races):
        raise ValueError("complete labeled races required")


def loss(races: list[Race], probabilities: list[np.ndarray]) -> float:
    require_labels(races)
    if len(races) != len(probabilities):
        raise ValueError("prediction count mismatch")
    for race, p in zip(races, probabilities, strict=True):
        if p.shape != race.y.shape or np.any(p < 0) or not np.isfinite(p).all():
            raise ValueError("invalid probability vector")
        if not np.isclose(p.sum(), 1):
            raise ValueError("probabilities must sum to one per race")
    return float(
        np.mean(
            [
                -np.dot(r.y, np.log(np.clip(p, 1e-15, 1)))
                for r, p in zip(races, probabilities, strict=True)
            ]
        )
    )


class ConditionalLogit:
    def __init__(self, l2: float = 1, market_offset: bool = False):
        if l2 <= 0 or not np.isfinite(l2):
            raise ValueError("l2 must be finite and positive")
        self.l2 = l2
        self.market_offset = market_offset

    def transform(self, x):
        return (np.where(np.isfinite(x), x, self.median) - self.mean) / self.scale

    def fit(self, races: list[Race]):
        require_labels(races)
        x = np.vstack([r.x for r in races])
        self.median = np.array(
            [
                np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0
                for column in x.T
            ]
        )
        filled = np.where(np.isfinite(x), x, self.median)
        self.mean = filled.mean(axis=0)
        self.scale = filled.std(axis=0)
        self.scale[self.scale < 1e-8] = 1
        groups = [(self.transform(r.x), self.offset(r), r.y) for r in races]

        def objective(beta):
            total = 0.0
            gradient = np.zeros_like(beta)
            for matrix, offset, y in groups:
                scores = matrix @ beta + offset
                total += logsumexp(scores) - np.dot(y, scores)
                gradient += matrix.T @ (softmax(scores) - y)
            return (
                total / len(groups) + self.l2 * np.dot(beta, beta) / 2,
                gradient / len(groups) + self.l2 * beta,
            )

        fitted = minimize(
            objective,
            np.zeros(x.shape[1]),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if not fitted.success or not np.isfinite(fitted.x).all():
            raise ValueError(f"logit optimization failed: {fitted.message}")
        self.beta = fitted.x
        return self

    def offset(self, race: Race):
        if self.market_offset:
            if race.market is None:
                raise ValueError("market-relative model requires complete same-cutoff prices")
            return np.log(race.market)
        return np.zeros(len(race.horse_ids))

    def predict(self, race: Race) -> np.ndarray:
        return softmax(self.transform(race.x) @ self.beta + self.offset(race))

    def to_dict(self):
        return {
            "l2": self.l2,
            "market_offset": self.market_offset,
            **{name: getattr(self, name).tolist() for name in ("median", "mean", "scale", "beta")},
        }

    @classmethod
    def from_dict(cls, data):
        model = cls(data["l2"], data["market_offset"])
        for name in ("median", "mean", "scale", "beta"):
            value = np.asarray(data[name], dtype=float)
            if value.shape != (len(FEATURES),) or not np.isfinite(value).all():
                raise ValueError("invalid logit artifact")
            setattr(model, name, value)
        if np.any(model.scale <= 0):
            raise ValueError("invalid preprocessing scale")
        return model


class TreeModel:
    def fit(self, races: list[Race], iterations: int = 500, seed: int = 42):
        from catboost import CatBoostRanker, Pool

        require_labels(races)
        x = np.vstack([r.x for r in races])
        y = np.concatenate([r.y for r in races])
        groups = np.concatenate([np.repeat(i, len(r.horse_ids)) for i, r in enumerate(races)])
        self.model = CatBoostRanker(
            loss_function="QuerySoftMax",
            iterations=iterations,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=10,
            random_seed=seed,
            thread_count=1,
            allow_writing_files=False,
            verbose=False,
        )
        self.model.fit(Pool(x, label=y, group_id=groups))
        return self

    def predict(self, race: Race) -> np.ndarray:
        return softmax(np.asarray(self.model.predict(race.x)))

    def to_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.cbm"
            self.model.save_model(str(path))
            return {"cbm": base64.b64encode(path.read_bytes()).decode()}

    @classmethod
    def from_dict(cls, data):
        from catboost import CatBoostRanker

        instance = cls()
        instance.model = CatBoostRanker()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.cbm"
            path.write_bytes(base64.b64decode(data["cbm"], validate=True))
            instance.model.load_model(str(path))
        return instance


def fit_weights(races: list[Race], components: list[list[np.ndarray]], penalty: float = 0.01):
    require_labels(races)
    if not components:
        raise ValueError("ensemble requires components")
    baseline = np.zeros(len(components))
    baseline[0] = 1

    def objective(weights):
        mixed = [
            sum(w * p for w, p in zip(weights, row, strict=True))
            for row in zip(*components, strict=True)
        ]
        return loss(races, mixed) + penalty * np.sum((weights - baseline) ** 2)

    fitted = minimize(
        objective,
        baseline,
        method="SLSQP",
        bounds=[(0, 1)] * len(components),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not fitted.success:
        raise ValueError(f"ensemble optimization failed: {fitted.message}")
    weights = np.maximum(fitted.x, 0)
    return weights / weights.sum()


def calibrate(probability: np.ndarray, temperature: float) -> np.ndarray:
    if not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive")
    return softmax(np.log(np.clip(probability, 1e-15, 1)) / temperature)


def fit_temperature(races: list[Race], probabilities: list[np.ndarray]) -> float:
    fitted = minimize_scalar(
        lambda log_t: loss(races, [calibrate(p, np.exp(log_t)) for p in probabilities]),
        bounds=(np.log(0.25), np.log(4)),
        method="bounded",
    )
    if not fitted.success:
        raise ValueError("temperature optimization failed")
    return float(np.exp(fitted.x))


def select(race: Race, probability: np.ndarray) -> int:
    if probability.shape != (len(race.horse_ids),) or not np.isfinite(probability).all():
        raise ValueError("invalid prediction")
    if np.any(probability < 0) or not np.isclose(probability.sum(), 1):
        raise ValueError("prediction must be a probability distribution")
    return min(
        range(len(probability)),
        key=lambda i: (
            -probability[i],
            -race.market[i] if race.market is not None else 0,
            race.horse_ids[i],
        ),
    )
