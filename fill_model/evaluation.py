"""Censor-aware factual evaluation, with no test-set fitting or calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import event_time_nll, probability_from_rate


class CensoringKM:
    """Marginal reverse Kaplan-Meier, fitted on training intervals only.

    IPCW is optional: it assumes censoring independent of first-fill time. Trader
    repricing/RFQs may violate that assumption. It does not identify causal effects.
    """
    def fit(self, df: pd.DataFrame) -> "CensoringKM":
        grouped = df.assign(censor=1 - df.event).groupby("exposure_minutes").agg(
            total=("event", "size"), censor=("censor", "sum"))
        self.times = grouped.index.to_numpy(float)
        risk = len(df) - np.r_[0, grouped.total.to_numpy().cumsum()[:-1]]
        self.survival = np.cumprod(1 - grouped.censor.to_numpy() / risk)
        return self

    def survival_left(self, times: np.ndarray | float) -> np.ndarray:
        values = np.atleast_1d(np.asarray(times, dtype=float))
        index = np.searchsorted(self.times, values, side="left") - 1
        result = np.ones(len(values))
        mask = index >= 0
        result[mask] = self.survival[index[mask]]
        return result


def ipcw_horizon_metrics(df: pd.DataFrame, rate: np.ndarray, km: CensoringKM,
                         horizon: float = 30.0, min_survival: float = 0.05) -> dict:
    """IPCW Brier/risk at a fixed horizon; early censored negatives get zero weight.

    Uses G(t-) for events at t and subjects observed through h, including a
    deterministic slice boundary at h. A drop in G at h must not invalidate
    subjects whose no-fill status through h is actually observed.
    """
    if horizon <= 0 or not 0 < min_survival <= 1:
        raise ValueError("Invalid IPCW horizon or minimum survival.")
    t, d = df.exposure_minutes.to_numpy(float), df.event.to_numpy(int)
    p = probability_from_rate(rate, horizon)
    event_by_h = (d == 1) & (t <= horizon)
    observed_through_h = (t >= horizon) & ~event_by_h
    gh = float(km.survival_left(horizon)[0])
    ge = km.survival_left(t[event_by_h])
    common = {"horizon_minutes": horizon, "early_censored_rows": int((~event_by_h & ~observed_through_h).sum()),
              "censor_survival_at_h_left": gh,
              "ipcw_assumption": "marginal independent censoring; training-only censor KM"}
    if gh < min_survival or (len(ge) and ge.min() < min_survival):
        return {**common, "ipcw_status": "insufficient_censoring_support", "ipcw_brier": np.nan,
                "ipcw_observed_risk": np.nan, "ipcw_predicted_to_observed": np.nan,
                "ipcw_effective_rows": np.nan}
    weights = np.zeros(len(df))
    weights[event_by_h] = 1 / ge
    weights[observed_through_h] = 1 / gh
    risk = float(np.sum(weights * event_by_h) / len(df))
    return {**common, "ipcw_status": "ok_assumption_required",
            "ipcw_brier": float(np.sum(weights * (event_by_h.astype(float) - p) ** 2) / len(df)),
            "ipcw_observed_risk": risk,
            "ipcw_predicted_to_observed": float(p.mean() / risk) if risk > 0 else np.nan,
            "ipcw_effective_rows": float(weights.sum() ** 2 / np.sum(weights ** 2)) if weights.any() else 0.0}


def evaluation_metrics(df: pd.DataFrame, rate: np.ndarray, km: CensoringKM | None = None,
                       horizon: float = 30.0, min_survival: float = 0.05) -> dict:
    t, d = df.exposure_minutes.to_numpy(float), df.event.to_numpy(int)
    losses = event_time_nll(rate, t, d)
    integrated = float(np.sum(rate * t))
    result = {
        "rows": len(df), "episodes": df._episode_group.nunique(), "events": int(d.sum()),
        "exposure_minutes": float(t.sum()), "event_time_nll_sum": float(losses.sum()),
        "event_time_nll_per_row": float(losses.mean()),
        "event_time_nll_per_30_exposure_minutes": float(losses.sum() / (t.sum() / 30)),
        "observed_events_per_1000_minutes": float(1000 * d.sum() / t.sum()),
        "predicted_integrated_hazard": integrated,
        "integrated_hazard_to_events": integrated / d.sum() if d.sum() else np.nan,
        "mean_predicted_p30": float(probability_from_rate(rate, 30).mean()),
    }
    # Integrated hazard uses observed stopping times. It is a compensator
    # diagnostic, NOT sum of prospective first-fill probabilities.
    if km is not None:
        result.update(ipcw_horizon_metrics(df, rate, km, horizon, min_survival))
    return result


def calibration_by_group(df: pd.DataFrame, rate: np.ndarray, group: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame({"group": group.astype(str).to_numpy(), "events": df.event.to_numpy(),
                      "exposure_minutes": df.exposure_minutes.to_numpy(),
                      "integrated_hazard": rate * df.exposure_minutes.to_numpy()})
    result = f.groupby("group", sort=True).agg(rows=("events", "size"), events=("events", "sum"),
                    exposure_minutes=("exposure_minutes", "sum"), integrated_hazard=("integrated_hazard", "sum")).reset_index()
    result["observed_rate_per_1000_minutes"] = 1000 * result.events / result.exposure_minutes
    result["predicted_rate_per_1000_minutes"] = 1000 * result.integrated_hazard / result.exposure_minutes
    result["integrated_hazard_to_events"] = result.integrated_hazard.div(result.events.replace(0, np.nan))
    return result


def rate_bin_edges(training_rates: np.ndarray, bins: int = 10) -> np.ndarray:
    # Boundaries are learned on training predictions, not evaluation labels.
    inside = np.unique(np.quantile(training_rates, np.linspace(0, 1, bins + 1)[1:-1]))
    if np.ptp(training_rates) < 1e-15:
        inside = np.array([])
    return np.r_[-np.inf, inside, np.inf]


def grouped_bootstrap(df: pd.DataFrame, predictions: dict[str, np.ndarray],
                      repetitions: int = 300, seed: int = 20261001) -> pd.DataFrame:
    """Paired episode bootstrap of test NLL differences and compensator ratios.

    Keeps within-episode rows together; does not capture cross-episode day/market
    dependence and does not include refitting/hyperparameter uncertainty.
    """
    if repetitions <= 0:
        return pd.DataFrame()
    codes, groups = pd.factorize(df._episode_group, sort=False)
    if len(groups) < 2:
        return pd.DataFrame()
    t, d = df.exposure_minutes.to_numpy(float), df.event.to_numpy(float)
    exposure = np.bincount(codes, weights=t)
    events = np.bincount(codes, weights=d)
    totals = {name: np.bincount(codes, weights=event_time_nll(rate, t, d)) for name, rate in predictions.items()}
    hazards = {name: np.bincount(codes, weights=rate * t) for name, rate in predictions.items()}
    pairs = [(name, "template_baseline") for name in predictions if name != "template_baseline"]
    if "price" in predictions and "no_price" in predictions:
        pairs.append(("price", "no_price"))
    draws = {f"nll_difference:{a}-{b}": [] for a, b in pairs}
    draws.update({f"hazard_to_events:{name}": [] for name in predictions})
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        idx = rng.integers(0, len(groups), size=len(groups))
        denom = exposure[idx].sum() / 30.0
        for a, b in pairs:
            draws[f"nll_difference:{a}-{b}"].append((totals[a][idx].sum() - totals[b][idx].sum()) / denom)
        for name in predictions:
            n = events[idx].sum()
            draws[f"hazard_to_events:{name}"].append(hazards[name][idx].sum() / n if n else np.nan)
    rows = []
    for key, values in draws.items():
        finite = np.asarray(values)[np.isfinite(values)]
        if not len(finite):
            continue
        if key.startswith("nll_difference"):
            a, b = key.split(":", 1)[1].split("-")
            estimate = (totals[a].sum() - totals[b].sum()) / (exposure.sum() / 30)
        else:
            name = key.split(":", 1)[1]
            estimate = hazards[name].sum() / events.sum() if events.sum() else np.nan
        rows.append({"metric": key, "estimate": estimate,
                     "ci_2.5_percent": np.quantile(finite, 0.025), "ci_97.5_percent": np.quantile(finite, 0.975),
                     "valid_bootstrap_repetitions": len(finite), "resampling_unit": "parent_episode"})
    return pd.DataFrame(rows)
