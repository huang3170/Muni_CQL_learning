"""First-fill rates using the supplied quote columns and unchanged timestamps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class TrainingConfig:
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    # Start-of-day boundaries in the same clock convention as the input.
    validation_start: str | None = None
    test_start: str | None = None
    alphas: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1)
    price_scale: float = 0.10
    max_iter: int = 1500
    strict: bool = True
    compute_ipcw: bool = False
    horizon_minutes: float = 30.0
    min_censor_survival: float = 0.05
    bootstrap_repetitions: int = 300
    seed: int = 20261001


def _numeric(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    values = df[name]
    present = values.notna() & values.astype("string").str.strip().ne("").fillna(False)
    result = pd.to_numeric(values, errors="coerce").astype(float)
    if (present & result.isna()).any() or np.isinf(result).any():
        raise ValueError(f"{name} must contain finite numeric values or missing values.")
    return result


def _datetime(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df or not pd.api.types.is_datetime64_any_dtype(df[name]):
        raise ValueError(f"{name} must already have a pandas datetime dtype; timestamps are used unchanged.")
    return df[name]


def fill_events(df: pd.DataFrame) -> pd.Series:
    """Arrival is defined solely by a nonempty first_fill_level."""
    values = df["first_fill_level"]
    return (values.notna() & values.astype("string").str.strip().ne("").fillna(False)).astype(int)


def price_offsets(df: pd.DataFrame) -> pd.DataFrame:
    """Read native offsets, using offer minus mid only to fill missing offsets.

    Prices retained for an inactive level are ignored. No input is mutated.
    """
    offsets = pd.DataFrame(index=df.index)
    mid = _numeric(df, "mid_price")
    for level in (1, 2, 3):
        name = f"l{level}_vs_mid"
        values = _numeric(df, name).fillna(_numeric(df, f"l{level}_price") - mid)
        offsets[name] = values.mask(_numeric(df, f"l{level}_active").eq(0))
    return offsets


def chronological_split(df: pd.DataFrame, config: TrainingConfig) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Split at input-clock midnight, purging individual crossing intervals.

    Episodes may span partitions. Input timestamp values are never modified.
    """
    if (config.validation_start is None) != (config.test_start is None):
        raise ValueError("Supply both validation_start and test_start, or neither.")
    start = _datetime(df, "cycle_time")
    _datetime(df, "quote_end_time")
    dates = pd.DatetimeIndex(start.dt.normalize().unique()).sort_values()
    if config.validation_start is not None:
        def boundary(value: str) -> pd.Timestamp:
            stamp = pd.Timestamp(value)
            if pd.isna(stamp) or stamp != stamp.normalize():
                raise ValueError("Split boundaries must be midnight; supply dates such as YYYY-MM-DD.")
            return stamp
        cut1, cut2 = boundary(config.validation_start), boundary(config.test_start)
    else:
        if len(dates) < 5:
            raise ValueError("Need at least five observed dates for automatic train/validation/test splitting.")
        if not (0 < config.train_fraction < 1 and 0 < config.validation_fraction < 1
                and config.train_fraction + config.validation_fraction < 1):
            raise ValueError("Invalid chronological split fractions.")
        i = max(1, int(len(dates) * config.train_fraction))
        j = min(len(dates) - 1, max(i + 1, int(len(dates) * (config.train_fraction + config.validation_fraction))))
        cut1, cut2 = dates[i], dates[j]
    if not cut1 < cut2:
        raise ValueError("validation_start must precede test_start.")
    frame = df.copy()
    try:
        frame["split"] = np.where(start < cut1, "train", np.where(start < cut2, "validation", "test"))
    except TypeError as exc:
        raise ValueError("Split boundaries and input timestamps must use the same clock convention.") from exc
    purge = pd.Series(False, index=frame.index)
    for cut in (cut1, cut2):
        purge |= frame.cycle_time.lt(cut) & frame.quote_end_time.gt(cut)
    frame.loc[purge, "split"] = "purged_boundary"
    parts = {name: frame.loc[frame.split.eq(name)].copy() for name in ("train", "validation", "test")}
    for name, part in parts.items():
        if part.empty:
            raise ValueError(f"{name} is empty after boundary purge. Set explicit dates and inspect interval timestamps.")
    if parts["train"].event.sum() < 1 or parts["validation"].event.sum() < 1:
        raise ValueError("Training and validation each need observed fill events.")
    audit_rows = []
    for name, part in frame.groupby("split", sort=False):
        audit_rows.append({"split": name, "rows": len(part), "events": int(part.event.sum()),
                           "exposure_minutes": float(part.exposure_minutes.sum()),
                           "episodes": part._episode_group.nunique(),
                           "start": part.cycle_time.min().isoformat(),
                           "end": part.quote_end_time.max().isoformat()})
    episode_partition_counts = frame.loc[~purge].groupby(
        ["cusip", "episode_id"], dropna=False, sort=False).split.nunique()
    details = {"validation_start": cut1.isoformat(), "test_start": cut2.isoformat(),
               "purged_row_fraction": float(purge.mean()),
               "purged_event_fraction": float(frame.loc[purge, "event"].sum() / max(frame.event.sum(), 1)),
               "purged_exposure_fraction": float(frame.loc[purge, "exposure_minutes"].sum() / frame.exposure_minutes.sum()),
               "eligible_episodes": int(frame._episode_group.nunique()),
               "episodes_in_multiple_splits": int(episode_partition_counts.gt(1).sum()),
               "split_rule": "chronological input-clock midnight boundaries; purge individual crossing intervals; allow shared episodes"}
    return parts, pd.DataFrame(audit_rows), details


def state_features(df: pd.DataFrame) -> pd.DataFrame:
    """Allowlisted quote-time state; IDs, outcomes and exposure never enter X.

    Offer offsets share one bounded price term in DesignMatrix. Only their
    differences enter the state, preserving the model's common-shift constraint.
    Numeric units and timestamp values are taken directly from the input.
    """
    start = _datetime(df, "cycle_time")
    minute = start.dt.hour * 60 + start.dt.minute + start.dt.second / 60
    features = pd.DataFrame(index=df.index)
    for source, target in (("quantity", "log_quantity"), ("cep_age_min", "log_cep_age")):
        values = _numeric(df, source)
        if values.lt(0).any():
            raise ValueError(f"{source} must be nonnegative or missing.")
        features[target] = np.log1p(values)
    offsets = price_offsets(df)
    for level in (2, 3):
        features[f"gap_l{level}"] = offsets[f"l{level}_vs_mid"] - offsets.l1_vs_mid
    for name in ("mid_price", "time_to_maturity", "coupon", "liquidity", "cep_bid_ask_width"):
        features[name] = _numeric(df, name)
    # Both supplied market fields and their original units remain authoritative.
    features["cep_bid_ask_width"] = features.cep_bid_ask_width.fillna(
        _numeric(df, "ask_price") - _numeric(df, "bid_price"))
    for name in ("time_to_maturity", "coupon", "cep_bid_ask_width"):
        if features[name].lt(0).any():
            raise ValueError(f"{name} must be nonnegative or missing.")
    if features.mid_price.le(0).any():
        raise ValueError("mid_price must be positive or missing.")
    features["time_sin"] = np.sin(2 * np.pi * minute / 1440)
    features["time_cos"] = np.cos(2 * np.pi * minute / 1440)
    features["active_template"] = template_keys(df)
    rating = df["rating"] if "rating" in df else pd.Series(pd.NA, index=df.index, dtype="string")
    features["rating"] = rating.astype("string").str.strip().replace("", pd.NA).fillna("__MISSING__").astype(str)
    return features


def template_keys(df: pd.DataFrame) -> pd.Series:
    masks = pd.concat([_numeric(df, f"l{i}_active") for i in (1, 2, 3)], axis=1)
    if (masks.notna() & ~masks.isin([0, 1])).any().any() or masks.eq(0).all(axis=1).any():
        raise ValueError("Active-level masks must be 0, 1 or missing, with at least one possible active level.")
    return masks.astype("Int64").astype("string").fillna("?").agg("".join, axis=1)


class DesignMatrix:
    def __init__(self, with_price: bool, price_scale: float = 0.10):
        self.with_price, self.price_scale = with_price, price_scale
        if not np.isfinite(price_scale) or price_scale <= 0:
            raise ValueError("price_scale must be finite and positive.")

    def fit(self, df: pd.DataFrame) -> "DesignMatrix":
        f = state_features(df)
        self.numeric = [c for c in f if pd.api.types.is_numeric_dtype(f[c]) and f[c].notna().any()]
        self.categorical = [c for c in f if not pd.api.types.is_numeric_dtype(f[c]) and f[c].nunique() > 1]
        names = []
        self.imputer, self.scaler = None, None
        if self.numeric:
            self.imputer = SimpleImputer(strategy="median", add_indicator=True)
            n = self.imputer.fit_transform(f[self.numeric])
            # Center first for stable weighted variance on constant columns.
            origin = n[0].copy()
            self.scaler = StandardScaler().fit(n - origin, sample_weight=df.exposure_minutes.to_numpy())
            self.scaler.mean_ += origin
            # Subtracting offer offsets can leave rounding noise in an otherwise
            # constant gap. Do not amplify that noise into a predictive signal.
            tolerance = 128 * np.finfo(float).eps * np.maximum(1, np.max(np.abs(n), axis=0))
            constant = np.ptp(n, axis=0) <= tolerance
            self.scaler.var_[constant] = 0.0
            self.scaler.scale_[constant] = 1.0
            names += list(self.imputer.get_feature_names_out(self.numeric))
        self.encoder = None
        if self.categorical:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                                         min_frequency=20, max_categories=20, dtype=float)
            self.encoder.fit(f[self.categorical])
            names += list(self.encoder.get_feature_names_out(self.categorical))
        if self.with_price:
            price = price_offsets(df)[["l1_vs_mid"]]
            if price.l1_vs_mid.isna().all():
                raise ValueError("Training the price model requires at least one observed l1_vs_mid or l1_price - mid_price.")
            self.price_imputer = SimpleImputer(strategy="median", add_indicator=True).fit(price)
            names += list(self.price_imputer.get_feature_names_out())[1:]
            names.append("l1_vs_mid_per_0.10" if self.price_scale == 0.1 else "l1_vs_mid_scaled")
        self.feature_names = ["intercept", *names]
        return self

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        f = state_features(df)
        arrays = [sparse.csr_matrix(np.ones((len(df), 1)))]
        if self.imputer is not None:
            num = self.scaler.transform(self.imputer.transform(f[self.numeric]))
            arrays.append(sparse.csr_matrix(num))
        if self.encoder is not None:
            arrays.append(self.encoder.transform(f[self.categorical]))
        if self.with_price:
            price = self.price_imputer.transform(price_offsets(df)[["l1_vs_mid"]])
            if price.shape[1] > 1:
                arrays.append(sparse.csr_matrix(price[:, 1:]))
            # The bounded common-shift coefficient always occupies the last slot.
            arrays.append(sparse.csr_matrix(price[:, :1] / self.price_scale))
        result = sparse.hstack(arrays, format="csr")
        if not np.isfinite(result.data).all():
            raise ValueError("Nonfinite design matrix.")
        return result


def event_time_nll(rate: np.ndarray, exposure: np.ndarray, event: np.ndarray) -> np.ndarray:
    rate = np.asarray(rate, dtype=float)
    if not np.isfinite(rate).all() or (rate <= 0).any():
        raise ValueError("Rates must be finite and strictly positive.")
    return rate * np.asarray(exposure) - np.asarray(event) * np.log(rate)


def probability_from_rate(rate: np.ndarray, horizon_minutes: float | np.ndarray = 30.0) -> np.ndarray:
    if np.any(np.asarray(horizon_minutes) < 0):
        raise ValueError("Prediction horizon cannot be negative.")
    return -np.expm1(-np.asarray(rate) * np.asarray(horizon_minutes))


class TemplateRateBaseline:
    """Template rates shrunk toward the pooled rate with 3,000 prior minutes."""
    def __init__(self, prior_minutes: float = 3000.0):
        self.prior_minutes = prior_minutes

    def fit(self, df: pd.DataFrame) -> "TemplateRateBaseline":
        event = fill_events(df)
        self.global_rate = float(event.sum() / df.exposure_minutes.sum())
        if self.global_rate <= 0:
            raise ValueError("At least one training event is required.")
        table = df.assign(event=event, template=template_keys(df)).groupby("template").agg(
            events=("event", "sum"), exposure=("exposure_minutes", "sum"))
        self.rates = ((table.events + self.prior_minutes * self.global_rate) /
                      (table.exposure + self.prior_minutes)).to_dict()
        return self

    def predict_rate(self, df: pd.DataFrame) -> np.ndarray:
        return template_keys(df).map(self.rates).fillna(self.global_rate).to_numpy(dtype=float)

    def predict_proba(self, df: pd.DataFrame, horizon_minutes: float = 30.0) -> np.ndarray:
        return probability_from_rate(self.predict_rate(df), horizon_minutes)


class ExponentialFillModel:
    """Regularized piecewise exponential rate with an optional nonpositive price coefficient.

    Objective: total event-time NLL / (total exposure / 30) + alpha/2 * ||beta||².
    The intercept is unpenalized. Exposure, not the number of rows, normalizes
    the likelihood, so arbitrary slicing does not change its relative weight.
    """
    def __init__(self, alpha: float = 0.001, with_price: bool = True,
                 price_scale: float = 0.10, max_iter: int = 1500):
        if alpha < 0:
            raise ValueError("alpha cannot be negative.")
        self.alpha, self.with_price = alpha, with_price
        self.design = DesignMatrix(with_price, price_scale)
        self.max_iter = max_iter

    @staticmethod
    def _objective(theta: np.ndarray, x: sparse.csr_matrix, t: np.ndarray,
                   d: np.ndarray, normalizer: float, alpha: float) -> tuple[float, np.ndarray]:
        eta = np.asarray(x @ theta).ravel()
        # Convex quadratic continuation avoids overflow during rejected search
        # steps. The accepted solution is required to lie in the exact exp region.
        high = eta > 30.0
        rate = np.exp(np.minimum(eta, 30.0))
        derivative = rate.copy()
        z = eta[high] - 30.0
        rate[high] = np.exp(30.0) * (1 + z + 0.5 * z * z)
        derivative[high] = np.exp(30.0) * (1 + z)
        value = float(np.sum(t * rate - d * eta) / normalizer + 0.5 * alpha * np.dot(theta[1:], theta[1:]))
        gradient = np.asarray(x.T @ (t * derivative - d)).ravel() / normalizer
        gradient[1:] += alpha * theta[1:]
        return value, gradient

    def fit(self, df: pd.DataFrame) -> "ExponentialFillModel":
        self.design.fit(df)
        x = self.design.transform(df)
        return self._fit_matrix(x, df)

    def _fit_matrix(self, x: sparse.csr_matrix, df: pd.DataFrame) -> "ExponentialFillModel":
        t, d = df.exposure_minutes.to_numpy(float), fill_events(df).to_numpy(float)
        if t.sum() <= 0 or d.sum() <= 0:
            raise ValueError("Training requires positive exposure and at least one event.")
        initial = np.zeros(x.shape[1])
        initial[0] = np.log(d.sum() / t.sum())
        bounds = [(None, None)] * x.shape[1]
        if self.with_price:
            bounds[-1] = (None, 0.0)
        opt = minimize(self._objective, initial, args=(x, t, d, t.sum() / 30.0, self.alpha),
                       jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": self.max_iter, "ftol": 1e-11, "gtol": 1e-7, "maxls": 50})
        if not opt.success:
            raise RuntimeError(f"Optimizer did not converge: {opt.message}")
        if np.max(x @ opt.x) >= 30:
            raise RuntimeError("Fitted rates exceeded the exact exponential region; inspect units/features.")
        self.coef_ = opt.x
        self.fit_details = {"iterations": int(opt.nit), "objective": float(opt.fun),
                            "converged": bool(opt.success), "message": str(opt.message)}
        return self

    def predict_rate(self, df: pd.DataFrame) -> np.ndarray:
        eta = np.asarray(self.design.transform(df) @ self.coef_).ravel()
        if not np.isfinite(eta).all() or np.any(eta > 50) or np.any(eta < -700):
            raise ValueError("Extreme out-of-domain predictions; inspect feature units and support.")
        return np.exp(eta)

    def predict_proba(self, df: pd.DataFrame, horizon_minutes: float = 30.0) -> np.ndarray:
        return probability_from_rate(self.predict_rate(df), horizon_minutes)

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.design.feature_names, "coefficient": self.coef_})

    @property
    def price_coefficient_per_price_point(self) -> float:
        return float(self.coef_[-1] / self.design.price_scale) if self.with_price else 0.0
