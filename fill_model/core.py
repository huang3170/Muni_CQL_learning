"""First-fill rates per active market minute; one event at most per interval."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import QuoteSchema, normalize_quote_states, normalize_quote_training


END_REASONS = {
    "FILL", "TIME_SLICE_END", "QUOTE_CHANGE", "QUOTE_INACTIVE",
    "INVENTORY_INCREASE", "RFQ_FILL", "INVENTORY_ADJUSTMENT", "CENSORED",
    "MARKET_CLOSE", "DATA_CUTOFF", "DATA_GAP", "UNKNOWN",
}


@dataclass(frozen=True)
class TrainingConfig:
    timezone: str = "America/New_York"
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    # Optional ISO dates: these are start-of-day boundaries in timezone above.
    validation_start: str | None = None
    test_start: str | None = None
    alphas: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1)
    price_scale: float = 0.10
    max_iter: int = 1500
    strict: bool = True
    # Opt-in: marginal independent-censoring assumption, see README.
    compute_ipcw: bool = False
    horizon_minutes: float = 30.0
    min_censor_survival: float = 0.05
    bootstrap_repetitions: int = 300
    seed: int = 20261001
    # Applies to user column names (cycle_time, quantity, time_to_maturity, ...).
    # None requires timezone-aware source timestamps instead of guessing.
    input_timezone: str | None = "America/New_York"
    quantity_multiplier: float = 1.0
    maturity_unit: str = "days"
    liquidity_kind: str = "numeric"
    quote_end_is_first_fill: bool = True

    def quote_schema(self) -> QuoteSchema:
        return QuoteSchema(self.input_timezone, self.quantity_multiplier,
                           self.maturity_unit, self.liquidity_kind,
                           self.quote_end_is_first_fill)


def _numeric(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").astype(float)


def _utc(series: pd.Series) -> pd.Series:
    # Columns named *_utc: timezone-naive timestamps are explicitly assumed UTC.
    return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")


def prepare_data(raw: pd.DataFrame, strict: bool = True,
                 schema: QuoteSchema | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate, derive state fields, and return eligible rows plus exclusion audit.

    Does not rebuild intervals or infer market calendars. Exposure must already
    represent observable active quoting time. All input frames are copied.
    """
    raw = normalize_quote_training(raw, schema or QuoteSchema())
    required = [
        "interval_id", "position_episode_id", "quote_config_id", "cusip",
        "start_time_utc", "end_time_utc", "exposure_minutes", "end_reason",
        "event", "fill_event_id", "fill_time_utc",
        "l1_active", "l2_active", "l3_active", "l1_price", "l2_price",
        "l3_price", "cep_mid", "cep_asof_time_utc",
        "train_eligible",
    ]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = raw.copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    reasons: list[list[str]] = [[] for _ in range(len(df))]

    def flag(mask: Any, name: str) -> None:
        for i in np.flatnonzero(np.asarray(pd.Series(mask).fillna(True), dtype=bool)):
            reasons[i].append(name)

    eligibility = _numeric(df, "train_eligible")
    if not eligibility.isin([0, 1]).all():
        raise ValueError("train_eligible must be numeric 0 or 1, without missing values.")
    requested = eligibility.eq(1)
    flag(~requested, "USER_EXCLUDED")
    for c in ["interval_id", "position_episode_id", "quote_config_id", "cusip"]:
        flag(df[c].isna() | df[c].astype(str).str.strip().eq(""), f"MISSING_{c}")
        df[c] = df[c].astype("string")
    flag(df.interval_id.duplicated(keep=False), "DUPLICATE_INTERVAL_ID")
    for c in ["start_time_utc", "end_time_utc", "cep_asof_time_utc", "fill_time_utc"]:
        df[c] = _utc(df[c])
    for c in ["event", "exposure_minutes", "inventory_par_start", "quote_quantity", "config_age_minutes",
              "l1_active", "l2_active", "l3_active", "l1_price", "l2_price", "l3_price", "cep_mid"]:
        df[c] = _numeric(df, c)
    for c in ["start_time_utc", "end_time_utc", "cep_asof_time_utc"]:
        flag(df[c].isna(), f"INVALID_{c}")
    wall = (df.end_time_utc - df.start_time_utc).dt.total_seconds() / 60.0
    flag(~np.isfinite(df.exposure_minutes) | df.exposure_minutes.le(0), "INVALID_EXPOSURE")
    flag(wall.le(0) | wall.isna(), "INVALID_INTERVAL_ORDER")
    flag(df.exposure_minutes.gt(wall + 1 / 60), "EXPOSURE_EXCEEDS_WALL_TIME")
    flag(df.cep_asof_time_utc.gt(df.start_time_utc), "FUTURE_CEP")
    flag(~df.event.isin([0, 1]), "UNKNOWN_EVENT")
    flag(~df.end_reason.isin(END_REASONS), "INVALID_END_REASON")
    flag(df.end_reason.eq("UNKNOWN"), "UNKNOWN_END_REASON")
    flag(df.event.eq(1).ne(df.end_reason.eq("FILL")), "EVENT_REASON_MISMATCH")
    positive = df.event.eq(1)
    flag(positive & (df.fill_time_utc.isna() | df.fill_time_utc.ne(df.end_time_utc)), "INVALID_FILL_TIME")
    missing_fill_id = df.fill_event_id.isna() | df.fill_event_id.astype(str).str.strip().eq("")
    flag(positive & missing_fill_id, "MISSING_FILL_EVENT_ID")
    duplicate_event = df.loc[positive & requested, "fill_event_id"].duplicated(keep=False)
    duplicate_index = df.loc[positive & requested].index[duplicate_event]
    flag(df.index.isin(duplicate_index), "DUPLICATE_FILL_EVENT_ID")
    flag(~positive & (df.fill_time_utc.notna() | ~missing_fill_id), "NONFILL_HAS_EVENT_LABEL")
    for c in ["inventory_par_start", "quote_quantity"]:
        flag(np.isinf(df[c]) | df[c].le(0), f"INVALID_{c.upper()}")
    flag(np.isinf(df.config_age_minutes) | df.config_age_minutes.lt(0), "INVALID_CONFIG_AGE")
    for l in (1, 2, 3):
        active, price = df[f"l{l}_active"], df[f"l{l}_price"]
        flag(~active.isin([0, 1]), f"INVALID_L{l}_MASK")
        flag(active.eq(1) & (~np.isfinite(price) | price.le(0)), f"INVALID_L{l}_PRICE")
        flag(active.eq(0) & price.notna(), f"INACTIVE_L{l}_HAS_PRICE")
    # This first version follows the proposal's Level-1 common price anchor.
    flag(df.l1_active.ne(1), "MISSING_L1_ANCHOR")
    flag(~np.isfinite(df.cep_mid) | df.cep_mid.le(0), "INVALID_CEP")
    derived = {
        "delta_l1": df.l1_price - df.cep_mid,
        "gap_l2": (df.l2_price - df.l1_price).where(df.l2_active.eq(1)),
        "gap_l3": (df.l3_price - df.l1_price).where(df.l3_active.eq(1)),
    }
    for c, value in derived.items():
        if c in df:
            supplied = _numeric(df, c)
            flag(supplied.notna() & ((supplied - value).abs().gt(1e-6) | value.isna()), f"INCONSISTENT_{c}")
        df[c] = value
    if "quality_reason" in df:
        flag(df.quality_reason.fillna("").astype(str).str.contains(
            r"UNRESOLVED_TIE|UNKNOWN_OUTCOME", regex=True), "UNRELIABLE_OUTCOME")
    # Bad level/size labels do not remove a valid any-fill observation. These
    # labels are never features and will need separate checks for later models.
    # Conversely, a known nonfill cannot carry a positive quote-fill quantity.
    if "fill_par" in df:
        flag(~positive & _numeric(df, "fill_par").fillna(0).ne(0), "NONFILL_HAS_FILL_PAR")
    audit = pd.DataFrame({
        "interval_id": df.interval_id,
        "requested_for_training": requested,
        "exclusion_reason": ["|".join(r) for r in reasons],
    })
    bad = audit.exclusion_reason.ne("")
    failures = audit.loc[bad & requested]
    if strict and len(failures):
        summary = failures.exclusion_reason.str.split("|").explode().value_counts().to_dict()
        raise ValueError(f"{len(failures)} train_eligible rows fail validation: {summary}. "
                         "Fix upstream labels or explicitly use strict=False and inspect exclusions.")
    clean = df.loc[~bad].copy()
    if clean.empty:
        raise ValueError("No eligible rows after validation.")
    clean["event"] = clean.event.astype(int)
    # Internal group key: a CUSIP plus its original parent episode.
    clean["_episode_group"] = list(zip(clean.cusip.astype(str), clean.position_episode_id.astype(str)))
    clean = clean.sort_values(["start_time_utc", "cusip", "interval_id"]).reset_index(drop=True)
    # Whole-position rows must not overlap; duplicated venue rows would inflate exposure.
    for _, group in clean.groupby("cusip", sort=False):
        max_previous_end = group.end_time_utc.cummax().shift()
        if group.start_time_utc.lt(max_previous_end).any():
            raise ValueError("Overlapping intervals within a CUSIP. Aggregate levels/venues first; "
                             "if multiple independent books exist, extend the entity key explicitly.")
    return clean, audit.loc[bad].reset_index(drop=True)


def chronological_split(df: pd.DataFrame, config: TrainingConfig) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Local-date split, purging only individual intervals crossing a boundary.

    Parent episodes and configuration IDs may appear in multiple partitions.
    This evaluates future quotes, including quotes for inventory already held.
    A row ending exactly at a cut stays on the left; one starting there goes right.
    """
    if (config.validation_start is None) != (config.test_start is None):
        raise ValueError("Supply both validation_start and test_start, or neither.")
    dates = pd.DatetimeIndex(df.start_time_utc.dt.tz_convert(config.timezone).dt.normalize().unique()).sort_values()
    if config.validation_start is not None:
        def boundary(s: str) -> pd.Timestamp:
            stamp = pd.Timestamp(s)
            stamp = stamp.tz_localize(config.timezone) if stamp.tzinfo is None else stamp
            stamp = stamp.tz_convert(config.timezone)
            if pd.isna(stamp) or stamp != stamp.normalize():
                raise ValueError(f"Split boundaries must be midnight in {config.timezone}; "
                                 "supply local dates such as YYYY-MM-DD.")
            return stamp.tz_convert("UTC")
        cut1, cut2 = boundary(config.validation_start), boundary(config.test_start)
    else:
        if len(dates) < 5:
            raise ValueError("Need at least five observed dates for automatic train/validation/test splitting.")
        if not (0 < config.train_fraction < 1 and 0 < config.validation_fraction < 1
                and config.train_fraction + config.validation_fraction < 1):
            raise ValueError("Invalid chronological split fractions.")
        i = max(1, int(len(dates) * config.train_fraction))
        j = min(len(dates) - 1, max(i + 1, int(len(dates) * (config.train_fraction + config.validation_fraction))))
        cut1, cut2 = dates[i].tz_convert("UTC"), dates[j].tz_convert("UTC")
    if not cut1 < cut2:
        raise ValueError("validation_start must precede test_start.")
    frame = df.copy()
    split = np.where(frame.start_time_utc < cut1, "train", np.where(frame.start_time_utc < cut2, "validation", "test"))
    purge = pd.Series(False, index=frame.index)
    for cut in [cut1, cut2]:
        # Never inspect the end of the whole position: later inventory activity
        # cannot make an already completed, historical quote row unavailable.
        purge |= frame.start_time_utc.lt(cut) & frame.end_time_utc.gt(cut)
    frame["split"] = split
    frame.loc[purge, "split"] = "purged_boundary"
    parts = {name: frame.loc[frame.split.eq(name)].copy() for name in ["train", "validation", "test"]}
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
                           "start": part.start_time_utc.min().isoformat(),
                           "end": part.end_time_utc.max().isoformat()})
    episode_partition_counts = frame.loc[~purge].groupby(
        ["cusip", "position_episode_id"], dropna=False, sort=False).split.nunique()
    details = {"validation_start_utc": cut1.isoformat(), "test_start_utc": cut2.isoformat(),
               "timezone": config.timezone,
               "validation_start_local": cut1.tz_convert(config.timezone).isoformat(),
               "test_start_local": cut2.tz_convert(config.timezone).isoformat(),
               "purged_row_fraction": float(purge.mean()),
               "purged_event_fraction": float(frame.loc[purge, "event"].sum() / max(frame.event.sum(), 1)),
               "purged_exposure_fraction": float(frame.loc[purge, "exposure_minutes"].sum() / frame.exposure_minutes.sum()),
               "eligible_episodes": int(frame._episode_group.nunique()),
               "episodes_in_multiple_splits": int(episode_partition_counts.gt(1).sum()),
               "split_rule": "chronological local-midnight boundaries; purge only individual crossing intervals; allow shared episodes/configurations"}
    return parts, pd.DataFrame(audit_rows), details


def state_features(df: pd.DataFrame, timezone: str = "America/New_York",
                   schema: QuoteSchema | None = None) -> pd.DataFrame:
    """Explicit feature allowlist. Labels/end reasons/durations cannot enter X."""
    df = normalize_quote_states(df, schema or QuoteSchema())
    start = _utc(df["start_time_utc"])
    local = start.dt.tz_convert(timezone)
    minute = local.dt.hour * 60 + local.dt.minute + local.dt.second / 60
    features = pd.DataFrame(index=df.index)
    for src, dst in [("inventory_par_start", "log_inventory"), ("quote_quantity", "log_quote_quantity"),
                     ("config_age_minutes", "log_config_age"),
                     ("market_trade_count_1d", "log_market_trade_count_1d"), ("market_trade_par_1d", "log_market_trade_par_1d"),
                     ("last_market_trade_age_minutes", "log_last_trade_age"), ("since_last_buy_minutes", "log_since_last_buy")]:
        values = _numeric(df, src)
        if values.lt(0).any() or np.isinf(values).any():
            raise ValueError(f"{src} must be finite nonnegative values or missing.")
        features[dst] = np.log1p(values)
    for c in ["gap_l2", "gap_l3", "cep_mid", "time_to_maturity_years", "coupon",
              "liquidity_score", "duration_years", "market_spread", "cep_change_30m", "cep_volatility_1d"]:
        features[c] = _numeric(df, c)
        if np.isinf(features[c]).any():
            raise ValueError(f"Infinite feature: {c}")
    for c in ["time_to_maturity_years", "market_spread", "coupon"]:
        if features[c].lt(0).any():
            raise ValueError(f"{c} must be nonnegative or missing.")
    features["time_sin"] = np.sin(2 * np.pi * minute / 1440)
    features["time_cos"] = np.cos(2 * np.pi * minute / 1440)
    cep_age = (start - _utc(df["cep_asof_time_utc"])).dt.total_seconds() / 60
    if cep_age.lt(0).any():
        raise ValueError("Future CEP at prediction time.")
    features["log_cep_age"] = np.log1p(cep_age)
    features["active_template"] = template_keys(df)
    for c in ["rating_bucket", "liquidity_bucket", "sector", "l1_venue_set", "l2_venue_set", "l3_venue_set"]:
        if c in df:
            values = df[c].astype("string").str.strip().replace("", pd.NA).fillna("__MISSING__").astype(str)
            if c.endswith("venue_set"):
                values = values.map(lambda s: "|".join(sorted(set(x.strip() for x in s.split("|") if x.strip()))))
            features[c] = values
        else:
            features[c] = "__MISSING__"
    return features


def template_keys(df: pd.DataFrame) -> pd.Series:
    masks = pd.concat([_numeric(df, f"l{i}_active") for i in (1, 2, 3)], axis=1)
    if not masks.isin([0, 1]).all().all() or masks.sum(axis=1).eq(0).any():
        raise ValueError("Invalid active-level mask.")
    return masks.astype(int).astype(str).agg("".join, axis=1)


class DesignMatrix:
    def __init__(self, with_price: bool, timezone: str, price_scale: float,
                 schema: QuoteSchema | None = None):
        self.with_price, self.timezone, self.price_scale = with_price, timezone, price_scale
        self.schema = schema or QuoteSchema()
        if price_scale <= 0:
            raise ValueError("price_scale must be positive.")

    def fit(self, df: pd.DataFrame) -> "DesignMatrix":
        f = state_features(df, self.timezone, self.schema)
        self.numeric = [c for c in f if pd.api.types.is_numeric_dtype(f[c]) and f[c].notna().any()]
        self.categorical = [c for c in f if not pd.api.types.is_numeric_dtype(f[c]) and f[c].nunique() > 1]
        self.imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        n = self.imputer.fit_transform(f[self.numeric])
        # Center before weighted variance accumulation to avoid tiny negative
        # variances (and sqrt warnings) for constant coupon/mid-price columns.
        origin = n[0].copy()
        self.scaler = StandardScaler().fit(n - origin, sample_weight=df.exposure_minutes.to_numpy())
        self.scaler.mean_ += origin
        names = list(self.imputer.get_feature_names_out(self.numeric))
        self.encoder = None
        if self.categorical:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                                         min_frequency=20, max_categories=20, dtype=float)
            self.encoder.fit(f[self.categorical])
            names += list(self.encoder.get_feature_names_out(self.categorical))
        if self.with_price:
            names.append("delta_l1_per_0.10" if self.price_scale == 0.1 else "delta_l1_scaled")
        self.feature_names = ["intercept", *names]
        return self

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        df = normalize_quote_states(df, self.schema)
        f = state_features(df, self.timezone, self.schema)
        num = self.scaler.transform(self.imputer.transform(f[self.numeric]))
        arrays = [sparse.csr_matrix(np.ones((len(df), 1))), sparse.csr_matrix(num)]
        if self.encoder is not None:
            arrays.append(self.encoder.transform(f[self.categorical]))
        if self.with_price:
            price = _numeric(df, "delta_l1").to_numpy()
            if not np.isfinite(price).all():
                raise ValueError("delta_l1 must be finite for every prediction.")
            arrays.append(sparse.csr_matrix((price / self.price_scale)[:, None]))
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
        self.global_rate = float(df.event.sum() / df.exposure_minutes.sum())
        if self.global_rate <= 0:
            raise ValueError("At least one training event is required.")
        table = df.assign(template=template_keys(df)).groupby("template").agg(
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
                 timezone: str = "America/New_York", price_scale: float = 0.10,
                 max_iter: int = 1500, schema: QuoteSchema | None = None):
        if alpha < 0:
            raise ValueError("alpha cannot be negative.")
        self.alpha, self.with_price = alpha, with_price
        self.design = DesignMatrix(with_price, timezone, price_scale, schema)
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
        t, d = df.exposure_minutes.to_numpy(float), df.event.to_numpy(float)
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
