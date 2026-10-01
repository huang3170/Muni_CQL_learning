"""Training orchestration and reviewable CSV/JSON/model outputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from .core import (TrainingConfig, DesignMatrix, ExponentialFillModel, TemplateRateBaseline,
                   chronological_split, event_time_nll, probability_from_rate, template_keys,
                   state_features, price_offsets, fill_events)
from .evaluation import (CensoringKM, evaluation_metrics, calibration_by_group,
                         rate_bin_edges, grouped_bootstrap)


@dataclass
class ExperimentResult:
    models: dict
    selected_model_name: str
    metrics: pd.DataFrame
    tuning_results: pd.DataFrame
    split_audit: pd.DataFrame
    exclusions: pd.DataFrame
    predictions: pd.DataFrame
    calibration: pd.DataFrame
    bootstrap_intervals: pd.DataFrame
    coefficients: pd.DataFrame
    feature_missingness: pd.DataFrame
    metadata: dict

    @property
    def selected_model(self):
        return self.models[self.selected_model_name]

    def save(self, output_dir: str | Path) -> None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for name in ["metrics", "tuning_results", "split_audit", "exclusions", "predictions",
                     "calibration", "bootstrap_intervals", "coefficients", "feature_missingness"]:
            getattr(self, name).to_csv(directory / f"{name}.csv", index=False)
        with (directory / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2, allow_nan=False)
        # Models are fitted only on training data, with validation-selected alpha.
        joblib.dump({"models": self.models, "selected_model_name": self.selected_model_name,
                     "metadata": self.metadata}, directory / "models.joblib")


def _validate_intervals(raw: pd.DataFrame, strict: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check the supplied columns without renaming fields or changing timestamps."""
    required = {
        "cusip", "episode_id", "cycle_time", "quote_end_time", "exposure_minutes",
        "first_fill_level", "quantity", "mid_price",
        *(f"l{level}_{field}" for level in (1, 2, 3) for field in ("active", "price")),
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if raw.empty:
        raise ValueError("Input dataframe is empty.")
    df = raw.copy().reset_index(drop=True)
    reasons: list[list[str]] = [[] for _ in range(len(df))]

    def flag(mask: pd.Series | np.ndarray, reason: str) -> None:
        for index in np.flatnonzero(np.asarray(pd.Series(mask).fillna(False), dtype=bool)):
            reasons[index].append(reason)

    for name in ["cusip", "episode_id"]:
        flag(df[name].isna() | df[name].astype("string").str.strip().eq(""), f"MISSING_{name}")
    for name in ["cycle_time", "quote_end_time"]:
        if not pd.api.types.is_datetime64_any_dtype(df[name]):
            raise ValueError(f"{name} must already have a pandas datetime dtype; timestamps are not converted.")
        flag(df[name].isna(), f"INVALID_{name}")
    if "cep_time" in df and df.cep_time.notna().any():
        if not pd.api.types.is_datetime64_any_dtype(df.cep_time):
            raise ValueError("cep_time must already have a pandas datetime dtype; timestamps are not converted.")

    numeric_columns = [
        "quantity", "exposure_minutes", "mid_price", "cep_age_min", "cep_bid_ask_width",
        "time_to_maturity", "coupon", "liquidity", "bid_price", "ask_price",
        *(f"l{level}_{field}" for level in (1, 2, 3) for field in ("active", "price", "vs_mid")),
    ]
    for name in numeric_columns:
        if name not in df:
            continue
        present = df[name].notna() & df[name].astype("string").str.strip().ne("")
        numeric = pd.to_numeric(df[name], errors="coerce").astype(float)
        flag((present & numeric.isna()) | np.isinf(numeric), f"INVALID_{name.upper()}")
        df[name] = numeric

    # The source label is authoritative, including when an old event column exists.
    df["event"] = fill_events(df)
    try:
        wall_minutes = (df.quote_end_time - df.cycle_time).dt.total_seconds() / 60
        if "cep_time" in df and df.cep_time.notna().any():
            flag(df.cep_time.gt(df.cycle_time), "FUTURE_CEP")
    except (TypeError, ValueError) as exc:
        raise ValueError("Timestamp columns must use compatible datetime conventions; no timezone conversion is performed.") from exc
    flag(~np.isfinite(df.exposure_minutes) | df.exposure_minutes.le(0), "INVALID_EXPOSURE")
    flag(wall_minutes.isna() | wall_minutes.le(0), "INVALID_INTERVAL_ORDER")
    flag(df.exposure_minutes.gt(wall_minutes + 1 / 60), "EXPOSURE_EXCEEDS_WALL_TIME")
    flag(df.quantity.le(0), "INVALID_QUANTITY")
    flag(df.mid_price.le(0), "INVALID_MID_PRICE")
    for name in ["cep_age_min", "cep_bid_ask_width", "time_to_maturity", "coupon"]:
        if name in df:
            flag(df[name].lt(0), f"INVALID_{name.upper()}")
    for level in (1, 2, 3):
        active, price = df[f"l{level}_active"], df[f"l{level}_price"]
        flag(active.notna() & ~active.isin([0, 1]), f"INVALID_L{level}_MASK")
        flag(active.ne(0) & price.le(0), f"INVALID_L{level}_PRICE")
    masks = df[[f"l{level}_active" for level in (1, 2, 3)]]
    flag(masks.eq(0).all(axis=1), "NO_ACTIVE_LEVEL")
    audit = pd.DataFrame({
        "row_index": raw.index.to_numpy(), "cusip": df.cusip, "episode_id": df.episode_id,
        "exclusion_reason": ["|".join(dict.fromkeys(row)) for row in reasons],
    })
    bad = audit.exclusion_reason.ne("")
    if strict and bad.any():
        summary = audit.loc[bad, "exclusion_reason"].str.split("|").explode().value_counts().to_dict()
        raise ValueError(f"{int(bad.sum())} rows fail validation: {summary}. "
                         "Fix the source data or use strict=False and inspect exclusions.")
    clean = df.loc[~bad].copy()
    if clean.empty:
        raise ValueError("No eligible rows after validation.")
    clean["_episode_group"] = list(zip(clean.cusip, clean.episode_id))
    clean = clean.sort_values("cycle_time", kind="stable").reset_index(drop=True)
    for _, group in clean.groupby("cusip", sort=False):
        if group.cycle_time.lt(group.quote_end_time.cummax().shift()).any():
            raise ValueError("Overlapping intervals within a CUSIP. Aggregate levels/venues first; "
                             "if multiple independent books exist, extend the entity key explicitly.")
    return clean, audit.loc[bad].reset_index(drop=True)


def run_experiment(df: pd.DataFrame, output_dir: str | Path | None = None,
                   config: TrainingConfig | None = None) -> ExperimentResult:
    """Fit three models, select using validation only, evaluate untouched test.

    Models are NOT refitted on train+validation; the evaluation refers exactly
    to the saved models. No probability recalibration is fitted on the test set.
    """
    config = config or TrainingConfig()
    if not config.alphas or any(a < 0 or not np.isfinite(a) for a in config.alphas):
        raise ValueError("alphas must be a nonempty list of finite nonnegative values.")
    clean, exclusions = _validate_intervals(df, strict=config.strict)
    parts, split_audit, split_details = chronological_split(clean, config)
    train, validation, test = (parts[k] for k in ["train", "validation", "test"])
    if split_details["purged_row_fraction"] > 0:
        warnings.warn("Intervals crossing date boundaries were purged. "
                      "Inspect split_audit, interval endpoints, and overnight rows. "
                      "Parent episodes are not purged as groups.")
    for name, part in parts.items():
        if part.event.sum() < 100:
            warnings.warn(f"{name} has only {int(part.event.sum())} fill events; comparisons may be unstable.")
    models = {"template_baseline": TemplateRateBaseline().fit(train)}
    tuning_rows = []
    for name, with_price in [("no_price", False), ("price", True)]:
        # Preprocessing is fitted on training rows once per model family.
        design = DesignMatrix(with_price, config.price_scale).fit(train)
        x_train = design.transform(train)
        best_score, best = np.inf, None
        for alpha in config.alphas:
            model = ExponentialFillModel(alpha, with_price, config.price_scale, config.max_iter)
            model.design = design
            try:
                model._fit_matrix(x_train, train)
                rate = model.predict_rate(validation)
                score = float(event_time_nll(rate, validation.exposure_minutes.to_numpy(), validation.event.to_numpy()).sum()
                              / (validation.exposure_minutes.sum() / 30))
                tuning_rows.append({"model": name, "alpha": alpha, "validation_nll_per_30_exposure_minutes": score,
                                    "converged": True, "iterations": model.fit_details["iterations"], "error": ""})
                if score < best_score:
                    best_score, best = score, model
            except RuntimeError as exc:
                tuning_rows.append({"model": name, "alpha": alpha, "validation_nll_per_30_exposure_minutes": np.nan,
                                    "converged": False, "iterations": None, "error": str(exc)})
        if best is None:
            raise RuntimeError(f"No {name} candidate converged; inspect features/units or increase max_iter.")
        models[name] = best
    # Winner is chosen before accessing test outcomes or fitting censor diagnostics.
    scores = {name: float(event_time_nll(model.predict_rate(validation), validation.exposure_minutes.to_numpy(),
                                       validation.event.to_numpy()).sum()) for name, model in models.items()}
    selected = min(scores, key=scores.get)
    km = CensoringKM().fit(train) if config.compute_ipcw else None
    metric_rows, prediction_frames, calibration_frames, missingness_rows = [], [], [], []
    test_predictions = {}
    bin_edges = {name: rate_bin_edges(model.predict_rate(train)) for name, model in models.items()}
    quantity_edges = rate_bin_edges(train.quantity.dropna().to_numpy()) if train.quantity.notna().any() else None
    for split, part in parts.items():
        features = state_features(part)
        offsets = price_offsets(part)
        reported_features = features.assign(l1_vs_mid=offsets.l1_vs_mid)
        for feature in reported_features:
            values = reported_features[feature]
            missing = values.isna()
            if not pd.api.types.is_numeric_dtype(values):
                missing |= values.eq("__MISSING__")
                if feature == "active_template":
                    missing |= values.str.contains("?", regex=False)
            missingness_rows.append({"split": split, "feature": feature, "rows": len(part),
                                     "missing_rows": int(missing.sum()), "missing_fraction": float(missing.mean())})
        for name, model in models.items():
            rate = model.predict_rate(part)
            metric_rows.append({"model": name, "split": split, **evaluation_metrics(
                part, rate, km, config.horizon_minutes, config.min_censor_survival)})
            if split == "test":
                test_predictions[name] = rate
            if split == "train":
                continue
            prediction = part[["cusip", "episode_id", "cycle_time", "quote_end_time", "first_fill_level",
                               "event", "exposure_minutes", "quantity"]].copy()
            prediction["l1_vs_mid"] = offsets.l1_vs_mid
            prediction["split"], prediction["model"] = split, name
            prediction["rate_per_minute"] = rate
            prediction["p_fill_30m"] = probability_from_rate(rate, 30)
            prediction["integrated_hazard_observed_exposure"] = rate * part.exposure_minutes.to_numpy()
            prediction["event_time_nll"] = event_time_nll(rate, part.exposure_minutes.to_numpy(), part.event.to_numpy())
            prediction_frames.append(prediction)
            groups = {
                "predicted_rate_bin": pd.cut(rate, bins=bin_edges[name], include_lowest=True).astype(str),
                "active_template": template_keys(part),
                "date": part.cycle_time.dt.strftime("%Y-%m-%d"),
                "price_shift": pd.cut(offsets.l1_vs_mid, [-np.inf, 0, .05, .1, .2, .3, .5, np.inf], include_lowest=True).astype(str),
            }
            if "cep_age_min" in train and train.cep_age_min.notna().any():
                groups["cep_age_min"] = pd.cut(part.cep_age_min, [-np.inf, 0, 30, 60, 120, np.inf], include_lowest=True).astype(str)
            if quantity_edges is not None:
                groups["quantity"] = pd.cut(part.quantity, quantity_edges, include_lowest=True).astype(str)
            if "liquidity" in train:
                groups["liquidity_missing"] = part.liquidity.isna().map({True: "missing", False: "observed"})
            for group_name, values in groups.items():
                grouped = calibration_by_group(part, rate, pd.Series(np.asarray(values)))
                grouped.insert(0, "grouping", group_name)
                grouped.insert(0, "split", split)
                grouped.insert(0, "model", name)
                calibration_frames.append(grouped)
    coef = pd.concat([model.coefficients().assign(model=name) for name, model in models.items()
                      if isinstance(model, ExponentialFillModel)], ignore_index=True)
    bootstrap = grouped_bootstrap(test, test_predictions, config.bootstrap_repetitions, config.seed)
    price_model = models["price"]
    training_offsets = price_offsets(train).l1_vs_mid.dropna()
    metadata = {
        "config": asdict(config), "input_rows": len(df), "eligible_rows": len(clean),
        "eligible_events": int(clean.event.sum()), "selected_model_name": selected,
        "selection_rule": "minimum validation event-time NLL; test never used for selection",
        "selected_alpha": {name: model.alpha for name, model in models.items() if isinstance(model, ExponentialFillModel)},
        "price_coefficient_per_price_point": price_model.price_coefficient_per_price_point,
        "price_coefficient_nonpositive": bool(price_model.price_coefficient_per_price_point <= 0),
        "training_l1_vs_mid_range": ([float(training_offsets.min()), float(training_offsets.max())]
                                     if len(training_offsets) else None),
        "preprocessing_fit_scope": "training data only", "refit_on_train_plus_validation": False,
        "rate_unit": "events per active market minute", "timestamp_convention": "input timestamps preserved; no timezone conversion",
        "quantity_note": "quantity is offered size in the supplied units",
        "maturity_note": "time_to_maturity is used in the supplied units",
        "label_note": "nonempty first_fill_level is an event; quote_end_time and exposure stop at first fill",
        "censor_note": "empty first_fill_level means no target fill observed until quote_end_time; exit mechanism unavailable",
        "integrated_hazard_note": "sum(lambda * observed exposure) is a compensator diagnostic, not a prospective expected fill count",
        "ipcw_note": "optional; assumes marginal independent censoring; RFQs/repricing may violate this; no IPCW by default",
        "bootstrap_note": "paired parent-episode bootstrap; no refitting or cross-episode market-day dependence accounted for",
        "scope": "factual any-first-fill model only; no level routing, fill-size model, causal price estimate, optimizer or RL",
        "split": split_details,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    result = ExperimentResult(models, selected, pd.DataFrame(metric_rows), pd.DataFrame(tuning_rows),
                              split_audit, exclusions, pd.concat(prediction_frames, ignore_index=True),
                              pd.concat(calibration_frames, ignore_index=True), bootstrap, coef,
                              pd.DataFrame(missingness_rows), metadata)
    if output_dir is not None:
        result.save(output_dir)
    return result


def score_price_grid(model, states: pd.DataFrame, deltas: list[float],
                     horizon_minutes: float = 30.0) -> pd.DataFrame:
    """Inspect mathematical price response holding pre-action state fixed.

    Recomputes offer prices and vs-mid offsets consistently, preserving level gaps.
    This is a factual model diagnostic, not a pricing recommendation.
    """
    if not deltas or not np.isfinite(deltas).all():
        raise ValueError("Supply a nonempty finite price grid.")
    offsets = price_offsets(states)
    gaps = {level: offsets[f"l{level}_vs_mid"] - offsets.l1_vs_mid for level in (2, 3)}
    mid = pd.to_numeric(states.mid_price, errors="coerce")
    frames = []
    for delta in sorted(set(deltas)):
        candidate = states.copy()
        for level in (1, 2, 3):
            active = pd.to_numeric(states[f"l{level}_active"], errors="coerce").astype(float).ne(0)
            offset = pd.Series(delta, index=states.index) if level == 1 else delta + gaps[level]
            candidate[f"l{level}_vs_mid"] = offset.where(active)
            candidate[f"l{level}_price"] = (mid + offset).where(active)
        rate = model.predict_rate(candidate)
        frames.append(pd.DataFrame({"state_index": np.arange(len(states)), "l1_vs_mid": delta,
                                    "rate_per_minute": rate, "p_fill": probability_from_rate(rate, horizon_minutes)}))
    return pd.concat(frames, ignore_index=True)
