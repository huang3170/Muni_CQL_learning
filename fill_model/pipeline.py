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
                   prepare_data, chronological_split, event_time_nll, probability_from_rate, template_keys,
                   state_features)
from .schema import normalize_quote_states
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


def run_experiment(df: pd.DataFrame, output_dir: str | Path | None = None,
                   config: TrainingConfig | None = None) -> ExperimentResult:
    """Fit three models, select using validation only, evaluate untouched test.

    Models are NOT refitted on train+validation; the evaluation refers exactly
    to the saved models. No probability recalibration is fitted on the test set.
    """
    config = config or TrainingConfig()
    if not config.alphas or any(a < 0 or not np.isfinite(a) for a in config.alphas):
        raise ValueError("alphas must be a nonempty list of finite nonnegative values.")
    schema = config.quote_schema()
    clean, exclusions = prepare_data(df, strict=config.strict, schema=schema)
    parts, split_audit, split_details = chronological_split(clean, config)
    train, validation, test = (parts[k] for k in ["train", "validation", "test"])
    if split_details["purged_row_fraction"] > 0:
        warnings.warn("Intervals crossing local-date boundaries were purged. "
                      "Inspect split_audit, interval endpoints, and overnight rows. "
                      "Parent episodes and configuration IDs are not purged as groups.")
    for name, part in parts.items():
        if part.event.sum() < 100:
            warnings.warn(f"{name} has only {int(part.event.sum())} fill events; comparisons may be unstable.")
    models = {"template_baseline": TemplateRateBaseline().fit(train)}
    tuning_rows = []
    for name, with_price in [("no_price", False), ("price", True)]:
        # Preprocessing is fitted on training rows once per model family.
        design = DesignMatrix(with_price, config.price_scale, schema).fit(train)
        x_train = design.transform(train)
        best_score, best = np.inf, None
        for alpha in config.alphas:
            model = ExponentialFillModel(alpha, with_price, config.price_scale, config.max_iter, schema)
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
    quantity_edges = rate_bin_edges(train.quote_quantity.dropna().to_numpy()) if train.quote_quantity.notna().any() else None
    for split, part in parts.items():
        features = state_features(part, schema)
        for feature in features:
            missing = features[feature].isna()
            if not pd.api.types.is_numeric_dtype(features[feature]):
                missing |= features[feature].eq("__MISSING__")
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
            prediction = part[["interval_id", "position_episode_id", "quote_config_id", "cusip",
                               "start_time", "end_time", "event", "exposure_minutes", "end_reason", "delta_l1",
                               "quote_quantity"]].copy()
            prediction["split"], prediction["model"] = split, name
            prediction["rate_per_minute"] = rate
            prediction["p_fill_30m"] = probability_from_rate(rate, 30)
            prediction["integrated_hazard_observed_exposure"] = rate * part.exposure_minutes.to_numpy()
            prediction["event_time_nll"] = event_time_nll(rate, part.exposure_minutes.to_numpy(), part.event.to_numpy())
            prediction_frames.append(prediction)
            groups = {
                "predicted_rate_bin": pd.cut(rate, bins=bin_edges[name], include_lowest=True).astype(str),
                "active_template": template_keys(part),
                "date": part.start_time.dt.strftime("%Y-%m-%d"),
                "price_shift": pd.cut(part.delta_l1, [-np.inf, 0, .05, .1, .2, .3, .5, np.inf], include_lowest=True).astype(str),
            }
            if train.config_age_minutes.notna().any():
                groups["config_age"] = pd.cut(part.config_age_minutes, [-np.inf, 0, 30, 60, 120, np.inf], include_lowest=True).astype(str)
            if train.inventory_par_start.notna().any():
                groups["inventory"] = pd.cut(part.inventory_par_start, [0, 25000, 100000, 250000, 1000000, np.inf], include_lowest=True).astype(str)
            if quantity_edges is not None:
                groups["quote_quantity"] = pd.cut(part.quote_quantity, quantity_edges, include_lowest=True).astype(str)
            if "liquidity_score" in train and train.liquidity_score.notna().any():
                groups["liquidity_missing"] = part.liquidity_score.isna().map({True: "missing", False: "observed"})
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
    metadata = {
        "config": asdict(config), "input_rows": len(df), "eligible_rows": len(clean),
        "eligible_events": int(clean.event.sum()), "selected_model_name": selected,
        "selection_rule": "minimum validation event-time NLL; test never used for selection",
        "selected_alpha": {name: model.alpha for name, model in models.items() if isinstance(model, ExponentialFillModel)},
        "price_coefficient_per_price_point": price_model.price_coefficient_per_price_point,
        "price_coefficient_nonpositive": bool(price_model.price_coefficient_per_price_point <= 0),
        "training_delta_range": [float(train.delta_l1.min()), float(train.delta_l1.max())],
        "preprocessing_fit_scope": "training data only", "refit_on_train_plus_validation": False,
        "rate_unit": "events per active market minute", "timestamp_convention": "naive New York local clock; no timezone conversion",
        "source_schema": asdict(schema),
        "quantity_note": "quantity is offered size in source units times quantity_multiplier; not position inventory",
        "maturity_note": "time_to_maturity converted to years; distinct from duration_years",
        "label_note": "nonempty first_fill_level is an event; quote_end_time and exposure stop at first fill",
        "generated_ids_note": "without supplied IDs, interval key is CUSIP/start; fill key is CUSIP/end; config ID is a row placeholder",
        "censor_note": "CENSORED means observed no target fill until the endpoint; specific exit mechanism unavailable",
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

    Recomputes all offer prices consistently. This does not model price-change
    effects on future state/configuration age and is not a pricing recommendation.
    """
    if not deltas or not np.isfinite(deltas).all():
        raise ValueError("Supply a nonempty finite price grid.")
    states = normalize_quote_states(states, getattr(getattr(model, "design", None), "schema", None))
    frames = []
    for delta in sorted(set(deltas)):
        candidate = states.copy()
        candidate["delta_l1"] = delta
        candidate["l1_price"] = candidate.cep_mid + delta
        for level in [2, 3]:
            candidate[f"l{level}_price"] = (candidate.l1_price + candidate[f"gap_l{level}"]).where(candidate[f"l{level}_active"].eq(1))
        rate = model.predict_rate(candidate)
        frames.append(pd.DataFrame({"state_index": np.arange(len(states)), "delta_l1": delta,
                                    "rate_per_minute": rate, "p_fill": probability_from_rate(rate, horizon_minutes)}))
    return pd.concat(frames, ignore_index=True)
