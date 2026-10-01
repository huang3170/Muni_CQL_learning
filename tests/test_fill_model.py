from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import warnings

import joblib
import numpy as np
import pandas as pd

from fill_model import TrainingConfig, run_experiment, score_price_grid, state_features
from fill_model.core import DesignMatrix, chronological_split, price_offsets


def native_quotes() -> pd.DataFrame:
    """Small chronological sample using only the supplied source column names."""
    rows = []
    labels = [1, 2, 3, None, "", "  ", np.nan, pd.NA]
    for day in range(10):
        for slot in range(8):
            start = pd.Timestamp("2026-01-05 09:30") + pd.Timedelta(days=day, minutes=15 * slot)
            mid = 100.0 + 0.02 * day
            offset = 0.02 + 0.02 * slot
            exposure = 7.0 + slot % 3
            active3 = slot % 2
            rows.append({
                "cusip": f"CUSIP{slot % 2}",
                "quantity": 10000.0 * (1 + slot % 3),
                "l1_price": mid + offset,
                "l2_price": mid + offset + 0.05,
                "l3_price": mid + offset + 0.10 if active3 else np.nan,
                "l1_active": 1, "l2_active": 1, "l3_active": active3,
                "cycle_time": start, "exposure_minutes": exposure,
                "episode_id": f"episode{slot % 2}",
                "quote_end_time": start + pd.Timedelta(minutes=exposure),
                "first_fill_level": labels[slot],
                "first_fill_quantity": 1000.0 if slot < 3 else np.nan,
                "cep_time": start - pd.Timedelta(minutes=2),
                "bid_price": mid - 0.1, "ask_price": mid + 0.1,
                "mid_price": mid, "cep_age_min": 2.0,
                "cep_bid_ask_width": 0.2,
                "l1_vs_mid": offset, "l2_vs_mid": offset + 0.05,
                "l3_vs_mid": offset + 0.10 if active3 else np.nan,
                "time_to_maturity": 1000.0 + 30 * slot,
                "rating": ["AA", "A", "BBB"][slot % 3],
                "liquidity": np.nan if slot % 4 == 0 else 1.0 + slot / 10,
                "coupon": 4.0 + slot % 2,
            })
    frame = pd.DataFrame(rows)
    frame.loc[6, "quantity"] = np.nan
    frame.loc[16, "mid_price"] = np.nan
    frame.loc[17, "rating"] = None
    frame.loc[[5, 33], ["l1_price", "l1_vs_mid"]] = np.nan
    return frame


def observed_events(frame: pd.DataFrame) -> pd.Series:
    labels = frame.first_fill_level
    return (labels.notna() & labels.astype("string").str.strip().ne("").fillna(False)).astype(int)


class NativeFeaturesTest(unittest.TestCase):
    def test_explicit_features_do_not_leak_identifiers_or_outcomes(self) -> None:
        source = native_quotes()
        features = state_features(source)
        expected = {
            "log_quantity", "gap_l2", "gap_l3", "mid_price", "time_to_maturity",
            "coupon", "liquidity", "cep_bid_ask_width", "time_sin", "time_cos",
            "log_cep_age", "active_template", "rating",
        }
        self.assertEqual(set(features), expected)
        self.assertAlmostEqual(features.loc[0, "log_quantity"], np.log1p(10000.0))
        self.assertAlmostEqual(features.loc[0, "time_sin"], np.sin(2 * np.pi * 570 / 1440))
        self.assertAlmostEqual(features.loc[0, "log_cep_age"], np.log1p(2.0))
        self.assertEqual(features.loc[0, "time_to_maturity"], 1000.0)

        changed = source.copy()
        changed["cusip"], changed["episode_id"] = "other", "other"
        changed["first_fill_level"], changed["first_fill_quantity"] = 3, 900000.0
        changed["exposure_minutes"] = 999.0
        changed["quote_end_time"] += pd.Timedelta(days=100)
        changed["event"] = 1
        pd.testing.assert_frame_equal(features, state_features(changed))

        # Shifting all offers together belongs only in the explicit price term.
        shifted = source.copy()
        for level in (1, 2, 3):
            shifted[f"l{level}_vs_mid"] += 0.2
            shifted[f"l{level}_price"] += 0.2
        pd.testing.assert_frame_equal(features, state_features(shifted), atol=1e-12)

    def test_supplied_offsets_take_priority_with_missing_fallback_and_inactive_mask(self) -> None:
        source = native_quotes().iloc[:3].copy()
        source.loc[0, ["l1_price", "l2_price", "l3_price"]] = [999.0, 999.0, 999.0]
        source.loc[0, "l3_vs_mid"] = 99.0  # Stale value at an inactive level.
        source.loc[1, "l2_vs_mid"] = np.nan
        offsets = price_offsets(source)
        self.assertEqual(offsets.loc[0, "l1_vs_mid"], source.loc[0, "l1_vs_mid"])
        self.assertEqual(offsets.loc[0, "l2_vs_mid"], source.loc[0, "l2_vs_mid"])
        self.assertTrue(np.isnan(offsets.loc[0, "l3_vs_mid"]))
        self.assertAlmostEqual(offsets.loc[1, "l2_vs_mid"], source.loc[1, "l2_price"] - source.loc[1, "mid_price"])
        self.assertAlmostEqual(state_features(source).loc[0, "gap_l2"], 0.05)

    def test_imputation_uses_training_statistics_and_keeps_liquidity_missingness(self) -> None:
        train = native_quotes().iloc[:8].copy()
        train["liquidity"] = [1.0, 3.0, np.nan, 5.0, 1.0, 3.0, np.nan, 5.0]
        design = DesignMatrix(with_price=True, price_scale=0.1).fit(train)
        liquidity_index = design.numeric.index("liquidity")
        self.assertEqual(design.imputer.statistics_[liquidity_index], 3.0)
        self.assertIn("missingindicator_liquidity", design.feature_names)
        self.assertIn("missingindicator_l1_vs_mid", design.feature_names)

        held_out = train.iloc[:2].copy()
        held_out["liquidity"] = [1000.0, np.nan]
        held_out.loc[held_out.index[1], ["l1_vs_mid", "l1_price"]] = np.nan
        stats = design.imputer.statistics_.copy()
        matrix = design.transform(held_out).toarray()
        np.testing.assert_array_equal(design.imputer.statistics_, stats)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertAlmostEqual(matrix[1, -1], price_offsets(train).l1_vs_mid.median() / 0.1)

    def test_split_uses_original_midnight_and_purges_only_crossing_rows(self) -> None:
        source = native_quotes()
        source["event"] = observed_events(source)
        source["_episode_group"] = list(zip(source.cusip, source.episode_id))
        boundary_rows = source.iloc[:3].copy()
        cut = pd.Timestamp("2026-01-11")
        boundary_rows["cycle_time"] = [cut - pd.Timedelta(minutes=7), cut - pd.Timedelta(minutes=5), cut]
        boundary_rows["quote_end_time"] = [cut, cut + pd.Timedelta(minutes=5), cut + pd.Timedelta(minutes=7)]
        boundary_rows["case"] = ["ends_at_cut", "crosses_cut", "starts_at_cut"]
        source = pd.concat([source, boundary_rows], ignore_index=True)
        parts, audit, details = chronological_split(
            source, TrainingConfig(validation_start="2026-01-11", test_start="2026-01-13")
        )
        self.assertIn("ends_at_cut", parts["train"].case.tolist())
        self.assertIn("starts_at_cut", parts["validation"].case.tolist())
        for part in parts.values():
            self.assertNotIn("crosses_cut", part.case.tolist())
            self.assertIsNone(part.cycle_time.dt.tz)
        self.assertEqual(int(audit.loc[audit.split.eq("purged_boundary"), "rows"].sum()), 1)
        self.assertGreater(details["episodes_in_multiple_splits"], 0)


class NativeExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = native_quotes()
        # Existing labels must never override the requested first-fill rule.
        cls.source["event"] = 1 - observed_events(cls.source)
        cls.before = cls.source.copy(deep=True)
        cls.config = TrainingConfig(alphas=(0.01,), bootstrap_repetitions=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cls.result = run_experiment(cls.source, config=cls.config)

    def test_native_training_preserves_rows_times_and_authoritative_labels(self) -> None:
        pd.testing.assert_frame_equal(self.source, self.before)
        self.assertEqual(self.result.metadata["eligible_rows"], len(self.source))
        self.assertEqual(self.result.metadata["eligible_events"], int(observed_events(self.source).sum()))
        self.assertTrue(self.result.exclusions.empty)
        expected = self.source.assign(expected_event=observed_events(self.source))
        predictions = self.result.predictions.merge(
            expected[["cusip", "cycle_time", "quote_end_time", "expected_event"]],
            on=["cusip", "cycle_time", "quote_end_time"], validate="many_to_one",
        )
        self.assertEqual(len(predictions), len(self.result.predictions))
        np.testing.assert_array_equal(predictions.event, predictions.expected_event)
        self.assertIsNone(predictions.cycle_time.dt.tz)
        self.assertIsNone(predictions.quote_end_time.dt.tz)
        liquidity = self.result.feature_missingness.query("feature == 'liquidity'")
        self.assertTrue(liquidity.missing_fraction.eq(0.25).all())

    def test_unlabeled_prediction_serialization_and_price_monotonicity(self) -> None:
        states = self.source.tail(8).drop(columns=[
            "cusip", "episode_id", "event", "first_fill_level", "first_fill_quantity",
            "exposure_minutes", "quote_end_time", "cep_time", "bid_price", "ask_price",
        ])
        model = self.result.models["price"]
        rates = model.predict_rate(states)
        self.assertTrue(np.isfinite(rates).all())
        self.assertTrue((rates > 0).all())
        fallback = states.copy()
        fallback[[f"l{level}_vs_mid" for level in (1, 2, 3)]] = np.nan
        # Equivalent offset subtraction must not amplify roundoff in constant gaps.
        np.testing.assert_allclose(model.predict_rate(fallback), rates, rtol=1e-10)
        before = states.copy(deep=True)
        grid = score_price_grid(model, states, [0.3, -0.1, 0.1])
        pd.testing.assert_frame_equal(states, before)
        for _, group in grid.groupby("state_index"):
            self.assertTrue(np.all(np.diff(group.sort_values("l1_vs_mid").rate_per_minute) <= 1e-12))
        self.assertLessEqual(model.price_coefficient_per_price_point, 0.0)

        with tempfile.TemporaryDirectory() as directory:
            self.result.save(directory)
            saved = joblib.load(Path(directory) / "models.joblib")
            np.testing.assert_allclose(saved["models"]["price"].predict_rate(states), rates)
            self.assertTrue((Path(directory) / "metadata.json").is_file())

    def test_invalid_observation_windows_still_fail_validation(self) -> None:
        for case in ("zero_exposure", "exposure_exceeds_window", "reversed_window", "missing_start", "string_timestamps"):
            with self.subTest(case=case):
                bad = self.source.copy()
                if case == "zero_exposure":
                    bad.loc[0, "exposure_minutes"] = 0.0
                elif case == "exposure_exceeds_window":
                    bad.loc[0, "exposure_minutes"] = 1000.0
                elif case == "reversed_window":
                    bad.loc[0, "quote_end_time"] = bad.loc[0, "cycle_time"] - pd.Timedelta(minutes=1)
                elif case == "missing_start":
                    bad.loc[0, "cycle_time"] = pd.NaT
                else:
                    bad["cycle_time"] = bad.cycle_time.astype(str)
                with self.assertRaises(ValueError):
                    run_experiment(bad, config=self.config)


if __name__ == "__main__":
    unittest.main()
