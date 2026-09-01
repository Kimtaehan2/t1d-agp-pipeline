"""단위 변환 · 리샘플링 · 결측 처리 · 캡핑 플래그 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    GRID_MINUTES,
    MGDL,
    MMOL,
    MMOL_TO_MGDL,
    SENSOR_LIMITS,
    UnitInferenceError,
    flag_capping,
    infer_glucose_unit,
    interpolate_short_gaps,
    preprocess_cgm,
    resample_patient,
    sensor_limits,
    snap_to_grid,
    summarize,
    to_mgdl,
    window_coverage,
)
from src.schema import (
    coerce_cgm_frame,
    coerce_processed_cgm_frame,
    validate_processed_cgm_frame,
)

NAN = float("nan")


# ============================================================================
# 단위
# ============================================================================


def test_infer_mmol():
    assert infer_glucose_unit(pd.Series([3.9, 7.2, 10.5, 14.0])) == MMOL


def test_infer_mgdl():
    assert infer_glucose_unit(pd.Series([70.0, 140.0, 180.0, 250.0])) == MGDL


def test_infer_ignores_nan():
    assert infer_glucose_unit(pd.Series([NAN, 6.0, 8.0])) == MMOL


def test_infer_on_empty_raises():
    with pytest.raises(UnitInferenceError):
        infer_glucose_unit(pd.Series([], dtype="float64"))


def test_infer_on_all_nan_raises():
    with pytest.raises(UnitInferenceError):
        infer_glucose_unit(pd.Series([NAN, NAN]))


def test_to_mgdl_converts_mmol():
    out = to_mgdl(pd.Series([5.0, 10.0]), MMOL)
    assert out.tolist() == pytest.approx([5.0 * MMOL_TO_MGDL, 10.0 * MMOL_TO_MGDL])


def test_to_mgdl_passes_through_mgdl():
    assert to_mgdl(pd.Series([90.0, 180.0]), MGDL).tolist() == [90.0, 180.0]


def test_to_mgdl_autodetects():
    assert to_mgdl(pd.Series([5.0, 10.0])).tolist() == pytest.approx(
        [5.0 * MMOL_TO_MGDL, 10.0 * MMOL_TO_MGDL]
    )
    assert to_mgdl(pd.Series([90.0, 180.0])).tolist() == [90.0, 180.0]


def test_to_mgdl_rejects_unknown_unit():
    with pytest.raises(ValueError, match="알 수 없는 단위"):
        to_mgdl(pd.Series([100.0]), "mg/L")


def test_to_mgdl_preserves_nan():
    assert to_mgdl(pd.Series([NAN, 6.0]), MMOL).isna().tolist() == [True, False]


# ============================================================================
# 센서 측정범위
# ============================================================================


def test_sensor_limits_are_per_dataset():
    assert sensor_limits("replace_bg") == (39.0, 401.0)
    assert sensor_limits("shanghai_t1dm") == (39.6, 500.4)
    assert sensor_limits("simulated") == (40.0, 400.0)


def test_sensor_limits_rejects_unregistered_dataset():
    """추측해서 40/400을 쓰느니 실패하는 편이 낫다."""
    with pytest.raises(KeyError, match="등록돼 있지 않다"):
        sensor_limits("ohio")


def test_replace_bg_limits_are_not_the_spec_defaults():
    assert SENSOR_LIMITS["replace_bg"] != (40.0, 400.0)


# ============================================================================
# 그리드 스냅 / 리샘플링
# ============================================================================


def test_snap_rounds_to_nearest_grid_point():
    ts = pd.Series(pd.to_datetime(["2021-07-30 16:43:00", "2021-07-30 16:58:00",
                                   "2021-07-30 17:13:00"]))
    assert snap_to_grid(ts).dt.strftime("%H:%M").tolist() == ["16:45", "17:00", "17:15"]


def test_snap_absorbs_device_clock_jitter():
    """Replace-BG는 300초 간격에 ±2초씩 흔들린다."""
    ts = pd.Series(pd.to_datetime(["2015-01-01 00:00:36", "2015-01-01 00:05:35",
                                   "2015-01-01 00:10:37"]))
    assert snap_to_grid(ts).dt.strftime("%H:%M").tolist() == ["00:00", "00:05", "00:10"]


def test_resample_fills_grid_between_15_minute_readings():
    """Shanghai는 15분 간격이라 5분 격자에서 사이사이가 빈다."""
    ts = pd.Series(pd.to_datetime(["2021-01-01 00:00", "2021-01-01 00:15"]))
    grid, values, _ = resample_patient(ts, pd.Series([100.0, 130.0]))

    assert len(grid) == 4                      # 00:00, 00:05, 00:10, 00:15
    assert values[0] == 100.0 and values[3] == 130.0
    assert np.isnan(values[1]) and np.isnan(values[2])


def test_resample_averages_collisions():
    """두 관측이 같은 격자점으로 반올림되면 평균을 쓰고 개수를 보고한다."""
    ts = pd.Series(pd.to_datetime(["2015-01-01 00:01:00", "2015-01-01 00:01:30"]))
    grid, values, collisions = resample_patient(ts, pd.Series([100.0, 110.0]))

    assert len(grid) == 1
    assert values[0] == 105.0
    assert collisions == 1


def test_resample_starts_and_ends_at_observations():
    ts = pd.Series(pd.to_datetime(["2015-01-01 03:00", "2015-01-01 03:10"]))
    grid, _, _ = resample_patient(ts, pd.Series([100.0, 120.0]))
    assert grid[0] == pd.Timestamp("2015-01-01 03:00")
    assert grid[-1] == pd.Timestamp("2015-01-01 03:10")


# ============================================================================
# 결측 처리 — 30분 규칙
# ============================================================================


def test_gap_of_exactly_30_minutes_is_interpolated():
    """빈 슬롯 5개 = 양 끝 관측 사이 30분. 경계는 포함이다."""
    values = np.array([100.0] + [NAN] * 5 + [160.0])
    out, imputed = interpolate_short_gaps(values)

    assert not np.isnan(out).any()
    assert imputed.tolist() == [False] + [True] * 5 + [False]
    assert out[1:6].tolist() == pytest.approx([110.0, 120.0, 130.0, 140.0, 150.0])


def test_gap_over_30_minutes_is_never_interpolated():
    """빈 슬롯 6개 = 35분. 절대 채우지 않는다."""
    values = np.array([100.0] + [NAN] * 6 + [170.0])
    out, imputed = interpolate_short_gaps(values)

    assert np.isnan(out[1:7]).all()
    assert not imputed.any()


def test_short_and_long_gaps_in_one_series():
    values = np.array([100.0, NAN, NAN, 130.0] + [NAN] * 8 + [200.0])
    out, imputed = interpolate_short_gaps(values)

    assert imputed[1:3].all()               # 15분 공백 → 보간
    assert not imputed[4:12].any()          # 45분 공백 → 그대로
    assert np.isnan(out[4:12]).all()


def test_leading_and_trailing_gaps_are_not_extrapolated():
    values = np.array([NAN, NAN, 100.0, 110.0, NAN])
    out, imputed = interpolate_short_gaps(values)

    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[4])
    assert not imputed.any()


def test_series_without_gaps_is_untouched():
    values = np.array([100.0, 110.0, 120.0])
    out, imputed = interpolate_short_gaps(values)

    assert out.tolist() == values.tolist()
    assert not imputed.any()


def test_gap_rule_scales_with_grid_size():
    """15분 격자라면 빈 슬롯 1개(=30분 공백)까지만 보간된다."""
    values = np.array([100.0, NAN, 130.0])
    _, imputed = interpolate_short_gaps(values, grid_minutes=15)
    assert imputed[1]

    values = np.array([100.0, NAN, NAN, 130.0])
    _, imputed = interpolate_short_gaps(values, grid_minutes=15)
    assert not imputed.any()


# ============================================================================
# 캡핑 플래그
# ============================================================================


def test_consecutive_low_sentinels_are_flagged():
    values = np.array([80.0, 39.0, 39.0, 39.0, 90.0])
    assert flag_capping(values, 39.0, 401.0).tolist() == [
        False, True, True, True, False
    ]


def test_isolated_sentinel_is_not_flagged_by_default():
    """스펙이 말하는 것은 '연속으로 반복되는 구간'이다."""
    values = np.array([80.0, 39.0, 90.0])
    assert not flag_capping(values, 39.0, 401.0).any()


def test_min_run_one_flags_every_censored_point():
    values = np.array([80.0, 39.0, 90.0])
    assert flag_capping(values, 39.0, 401.0, min_run=1).tolist() == [
        False, True, False
    ]


def test_high_sentinel_is_flagged_too():
    values = np.array([300.0, 401.0, 401.0, 350.0])
    assert flag_capping(values, 39.0, 401.0).tolist() == [False, True, True, False]


def test_capping_does_not_modify_values():
    values = np.array([39.0, 39.0, 100.0])
    before = values.copy()
    flag_capping(values, 39.0, 401.0)
    assert values.tolist() == before.tolist()


def test_capping_ignores_nan():
    values = np.array([39.0, NAN, 39.0])
    assert not flag_capping(values, 39.0, 401.0).any()


def test_capping_uses_dataset_limits_not_40_400():
    """Replace-BG에서 40/400을 기준으로 잡으면 센티널을 통째로 놓친다."""
    values = np.array([39.0, 39.0, 401.0, 401.0])
    assert flag_capping(values, 39.0, 401.0).all()
    assert not flag_capping(values, 40.0, 400.0).any()


def test_shanghai_limits_are_fractional():
    values = np.array([39.6, 39.6, 120.0])
    low, high = sensor_limits("shanghai_t1dm")
    assert flag_capping(values, low, high).tolist() == [True, True, False]


# ============================================================================
# 파이프라인
# ============================================================================


def _cgm(source: str, times: list[str], values: list[float], pid: str = "p1"):
    return coerce_cgm_frame(
        pd.DataFrame(
            {
                "patient_id": f"{source}_{pid}",
                "timestamp": pd.to_datetime(times),
                "glucose_mgdl": values,
                "source": source,
            }
        )
    )


def test_pipeline_output_satisfies_processed_schema():
    df = _cgm("replace_bg", ["2015-01-01 00:00:36", "2015-01-01 00:05:35",
                             "2015-01-01 00:10:37"], [100.0, 110.0, 120.0])
    out = preprocess_cgm(df)

    validate_processed_cgm_frame(out)
    assert out["timestamp"].dt.second.eq(0).all()
    assert out["timestamp"].dt.minute.mod(GRID_MINUTES).eq(0).all()


def test_pipeline_interpolates_shanghai_15_minute_data():
    df = _cgm("shanghai_t1dm", ["2021-01-01 00:00", "2021-01-01 00:15"],
              [100.0, 130.0])
    out = preprocess_cgm(df)

    assert len(out) == 4
    assert out["glucose_mgdl"].tolist() == pytest.approx([100.0, 110.0, 120.0, 130.0])
    assert out["is_imputed"].tolist() == [False, True, True, False]


def test_pipeline_leaves_long_gaps_as_nan():
    df = _cgm("replace_bg", ["2015-01-01 00:00", "2015-01-01 01:00"], [100.0, 200.0])
    out = preprocess_cgm(df)

    assert len(out) == 13                       # 5분 격자로 1시간
    assert out["glucose_mgdl"].isna().sum() == 11
    assert not out["is_imputed"].any()
    validate_processed_cgm_frame(out)


def test_pipeline_flags_capping_using_source():
    df = _cgm("replace_bg", ["2015-01-01 00:00", "2015-01-01 00:05",
                             "2015-01-01 00:10"], [39.0, 39.0, 80.0])
    out = preprocess_cgm(df)
    assert out["is_capped"].tolist() == [True, True, False]


def test_pipeline_handles_multiple_patients_and_sources():
    a = _cgm("replace_bg", ["2015-01-01 00:00", "2015-01-01 00:05"],
             [39.0, 39.0], pid="1")
    b = _cgm("shanghai_t1dm", ["2021-01-01 00:00", "2021-01-01 00:05"],
             [39.6, 39.6], pid="2")
    out = preprocess_cgm(pd.concat([a, b], ignore_index=True))

    validate_processed_cgm_frame(out)
    assert out["is_capped"].all()               # 각자의 센티널로 판정됐다
    assert out["patient_id"].nunique() == 2


def test_pipeline_rejects_unregistered_source():
    df = _cgm("ohio", ["2015-01-01 00:00", "2015-01-01 00:05"], [100.0, 110.0])
    with pytest.raises(KeyError, match="등록돼 있지 않다"):
        preprocess_cgm(df)


def test_pipeline_accepts_explicit_limits():
    df = _cgm("ohio", ["2015-01-01 00:00", "2015-01-01 00:05"], [40.0, 40.0])
    out = preprocess_cgm(df, limits=(40.0, 400.0))
    assert out["is_capped"].all()


def test_pipeline_on_empty_frame():
    from src.schema import empty_cgm_frame

    out = preprocess_cgm(empty_cgm_frame())
    assert out.empty
    validate_processed_cgm_frame(out)


def test_preprocess_parquet_matches_in_memory_result(tmp_path):
    """Parquet 스트리밍 경로와 메모리 경로의 결과가 같아야 한다."""
    from src.preprocess import preprocess_parquet
    from src.storage import read_parquet, write_parquet

    frames = [
        _cgm("replace_bg", ["2015-01-01 00:00", "2015-01-01 00:05",
                            "2015-01-01 00:20"], [100.0, 39.0, 39.0], pid="10"),
        _cgm("replace_bg", ["2015-01-01 00:00", "2015-01-01 00:05"],
             [150.0, 160.0], pid="2"),
        _cgm("replace_bg", ["2015-01-01 00:00", "2015-01-01 02:00"],
             [120.0, 130.0], pid="100"),
    ]
    source = pd.concat(frames, ignore_index=True)

    in_path = tmp_path / "in.parquet"
    out_path = tmp_path / "out.parquet"
    write_parquet(source, in_path)

    stats = preprocess_parquet(in_path, out_path, patients_per_batch=2)
    actual = coerce_processed_cgm_frame(read_parquet(out_path))
    expected = preprocess_cgm(source)

    assert stats["n_patients"] == 3
    assert stats["n_slots"] == len(expected)
    pd.testing.assert_frame_equal(
        actual.sort_values(["patient_id", "timestamp"]).reset_index(drop=True),
        expected,
    )
    validate_processed_cgm_frame(actual)


# ============================================================================
# AGP 창 유효성
# ============================================================================


def _dense_patient(days: float, coverage: float, pid: str = "replace_bg_1"):
    """``days``일치 5분 격자에서 앞쪽 ``coverage`` 비율만 값이 있는 프레임."""
    n = int(days * 24 * 60 / GRID_MINUTES) + 1
    values = np.full(n, 120.0)
    values[int(n * coverage):] = NAN
    return pd.DataFrame(
        {
            "patient_id": pid,
            "timestamp": pd.date_range("2015-01-01", periods=n, freq="5min"),
            "glucose_mgdl": values,
            "source": "replace_bg",
            "is_imputed": False,
            "is_capped": False,
        }
    )


def test_window_marks_long_dense_window_valid():
    cov = window_coverage(_dense_patient(days=14, coverage=1.0))
    row = cov.iloc[0]
    assert row["is_valid_agp"]
    assert row["meets_recommended_days"]
    assert row["coverage"] == pytest.approx(1.0)


def test_window_rejects_too_few_days():
    cov = window_coverage(_dense_patient(days=9, coverage=1.0))
    assert not cov.iloc[0]["is_valid_agp"]


def test_window_rejects_low_coverage():
    cov = window_coverage(_dense_patient(days=14, coverage=0.5))
    row = cov.iloc[0]
    assert row["coverage"] == pytest.approx(0.5, abs=0.01)
    assert not row["is_valid_agp"]


def test_window_10_days_is_valid_but_not_recommended():
    cov = window_coverage(_dense_patient(days=10, coverage=1.0))
    row = cov.iloc[0]
    assert row["is_valid_agp"]
    assert not row["meets_recommended_days"]


def test_window_can_exclude_imputed_from_coverage():
    df = _dense_patient(days=14, coverage=1.0)
    df.loc[df.index[: len(df) // 2], "is_imputed"] = True

    with_imputed = window_coverage(df).iloc[0]["coverage"]
    without = window_coverage(df, include_imputed=False).iloc[0]["coverage"]

    assert with_imputed == pytest.approx(1.0)
    assert without == pytest.approx(0.5, abs=0.01)


def test_summarize_reports_flag_rates():
    df = _cgm("replace_bg", ["2015-01-01 00:00", "2015-01-01 00:15"], [100.0, 130.0])
    stats = summarize(preprocess_cgm(df))

    assert stats["n_slots"] == 4
    assert stats["n_imputed"] == 2
    assert stats["pct_imputed"] == pytest.approx(50.0)
    assert stats["n_patients"] == 1
