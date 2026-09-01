"""AGP 지표 계산 테스트.

지표는 손으로 검산할 수 있는 값으로 확인한다. 정의가 틀리면 이후 단계가 전부
틀리므로 계산식을 직접 고정한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    DEFAULT_WINDOW_DAYS,
    GMI_INTERCEPT,
    GMI_SLOPE,
    METRIC_COLUMNS,
    TARGETS,
    evaluate_targets,
    glucose_metrics,
    metrics_for_window,
    recent_metrics,
    sliding_metrics,
)
from src.preprocess import GRID_MINUTES
from src.schema import SchemaError, coerce_processed_cgm_frame

NAN = float("nan")
SLOTS_PER_DAY = 24 * 60 // GRID_MINUTES


# ============================================================================
# 순수 수치 계산
# ============================================================================


def test_tir_counts_range_inclusively():
    """70과 180은 범위 안이다."""
    m = glucose_metrics(np.array([70.0, 180.0, 69.9, 180.1]))
    assert m["tir"] == pytest.approx(50.0)
    assert m["tbr"] == pytest.approx(25.0)
    assert m["tar"] == pytest.approx(25.0)


def test_tir_tbr_tar_sum_to_100():
    values = np.array([50.0, 60.0, 100.0, 150.0, 200.0, 300.0])
    m = glucose_metrics(values)
    assert m["tir"] + m["tbr"] + m["tar"] == pytest.approx(100.0)


def test_tbr_level2_is_subset_of_tbr():
    m = glucose_metrics(np.array([40.0, 60.0, 100.0, 100.0]))
    assert m["tbr"] == pytest.approx(50.0)       # 40, 60 둘 다 70 미만
    assert m["tbr_level2"] == pytest.approx(25.0)  # 40만 54 미만


def test_gmi_uses_the_spec_formula():
    values = np.full(10, 150.0)
    m = glucose_metrics(values)
    assert m["gmi"] == pytest.approx(GMI_INTERCEPT + GMI_SLOPE * 150.0)
    assert m["gmi"] == pytest.approx(6.898)


def test_cv_is_sample_sd_over_mean():
    values = np.array([100.0, 120.0, 140.0])
    m = glucose_metrics(values)
    expected = 100.0 * np.std(values, ddof=1) / np.mean(values)
    assert m["cv"] == pytest.approx(expected)
    assert m["sd_glucose"] == pytest.approx(np.std(values, ddof=1))


def test_nan_values_are_ignored():
    a = glucose_metrics(np.array([100.0, 150.0]))
    b = glucose_metrics(np.array([100.0, NAN, 150.0, NAN]))
    assert a == pytest.approx(b)


def test_all_nan_yields_nan_metrics():
    m = glucose_metrics(np.array([NAN, NAN]))
    assert all(np.isnan(v) for v in m.values())


def test_empty_yields_nan_metrics():
    m = glucose_metrics(np.array([]))
    assert np.isnan(m["tir"])


def test_single_value_has_no_sd():
    m = glucose_metrics(np.array([120.0]))
    assert np.isnan(m["sd_glucose"]) and np.isnan(m["cv"])
    assert m["mean_glucose"] == 120.0


def test_capped_values_are_counted_not_dropped():
    """39는 실제 저혈당, 401은 실제 고혈당이다. 빼면 TIR이 부풀려진다."""
    m = glucose_metrics(np.array([39.0, 401.0, 100.0, 100.0]))
    assert m["tir"] == pytest.approx(50.0)
    assert m["tbr_level2"] == pytest.approx(25.0)
    assert m["tar"] == pytest.approx(25.0)


def test_tar_level2_is_subset_of_tar():
    m = glucose_metrics(np.array([200.0, 300.0, 100.0, 100.0]))
    assert m["tar"] == pytest.approx(50.0)         # 200, 300 둘 다 180 초과
    assert m["tar_level2"] == pytest.approx(25.0)  # 300만 250 초과


# ============================================================================
# 목표 판정
# ============================================================================


def test_targets_match_the_spec_table():
    assert TARGETS["tir"] == (">=", 70.0)
    assert TARGETS["tar"] == ("<=", 25.0)
    assert TARGETS["tar_level2"] == ("<=", 5.0)   # 스펙 표에는 없고 팀 합의로 추가
    assert TARGETS["tbr"] == ("<=", 4.0)
    assert TARGETS["tbr_level2"] == ("<=", 1.0)
    assert TARGETS["cv"] == ("<=", 36.0)


def test_evaluate_targets_on_dict():
    ok = evaluate_targets({"tir": 72.0, "tar": 24.0, "tar_level2": 3.0,
                           "tbr": 3.0, "tbr_level2": 0.5, "cv": 35.0})
    assert all(ok.values())


def test_evaluate_targets_flags_failures():
    ok = evaluate_targets({"tir": 65.0, "tar": 30.0, "tar_level2": 8.0,
                           "tbr": 5.0, "tbr_level2": 2.0, "cv": 40.0})
    assert not any(ok.values())


def test_target_boundaries_are_inclusive():
    ok = evaluate_targets({"tir": 70.0, "tar": 25.0, "tar_level2": 5.0,
                           "tbr": 4.0, "tbr_level2": 1.0, "cv": 36.0})
    assert all(ok.values())


def test_evaluate_targets_on_frame():
    df = pd.DataFrame({"tir": [80.0, 60.0], "cv": [30.0, 40.0]})
    out = evaluate_targets(df)
    assert out["tir_ok"].tolist() == [True, False]
    assert out["cv_ok"].tolist() == [True, False]


def test_evaluate_targets_keeps_nan_as_na():
    ok = evaluate_targets({"tir": NAN})
    assert ok["tir_ok"] is pd.NA


# ============================================================================
# 창 계산
# ============================================================================


def _series(n_slots: int, values, start="2015-01-01"):
    ts = pd.date_range(start, periods=n_slots, freq=f"{GRID_MINUTES}min")
    return ts.to_numpy(), np.asarray(values, dtype="float64")


def test_coverage_denominator_comes_from_window_length_not_row_count():
    """창 끝이 데이터 범위 밖이어도 그만큼 결측으로 잡혀야 한다."""
    ts, gl = _series(SLOTS_PER_DAY, np.full(SLOTS_PER_DAY, 120.0))  # 1일치만 있음
    row = metrics_for_window(
        ts, gl, pd.Timestamp("2015-01-01"), pd.Timestamp("2015-01-03")  # 2일 창
    )
    assert row["n_expected"] == 2 * SLOTS_PER_DAY
    assert row["n_present"] == SLOTS_PER_DAY
    assert row["coverage"] == pytest.approx(0.5)


def test_window_is_half_open():
    ts, gl = _series(3, [100.0, 200.0, 300.0])
    row = metrics_for_window(
        ts, gl,
        pd.Timestamp("2015-01-01 00:00"),
        pd.Timestamp("2015-01-01 00:10"),   # 00:10 슬롯은 제외
        min_days=0,
    )
    assert row["n_present"] == 2
    assert row["mean_glucose"] == pytest.approx(150.0)


def test_window_invalid_when_too_short():
    ts, gl = _series(SLOTS_PER_DAY, np.full(SLOTS_PER_DAY, 120.0))
    row = metrics_for_window(
        ts, gl, pd.Timestamp("2015-01-01"), pd.Timestamp("2015-01-02")
    )
    assert not row["is_valid"]
    assert "관찰기간" in row["invalid_reason"]


def test_window_invalid_when_coverage_too_low():
    n = 14 * SLOTS_PER_DAY
    values = np.full(n, 120.0)
    values[round(n * 0.5):] = NAN
    ts, gl = _series(n, values)
    row = metrics_for_window(
        ts, gl, pd.Timestamp("2015-01-01"), pd.Timestamp("2015-01-15")
    )
    assert not row["is_valid"]
    assert "확보율" in row["invalid_reason"]


def test_window_valid_at_exactly_the_thresholds():
    n = 10 * SLOTS_PER_DAY
    values = np.full(n, 120.0)
    values[round(n * 0.70):] = NAN      # 정확히 70% 확보
    ts, gl = _series(n, values)
    row = metrics_for_window(
        ts, gl, pd.Timestamp("2015-01-01"), pd.Timestamp("2015-01-11")
    )
    assert row["days"] == 10.0
    assert row["coverage"] == pytest.approx(0.70)
    assert row["is_valid"]


def test_flag_shares_are_relative_to_present_values():
    ts, gl = _series(4, [100.0, 110.0, 120.0, NAN])
    row = metrics_for_window(
        ts, gl,
        pd.Timestamp("2015-01-01 00:00"), pd.Timestamp("2015-01-01 00:20"),
        is_imputed=np.array([False, True, False, False]),
        is_capped=np.array([False, False, False, True]),
        min_days=0,
    )
    assert row["n_present"] == 3
    assert row["pct_imputed"] == pytest.approx(100 / 3)
    assert row["pct_capped"] == 0.0     # NaN 슬롯의 플래그는 세지 않는다


# ============================================================================
# 환자 단위 래퍼
# ============================================================================


def _patient(days: float, value: float = 120.0, pid: str = "replace_bg_1",
             start="2015-01-01", coverage: float = 1.0) -> pd.DataFrame:
    n = int(days * SLOTS_PER_DAY)
    values = np.full(n, value)
    if coverage < 1.0:
        values[round(n * coverage):] = NAN
    return coerce_processed_cgm_frame(
        pd.DataFrame({
            "patient_id": pid,
            "timestamp": pd.date_range(start, periods=n, freq=f"{GRID_MINUTES}min"),
            "glucose_mgdl": values,
            "source": "replace_bg",
            "is_imputed": False,
            "is_capped": False,
        })
    )


def test_recent_returns_one_row_per_patient():
    df = pd.concat([_patient(30, pid="replace_bg_1"),
                    _patient(30, pid="replace_bg_2")], ignore_index=True)
    out = recent_metrics(df)

    assert len(out) == 2
    assert list(out.columns) == METRIC_COLUMNS
    assert out["days"].eq(DEFAULT_WINDOW_DAYS).all()


def test_recent_window_ends_at_the_last_observation():
    df = _patient(30)
    out = recent_metrics(df)
    last = df["timestamp"].max()

    assert out["window_end"].iloc[0] == last + pd.Timedelta(minutes=GRID_MINUTES)
    assert out["window_start"].iloc[0] == out["window_end"].iloc[0] - pd.Timedelta(
        days=DEFAULT_WINDOW_DAYS
    )


def test_recent_window_covers_the_full_expected_slot_count():
    out = recent_metrics(_patient(30))
    assert out["n_expected"].iloc[0] == DEFAULT_WINDOW_DAYS * SLOTS_PER_DAY
    assert out["coverage"].iloc[0] == pytest.approx(1.0)


def test_recent_drops_invalid_windows_by_default():
    """스펙: 미달 창은 AGP 계산에서 제외한다."""
    df = _patient(days=5)                       # 14일 창을 못 채운다
    assert recent_metrics(df).empty


def test_recent_can_keep_invalid_windows_for_diagnosis():
    df = _patient(days=30, coverage=0.3)
    out = recent_metrics(df, drop_invalid=False)

    assert len(out) == 1
    assert not out["is_valid"].iloc[0]
    assert "확보율" in out["invalid_reason"].iloc[0]


def test_patient_shorter_than_window_is_skipped_entirely():
    short = _patient(days=10, pid="replace_bg_short")
    long = _patient(days=30, pid="replace_bg_long")
    out = recent_metrics(pd.concat([short, long], ignore_index=True))

    assert out["patient_id"].tolist() == ["replace_bg_long"]


# --- 슬라이딩 ---------------------------------------------------------------


def test_sliding_first_window_equals_recent_window():
    """뒤에서부터 창을 만들기 때문에 성립해야 하는 성질."""
    df = _patient(days=60)
    recent = recent_metrics(df)
    sliding = sliding_metrics(df)

    newest = sliding.sort_values("window_start").iloc[-1]
    assert newest["window_start"] == recent["window_start"].iloc[0]
    assert newest["tir"] == pytest.approx(recent["tir"].iloc[0])


def test_sliding_window_count_follows_stride():
    """60일 데이터, 14일 창, 7일 stride → 창 시작이 0·7·…·46일."""
    out = sliding_metrics(_patient(days=60))
    assert len(out) == 7

    starts = out["window_start"].sort_values().to_numpy()
    gaps = np.diff(starts) / np.timedelta64(1, "D")
    assert np.allclose(gaps, 7.0)


def test_sliding_windows_do_not_run_off_the_start():
    df = _patient(days=60)
    out = sliding_metrics(df)
    assert out["window_start"].min() >= df["timestamp"].min()


def test_sliding_yields_far_more_samples_than_recent():
    df = pd.concat(
        [_patient(days=90, pid=f"replace_bg_{i}") for i in range(3)],
        ignore_index=True,
    )
    assert len(sliding_metrics(df)) > 3 * len(recent_metrics(df))


def test_sliding_max_windows_caps_per_patient():
    out = sliding_metrics(_patient(days=90), max_windows=3)
    assert len(out) == 3


def test_stride_equal_to_window_gives_disjoint_windows():
    out = sliding_metrics(_patient(days=42), stride_days=14).sort_values(
        "window_start"
    )
    ends = out["window_end"].to_numpy()[:-1]
    starts = out["window_start"].to_numpy()[1:]
    assert (ends == starts).all()


# --- 입력 검증 --------------------------------------------------------------


def test_raw_frame_is_rejected_with_a_useful_message():
    from src.schema import empty_cgm_frame

    with pytest.raises(SchemaError, match="preprocess_cgm"):
        recent_metrics(empty_cgm_frame())


def test_empty_processed_frame_returns_empty_table():
    from src.schema import empty_processed_cgm_frame

    out = recent_metrics(empty_processed_cgm_frame())
    assert out.empty
    assert list(out.columns) == METRIC_COLUMNS


def test_metrics_are_computed_per_patient_not_pooled():
    low = _patient(days=30, value=60.0, pid="replace_bg_low")
    high = _patient(days=30, value=300.0, pid="replace_bg_high")
    out = recent_metrics(pd.concat([low, high], ignore_index=True)).set_index(
        "patient_id"
    )

    assert out.loc["replace_bg_low", "tbr"] == pytest.approx(100.0)
    assert out.loc["replace_bg_high", "tar"] == pytest.approx(100.0)
