"""규칙 기반 패턴 탐지 테스트.

각 규칙이 **정의대로 켜지고, 경계 바로 아래에서는 꺼지는지**를 합성 시계열로
고정한다. 임계값이 조용히 바뀌면 여기서 깨져야 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import recent_metrics
from src.patterns import (
    DAWN_RISE_THRESHOLD,
    FINDING_COLUMNS,
    NIGHT_HYPO_MIN_DAYS,
    PATTERN_DEFINITIONS,
    PATTERN_ORDER,
    POSTPRANDIAL_RISE_THRESHOLD,
    PROLONGED_HYPO_MINUTES,
    RAPID_DROP_MIN_EPISODES,
    SICKDAY_MIN_MINUTES,
    detect_patterns,
    pattern_catalog,
    summarize_patterns,
)
from src.preprocess import GRID_MINUTES
from src.schema import (
    coerce_event_frame,
    coerce_processed_cgm_frame,
    empty_cgm_frame,
    SchemaError,
)

SLOTS_PER_DAY = 24 * 60 // GRID_MINUTES
DAYS = 15                       # 14일 창을 확실히 채우려고 하루 더 준다
START = pd.Timestamp("2015-01-01")


def _frame(values: np.ndarray, pid: str = "replace_bg_1") -> pd.DataFrame:
    return coerce_processed_cgm_frame(
        pd.DataFrame({
            "patient_id": pid,
            "timestamp": pd.date_range(START, periods=len(values),
                                       freq=f"{GRID_MINUTES}min"),
            "glucose_mgdl": values,
            "source": "replace_bg",
            "is_imputed": False,
            "is_capped": False,
        })
    )


def _flat(value: float = 120.0, days: int = DAYS) -> np.ndarray:
    return np.full(days * SLOTS_PER_DAY, value, dtype="float64")


def _index(day: int, hour: int, minute: int = 0) -> int:
    return day * SLOTS_PER_DAY + (hour * 60 + minute) // GRID_MINUTES


def _patterns(values: np.ndarray, events=None, pid="replace_bg_1") -> set[str]:
    found = detect_patterns(_frame(values, pid), events=events)
    return set(found["pattern"])


# ============================================================================
# 카탈로그
# ============================================================================


def test_every_pattern_has_a_quotable_definition():
    catalog = pattern_catalog()
    assert len(catalog) == len(PATTERN_DEFINITIONS)
    assert catalog["definition"].str.len().min() > 10
    assert catalog["basis"].notna().all()


def test_catalog_joins_on_the_finding_pattern_column():
    values = _flat(300.0)
    found = detect_patterns(_frame(values))
    merged = found.merge(pattern_catalog(), on="pattern", how="left")
    assert merged["definition"].notna().all()


def test_definitions_do_not_assert_causes():
    """원인 단정 금지. 정의문에 원인·조치 어휘가 들어가면 안 된다."""
    banned = ["인슐린 부족", "때문", "원인", "권장", "늘리", "줄이", "조절하"]
    for definition in PATTERN_DEFINITIONS.values():
        for word in banned:
            assert word not in definition.definition, (definition.key, word)


# ============================================================================
# 지표 임계 기반 (국제 합의 목표치)
# ============================================================================


def test_low_tir_and_high_variability_fire_together():
    """혈당이 널뛰면서 범위를 벗어나는 시계열."""
    values = _flat()
    values[::2] = 60.0
    values[1::2] = 300.0
    assert {"low_time_in_range", "high_variability"} <= _patterns(values)


def test_in_range_patient_has_no_metric_findings():
    found = _patterns(_flat(120.0))
    assert "low_time_in_range" not in found
    assert "excess_hypoglycemia" not in found
    assert "high_variability" not in found


# 창은 뒤 14일만 보므로 저혈당은 시계열 끝쪽에 넣어야 창 안에 들어온다.


def test_excess_hypoglycemia_fires_above_4_percent():
    values = _flat(120.0)
    values[-int(14 * SLOTS_PER_DAY * 0.06):] = 60.0    # 창 내 TBR 6% > 4%
    assert "excess_hypoglycemia" in _patterns(values)


def test_excess_hypoglycemia_silent_below_target():
    values = _flat(120.0)
    values[-int(14 * SLOTS_PER_DAY * 0.02):] = 60.0    # 창 내 TBR 2% <= 4%
    assert "excess_hypoglycemia" not in _patterns(values)


# ============================================================================
# 야간 저혈당
# ============================================================================


def _with_night_hypo(n_nights: int) -> np.ndarray:
    values = _flat(120.0)
    for day in range(1, 1 + n_nights):         # 02:00~02:30에 저혈당
        values[_index(day, 2, 0):_index(day, 2, 30)] = 60.0
    return values


def test_nocturnal_hypoglycemia_fires_at_three_nights():
    assert "nocturnal_hypoglycemia" in _patterns(_with_night_hypo(NIGHT_HYPO_MIN_DAYS))


def test_nocturnal_hypoglycemia_silent_at_two_nights():
    assert "nocturnal_hypoglycemia" not in _patterns(
        _with_night_hypo(NIGHT_HYPO_MIN_DAYS - 1)
    )


def test_daytime_hypoglycemia_is_not_nocturnal():
    values = _flat(120.0)
    for day in range(1, 8):                    # 14:00 낮 저혈당 7일
        values[_index(day, 14, 0):_index(day, 14, 30)] = 60.0
    assert "nocturnal_hypoglycemia" not in _patterns(values)


def test_nocturnal_finding_reports_nights_and_denominator():
    found = detect_patterns(_frame(_with_night_hypo(5)))
    row = found[found["pattern"] == "nocturnal_hypoglycemia"].iloc[0]
    assert row["value"] == 5
    assert row["n_events"] == 5
    assert row["n_total"] >= 14                # 야간 데이터가 있는 밤 수
    assert row["unit"] == "일"


# ============================================================================
# 새벽 상승
# ============================================================================


def _with_dawn_rise(rise: float) -> np.ndarray:
    values = _flat(120.0)
    for day in range(DAYS):
        values[_index(day, 6, 30):_index(day, 7, 0)] = 120.0 + rise
    return values


def test_dawn_rise_fires_at_threshold():
    assert "dawn_rise" in _patterns(_with_dawn_rise(DAWN_RISE_THRESHOLD))


def test_dawn_rise_silent_below_threshold():
    assert "dawn_rise" not in _patterns(_with_dawn_rise(DAWN_RISE_THRESHOLD - 5))


def test_dawn_fall_does_not_fire():
    assert "dawn_rise" not in _patterns(_with_dawn_rise(-40.0))


def test_dawn_finding_reports_the_measured_rise():
    found = detect_patterns(_frame(_with_dawn_rise(40.0)))
    row = found[found["pattern"] == "dawn_rise"].iloc[0]
    assert row["value"] == pytest.approx(40.0)
    assert row["unit"] == "mg/dL"


def test_dawn_rise_ignores_rises_outside_the_band():
    """저녁에 오르는 것은 새벽 상승이 아니다."""
    values = _flat(120.0)
    for day in range(DAYS):
        values[_index(day, 19, 0):_index(day, 20, 0)] = 200.0
    assert "dawn_rise" not in _patterns(values)


# ============================================================================
# 식후 스파이크
# ============================================================================


def _meals_and_series(rise: float, n_meals: int = 14):
    values = _flat(120.0)
    times = []
    for day in range(n_meals):
        meal = START + pd.Timedelta(days=day, hours=12)
        times.append(meal)
        peak = _index(day, 13, 0)
        values[peak:peak + 6] = 120.0 + rise
    events = coerce_event_frame(pd.DataFrame({
        "patient_id": "replace_bg_1",
        "timestamp": times,
        "event_type": "meal",
        "value": 40.0,
        "unit": "g",
        "text": None,
        "source": "replace_bg",
    }))
    return values, events


def test_postprandial_spike_fires_at_threshold():
    values, events = _meals_and_series(POSTPRANDIAL_RISE_THRESHOLD)
    assert "postprandial_spike" in _patterns(values, events=events)


def test_postprandial_spike_silent_below_threshold():
    values, events = _meals_and_series(POSTPRANDIAL_RISE_THRESHOLD - 20)
    assert "postprandial_spike" not in _patterns(values, events=events)


def test_postprandial_spike_needs_enough_meals():
    """식사가 몇 끼 없으면 판정하지 않는다."""
    values, events = _meals_and_series(150.0, n_meals=3)
    assert "postprandial_spike" not in _patterns(values, events=events)


def test_postprandial_spike_skipped_without_events():
    values, _ = _meals_and_series(150.0)
    assert "postprandial_spike" not in _patterns(values, events=None)


def test_postprandial_finding_reports_meal_counts():
    values, events = _meals_and_series(120.0)
    found = detect_patterns(_frame(values), events=events)
    row = found[found["pattern"] == "postprandial_spike"].iloc[0]
    assert row["n_events"] == row["n_total"]      # 전부 스파이크
    assert row["value"] == pytest.approx(100.0)


def test_rise_measured_from_the_meal_not_the_window_minimum():
    """식전 값이 낮은 게 아니라 식후에 오른 폭을 봐야 한다."""
    values = _flat(120.0)
    times = []
    for day in range(14):
        values[_index(day, 8, 0):_index(day, 12, 0)] = 60.0   # 식전 저혈당 구간
        times.append(START + pd.Timedelta(days=day, hours=12))
        values[_index(day, 13, 0):_index(day, 13, 30)] = 160.0  # 식후 +40
    events = coerce_event_frame(pd.DataFrame({
        "patient_id": "replace_bg_1", "timestamp": times, "event_type": "meal",
        "value": 40.0, "unit": "g", "text": None, "source": "replace_bg",
    }))
    # 식전 120 → 식후 최고 160이면 상승폭 40 < 80이라 켜지면 안 된다
    assert "postprandial_spike" not in _patterns(values, events=events)


# ============================================================================
# 시간대 집중
# ============================================================================


def test_hypo_concentration_names_the_time_band():
    values = _flat(120.0)
    for day in range(DAYS):                     # 04:00~06:00에만 저혈당
        values[_index(day, 4, 0):_index(day, 6, 0)] = 60.0
    found = detect_patterns(_frame(values))
    hypo = found[found["pattern"] == "hypo_concentration"]

    assert len(hypo) == 1
    assert "04:00–06:00" in hypo["detail"].iloc[0]


def test_hyper_concentration_names_the_time_band():
    values = _flat(120.0)
    for day in range(DAYS):
        values[_index(day, 20, 0):_index(day, 22, 0)] = 300.0
    found = detect_patterns(_frame(values))
    hyper = found[found["pattern"] == "hyper_concentration"]

    assert len(hyper) == 1
    assert "20:00–22:00" in hyper["detail"].iloc[0]


def test_uniformly_high_patient_has_no_concentration_finding():
    """하루 종일 높으면 '몰리는 시간대'가 아니다."""
    found = _patterns(_flat(300.0))
    assert "hyper_concentration" not in found
    assert "low_time_in_range" in found          # 대신 이쪽이 켜진다


# ============================================================================
# 진입점 / 출력 형태
# ============================================================================


def test_output_columns_are_fixed():
    found = detect_patterns(_frame(_flat(300.0)))
    assert list(found.columns) == FINDING_COLUMNS


def test_findings_carry_the_metrics_window():
    frame = _frame(_flat(300.0))
    metrics = recent_metrics(frame)
    found = detect_patterns(frame, metrics=metrics)

    assert (found["window_start"] == metrics["window_start"].iloc[0]).all()
    assert (found["window_end"] == metrics["window_end"].iloc[0]).all()


def test_healthy_patient_yields_no_findings():
    found = detect_patterns(_frame(_flat(120.0)))
    assert found.empty
    assert list(found.columns) == FINDING_COLUMNS


def test_patient_without_a_valid_window_is_absent():
    short = _frame(_flat(300.0, days=5))
    assert detect_patterns(short).empty


def test_patterns_are_detected_per_patient():
    low = _frame(_flat(60.0), pid="replace_bg_low")
    fine = _frame(_flat(120.0), pid="replace_bg_fine")
    found = detect_patterns(pd.concat([low, fine], ignore_index=True))

    assert set(found["patient_id"]) == {"replace_bg_low"}


def test_raw_frame_is_rejected():
    with pytest.raises(SchemaError, match="preprocess_cgm"):
        detect_patterns(empty_cgm_frame())


def test_summarize_counts_patients_per_pattern():
    a = _frame(_flat(300.0), pid="replace_bg_a")
    b = _frame(_flat(300.0), pid="replace_bg_b")
    found = detect_patterns(pd.concat([a, b], ignore_index=True))
    summary = summarize_patterns(found, n_patients=2)

    row = summary[summary["pattern"] == "low_time_in_range"].iloc[0]
    assert row["n_patients"] == 2
    assert row["pct_patients"] == pytest.approx(100.0)
    assert row["name"] == "목표범위 미달"


def test_detail_text_is_descriptive_not_prescriptive():
    """LLM이 인용할 문장에 원인·조치가 섞이면 안 된다."""
    values = _flat(120.0)
    values[::2] = 60.0
    values[1::2] = 300.0
    found = detect_patterns(_frame(values))

    banned = ["해야", "권장", "늘리", "줄이", "때문", "원인"]
    for detail in found["detail"]:
        for word in banned:
            assert word not in detail, (detail, word)


# ============================================================================
# 2026-09 문헌 조사로 추가한 규칙
# ============================================================================


def _events_of(kind: str, times, values=None, unit=None):
    return coerce_event_frame(pd.DataFrame({
        "patient_id": "replace_bg_1",
        "timestamp": list(times),
        "event_type": kind,
        "value": values if values is not None else np.nan,
        "unit": unit,
        "text": None,
        "source": "replace_bg",
    }))


def test_findings_are_ordered_hypoglycemia_first():
    """저혈당 우선 해석 순서. 창 안 어떤 조합이 나와도 순위대로 나온다."""
    values = _flat(120.0)
    values[::2] = 60.0            # TBR 과다 + level 2 + 변동성
    values[1::2] = 300.0
    found = detect_patterns(_frame(values))
    order = found["pattern"].tolist()
    assert order.index("excess_hypoglycemia") < order.index("excess_hyperglycemia")
    assert order.index("excess_hyperglycemia") < order.index("high_variability")
    assert order == sorted(order, key=PATTERN_ORDER.index)


def test_low_coverage_yields_only_that_finding():
    values = _flat(300.0)
    values[len(values) // 2:] = np.nan       # 확보율 ~50%
    found = detect_patterns(_frame(values))
    assert found["pattern"].tolist() == ["low_data_coverage"]
    assert found["value"].iloc[0] < 70.0


def test_low_coverage_can_be_suppressed():
    values = _flat(300.0)
    values[len(values) // 2:] = np.nan
    assert detect_patterns(_frame(values), include_low_coverage=False).empty


def test_level2_thresholds_fire_independently():
    values = _flat(120.0)
    values[:: 50] = 50.0                  # 2% 미만 54 → level 2 켜짐(>1%)
    assert "excess_hypoglycemia_level2" in _patterns(values)
    values = _flat(120.0)
    values[:: 10] = 260.0                 # 10% 초과 250 → level 2 켜짐(>5%)
    found = _patterns(values)
    assert "excess_hyperglycemia_level2" in found
    assert "excess_hyperglycemia" not in found   # TAR 10% < 25%


def test_prolonged_hypo_needs_more_than_120_minutes():
    values = _flat(120.0)
    slots = PROLONGED_HYPO_MINUTES // GRID_MINUTES
    values[_index(3, 14):_index(3, 14) + slots] = 60.0         # 정확히 120분
    assert "prolonged_hypoglycemia" not in _patterns(values)
    values[_index(3, 14):_index(3, 14) + slots + 1] = 60.0     # 125분
    assert "prolonged_hypoglycemia" in _patterns(values)


def test_sickday_needs_two_long_episodes():
    values = _flat(120.0)
    slots = SICKDAY_MIN_MINUTES // GRID_MINUTES
    values[_index(2, 10):_index(2, 10) + slots] = 300.0
    assert "sickday_ketone_risk" not in _patterns(values)
    values[_index(5, 10):_index(5, 10) + slots] = 300.0
    assert "sickday_ketone_risk" in _patterns(values)


def test_fasting_hyperglycemia_uses_morning_window():
    values = _flat(120.0)
    for day in range(DAYS):
        values[_index(day, 6):_index(day, 7)] = 150.0
    assert "fasting_hyperglycemia" in _patterns(values)
    for day in range(DAYS):
        values[_index(day, 6):_index(day, 7)] = 125.0
    assert "fasting_hyperglycemia" not in _patterns(values)


def test_rapid_drop_counts_episodes():
    values = _flat(200.0)
    for day in range(1, RAPID_DROP_MIN_EPISODES + 1):     # 0일차는 최근 14일 창 밖
        i = _index(day, 15)
        values[i:i + 3] = [200.0, 160.0, 120.0]     # 10분에 80 = 8 mg/dL/min
        values[i + 3:i + 6] = 100.0
    assert "rapid_drop" in _patterns(values)


def test_exercise_delayed_hypo_needs_exercise_log():
    values = _flat(120.0)
    times = []
    for day in range(1, 4):                                         # 0일차는 창 밖
        times.append(START + pd.Timedelta(days=day, hours=17))     # 17:00 운동 60분
        night = _index(day + 1, 3)                                  # 다음날 03:00 저혈당
        values[night:night + 6] = 60.0
    events = _events_of("exercise", times, values=60.0, unit="min")

    found = detect_patterns(_frame(values), events=events)
    row = found[found["pattern"] == "exercise_delayed_hypoglycemia"].iloc[0]
    assert row["n_events"] == 3 and row["n_total"] == 3
    assert "야간 시작 3회" in row["detail"]
    assert "exercise_delayed_hypoglycemia" not in _patterns(values, events=None)


def test_exercise_hypo_within_two_hours_is_not_delayed():
    values = _flat(120.0)
    times = []
    for day in range(1, 4):
        times.append(START + pd.Timedelta(days=day, hours=17))
        soon = _index(day, 18, 30)                                  # 종료 30분 뒤
        values[soon:soon + 6] = 60.0
    events = _events_of("exercise", times, values=60.0, unit="min")
    assert "exercise_delayed_hypoglycemia" not in _patterns(values, events=events)
