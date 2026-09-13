"""OhioT1DM 로더 테스트.

합성 XML로 파싱 규칙을 검증하고, 원본이 있을 때만 실데이터 통합 테스트를 돈다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.schema import validate_cgm_frame, validate_event_frame
from src.loaders import ohio_t1dm as ohio

RAW_AVAILABLE = ohio.DEFAULT_RAW_DIR.is_dir()
requires_raw = pytest.mark.skipif(
    not RAW_AVAILABLE, reason=f"원본이 없다: {ohio.DEFAULT_RAW_DIR}"
)

_GLUCOSE = """
  <glucose_level>
    <event ts="07-12-2021 01:17:00" value="101"/>
    <event ts="07-12-2021 01:22:00" value="98"/>
  </glucose_level>
"""


def _xml(body: str, pid: str = "559") -> str:
    return f'<?xml version="1.0"?>\n<patient id="{pid}" weight="99" insulin_type="Novalog">{_GLUCOSE}{body}</patient>'


def _make_raw(tmp_path, train_body: str = "", test_body: str | None = None, pid="559"):
    """2018 코호트 환자 하나짜리 최소 트리. test 파일은 body를 주면 만든다."""
    root = tmp_path / "raw"
    (root / "2018" / "train").mkdir(parents=True)
    (root / "2018" / "train" / f"{pid}-ws-training.xml").write_text(
        _xml(train_body, pid), encoding="utf-8"
    )
    if test_body is not None:
        (root / "2018" / "test").mkdir(parents=True)
        (root / "2018" / "test" / f"{pid}-ws-testing.xml").write_text(
            _xml(test_body, pid), encoding="utf-8"
        )
    return root


def _events(tmp_path, body: str) -> pd.DataFrame:
    return ohio.load(_make_raw(tmp_path, body)).events


# --- 파일 탐색 / 기본 --------------------------------------------------------


def test_train_and_test_files_become_one_patient(tmp_path):
    root = _make_raw(tmp_path, "", "")
    result = ohio.load(root)

    assert result.patients["patient_id"].tolist() == ["ohio_t1dm_559"]
    assert result.patients["cohort"].tolist() == ["2018"]
    # 두 파일이 같은 혈당 행을 실었으므로 중복이 접혀 2행만 남는다.
    assert len(result.cgm) == 2


def test_dates_are_parsed_day_first():
    ts = ohio._ts(pd.Series(["07-12-2021 01:17:00"])).iloc[0]
    assert (ts.year, ts.month, ts.day) == (2021, 12, 7)


def test_glucose_is_already_mgdl(tmp_path):
    cgm = ohio.load(_make_raw(tmp_path)).cgm
    assert cgm["glucose_mgdl"].tolist() == [101.0, 98.0]


def test_missing_raw_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ohio.load(tmp_path / "nope")


# --- 볼러스 -----------------------------------------------------------------


def test_normal_bolus_keeps_bwz_carb_input_in_text(tmp_path):
    ev = _events(tmp_path, """
      <bolus>
        <event ts_begin="07-12-2021 01:08:04" ts_end="07-12-2021 01:08:04" type="normal" dose="1.6" bwz_carb_input="25"/>
      </bolus>""")

    b = ev[ev["event_type"] == "insulin_bolus"].iloc[0]
    assert b["value"] == 1.6
    assert b["unit"] == "IU"
    assert b["text"] == "normal; bwz_carb_input=25"


def test_square_bolus_becomes_extended_with_duration(tmp_path):
    ev = _events(tmp_path, """
      <bolus>
        <event ts_begin="07-12-2021 12:41:00" ts_end="07-12-2021 13:11:00" type="square dual" dose="5.6"/>
      </bolus>""")

    assert ev["event_type"].tolist() == ["insulin_bolus_extended"]
    assert ev["text"].iloc[0] == "square dual; duration_min=30"


def test_unknown_bolus_type_raises(tmp_path):
    from src.schema import SchemaError

    with pytest.raises(SchemaError, match="bolus type"):
        _events(tmp_path, """
          <bolus><event ts_begin="07-12-2021 01:08:04" ts_end="07-12-2021 01:08:04" type="mystery" dose="1"/></bolus>""")


# --- 기저 / 임시 기저 ------------------------------------------------------


def test_temp_basal_zero_is_a_pump_suspend(tmp_path):
    ev = _events(tmp_path, """
      <temp_basal>
        <event ts_begin="07-12-2021 04:49:28" ts_end="07-12-2021 05:08:39" value="0.0"/>
        <event ts_begin="07-12-2021 12:32:06" ts_end="07-12-2021 13:13:06" value="1.9"/>
      </temp_basal>""")

    by_type = ev.set_index("event_type")
    assert by_type.loc["pump_suspend", "text"] == "temp_basal; duration_min=19"
    assert by_type.loc["insulin_basal_rate", "value"] == 1.9
    assert by_type.loc["insulin_basal_rate", "unit"] == "IU/h"


# --- 자기보고 이벤트 --------------------------------------------------------


def test_exercise_keeps_duration_and_intensity(tmp_path):
    ev = _events(tmp_path, """
      <exercise>
        <event ts="13-12-2021 13:55:00" intensity="3" type=" " duration="150" competitive=""/>
      </exercise>""")

    e = ev.iloc[0]
    assert e["event_type"] == "exercise"
    assert e["value"] == 150.0
    assert e["unit"] == "min"
    assert e["text"] == "intensity=3"   # type이 공백이면 빠진다


def test_hypo_event_has_timestamp_only(tmp_path):
    ev = _events(tmp_path, '<hypo_event><event ts="07-12-2021 16:54:00"/></hypo_event>')

    assert ev["event_type"].tolist() == ["hypo_event"]
    assert pd.isna(ev["value"].iloc[0])


def test_sleep_with_swapped_begin_end_uses_the_earlier_time(tmp_path):
    """2018 코호트에는 ts_begin이 ts_end보다 늦은 행이 있다."""
    ev = _events(tmp_path, """
      <sleep>
        <event ts_begin="08-12-2021 04:58:00" ts_end="07-12-2021 21:16:00" quality="2"/>
      </sleep>""")

    s = ev.iloc[0]
    assert s["timestamp"] == pd.Timestamp("2021-12-07 21:16:00")
    assert s["value"] == pytest.approx(7.7, abs=0.01)
    assert s["unit"] == "h"


def test_event_listed_in_both_files_is_kept_once(tmp_path):
    ex = '<exercise><event ts="02-07-2027 21:30:00" intensity="6" duration="30"/></exercise>'
    ev = ohio.load(_make_raw(tmp_path, ex, ex)).events

    assert len(ev[ev["event_type"] == "exercise"]) == 1


# --- 실데이터 통합 -----------------------------------------------------------


@requires_raw
def test_real_data_satisfies_the_schema_contract():
    result = ohio.load()
    validate_cgm_frame(result.cgm)
    validate_event_frame(result.events)


@requires_raw
def test_real_data_has_twelve_patients_in_two_cohorts():
    p = ohio.load().patients
    assert len(p) == 12
    assert p["cohort"].value_counts().to_dict() == {"2018": 6, "2020": 6}


@requires_raw
def test_real_data_glucose_hits_the_sensor_limits_exactly():
    cgm = ohio.load().cgm["glucose_mgdl"]
    assert (cgm.min(), cgm.max()) == (ohio.LOW_SENTINEL_MGDL, ohio.HIGH_SENTINEL_MGDL)


@requires_raw
def test_real_data_self_reported_events_are_present():
    counts = ohio.load().events["event_type"].value_counts()
    assert counts["exercise"] >= 200
    assert counts["hypo_event"] >= 100
