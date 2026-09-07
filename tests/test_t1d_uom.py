"""T1D-UOM 로더 테스트.

합성 데이터로 파싱 규칙을 검증하고, 원본이 있을 때만 실데이터 통합 테스트를 돈다.

이 데이터셋에서 가장 위험한 것은 **날짜 형식**이다. 원본 README가 ``MM/DD/YYYY``
라고 적어 놨지만 실제로는 ``DD/MM/YYYY``이고, 두 필드가 모두 12 이하인 날짜는
뒤집어 읽어도 예외가 나지 않는다. 그래서 여기 테스트를 제일 앞에 둔다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocess import MMOL_TO_MGDL
from src.schema import SchemaError, validate_cgm_frame, validate_event_frame
from src.loaders import t1d_uom as uom

RAW_AVAILABLE = uom.DEFAULT_RAW_DIR.is_dir()
requires_raw = pytest.mark.skipif(
    not RAW_AVAILABLE, reason=f"원본이 없다: {uom.DEFAULT_RAW_DIR}"
)


# --- 합성 원본 만들기 --------------------------------------------------------


def _write(root, folder: str, name: str, rows: str) -> None:
    path = root / folder
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(rows, encoding="utf-8")


def _make_raw(
    tmp_path,
    pid: str = "2302",
    glucose: str | None = None,
    basal: str | None = None,
    bolus: str | None = None,
    nutrition: str | None = None,
):
    """MODALITY에 등록된 환자 하나짜리 최소 원본 트리."""
    root = tmp_path / "raw"
    _write(
        root,
        "Glucose Data",
        f"UoMGlucose{pid}.csv",
        glucose
        if glucose is not None
        else "bg_ts,value\n01/10/2023 00:00,7.5\n01/10/2023 00:15,8.0\n",
    )
    if basal is not None:
        _write(root, "Insulin Data/Basal Data", f"UoMBasal{pid}.csv", basal)
    if bolus is not None:
        _write(root, "Insulin Data/Bolus Data", f"UoMBolus{pid}.csv", bolus)
    if nutrition is not None:
        _write(root, "Nutrition Data", f"UoMNutrition{pid}.csv", nutrition)
    return root


# --- 날짜 형식 ---------------------------------------------------------------


def test_dates_are_parsed_day_first_not_month_first(tmp_path):
    """``01/10/2023``은 2023년 10월 1일이다. README의 MM/DD를 따르면 안 된다."""
    root = _make_raw(tmp_path, glucose="bg_ts,value\n01/10/2023 09:00,7.5\n")

    ts = uom.load(root).cgm["timestamp"].iloc[0]
    assert (ts.year, ts.month, ts.day) == (2023, 10, 1)


def test_unambiguous_day_over_twelve_parses(tmp_path):
    """``22/10/2023``은 월-일을 뒤집으면 파싱 자체가 안 되는 날짜다."""
    root = _make_raw(tmp_path, glucose="bg_ts,value\n22/10/2023 09:00,7.5\n")

    ts = uom.load(root).cgm["timestamp"].iloc[0]
    assert (ts.month, ts.day) == (10, 22)


def test_rows_outside_the_study_period_are_dropped(tmp_path):
    """연구 기간 밖 기록이 실제로 존재한다(2302 볼러스 2023-01, 2314 식사 2024-12)."""
    root = _make_raw(
        tmp_path,
        glucose="bg_ts,value\n01/10/2023 09:00,7.5\n",
        bolus=(
            "bolus_ts,bolus_dose\n"
            "14/01/2023 12:00,3.0\n"   # 연구 시작 전
            "02/10/2023 12:00,4.0\n"   # 정상
            "15/12/2024 12:00,5.0\n"   # 연구 종료 후
        ),
    )

    bolus = uom.load(root).events.query("event_type == 'insulin_bolus'")
    assert len(bolus) == 1
    assert bolus["value"].iloc[0] == 4.0


# --- 단위 -------------------------------------------------------------------


def test_mmol_is_converted_to_mgdl(tmp_path):
    root = _make_raw(tmp_path, glucose="bg_ts,value\n01/10/2023 00:00,7.5\n")

    assert uom.load(root).cgm["glucose_mgdl"].iloc[0] == pytest.approx(
        7.5 * MMOL_TO_MGDL
    )


def test_physiologically_impossible_values_are_dropped(tmp_path):
    """2307에 0.1 mmol/L(=1.8 mg/dL)가 있다. 스키마 검증에 걸리기 전에 버린다."""
    root = _make_raw(
        tmp_path,
        glucose="bg_ts,value\n01/10/2023 00:00,0.1\n01/10/2023 00:15,7.5\n",
    )

    cgm = uom.load(root).cgm
    assert len(cgm) == 1
    assert cgm["glucose_mgdl"].iloc[0] == pytest.approx(7.5 * MMOL_TO_MGDL)


# --- 치료 방식 ---------------------------------------------------------------


@pytest.mark.parametrize(
    "pid, why",
    [
        ("2301", "폐루프. 알고리즘이 기저를 조절한다"),
        ("2304", "개방루프 펌프. 기저를 사람이 입력하지 않는다"),
        ("2303", "기저 파일이 없어 치료 방식을 모른다"),
    ],
)
def test_non_mdi_patients_are_excluded_by_default(tmp_path, pid, why):
    """기본 대상은 MDI뿐이다. 나머지는 명시적으로 요청해야 들어온다."""
    root = _make_raw(tmp_path, pid=pid)

    with pytest.raises(SchemaError, match="해당하는 환자가 없다"):
        uom.load(root)


def test_mdi_patient_loads_by_default(tmp_path):
    root = _make_raw(tmp_path, pid="2302")

    assert uom.load(root).patients["modality"].tolist() == ["MDI"]


def test_mdi_patients_constant_matches_the_modality_map():
    assert uom.MDI_PATIENTS == (
        "2302", "2305", "2306", "2313", "2314", "2401", "2403", "2405",
    )
    assert uom.DEFAULT_MODALITIES == frozenset({"MDI"})


def test_closed_loop_can_be_requested_explicitly(tmp_path):
    root = _make_raw(tmp_path, pid="2301")

    result = uom.load(root, modalities={"CLOSED_LOOP"})
    assert result.patients["patient_id"].tolist() == ["t1d_uom_2301"]
    assert result.patients["modality"].tolist() == ["CLOSED_LOOP"]


def test_unknown_modality_is_rejected(tmp_path):
    root = _make_raw(tmp_path)

    with pytest.raises(ValueError, match="알 수 없는 치료 방식"):
        uom.load(root, modalities={"HYBRID"})


def test_unregistered_patient_raises(tmp_path):
    """새 환자가 배포본에 추가되면 조용히 넘어가지 말고 실측을 강제한다."""
    root = _make_raw(tmp_path)
    _write(root, "Glucose Data", "UoMGlucose9999.csv", "bg_ts,value\n01/10/2023 00:00,7.5\n")

    with pytest.raises(SchemaError, match="MODALITY에 등록되지 않은 환자"):
        uom.find_patients(root)


def test_patient_id_prefix():
    assert uom.patient_id("2302") == "t1d_uom_2302"


# --- 기저: 주사 vs 주입률 ----------------------------------------------------


def test_long_acting_basal_becomes_subcutaneous_injection(tmp_path):
    """MDI의 지속형 기저는 하루 1~2회 **주사**다. IU/h 주입률이 아니다."""
    root = _make_raw(
        tmp_path,
        pid="2302",
        basal="basal_ts,basal_dose,insulin_kind\n02/10/2023 08:30,9.0,L\n",
    )

    events = uom.load(root).events
    assert events["event_type"].tolist() == ["insulin_sc"]
    assert events["unit"].iloc[0] == "IU"
    assert events["value"].iloc[0] == 9.0


def test_rapid_basal_becomes_a_pump_rate(tmp_path):
    root = _make_raw(
        tmp_path,
        pid="2304",
        basal="basal_ts,basal_dose,insulin_kind\n09/10/2023 00:00,1.4,R\n",
    )

    events = uom.load(root, modalities={"PUMP_OPEN"}).events
    assert events["event_type"].tolist() == ["insulin_basal_rate"]
    assert events["unit"].iloc[0] == "IU/h"


# --- 원본 파일의 지저분한 부분 -----------------------------------------------


def test_bom_and_trailing_empty_columns_are_tolerated(tmp_path):
    """UoMBasal2301.csv에는 BOM과 빈 꼬리 컬럼 2개가 같이 있다."""
    root = _make_raw(
        tmp_path,
        pid="2302",
        basal="﻿basal_ts,basal_dose,insulin_kind,,\n02/10/2023 08:30,9.0,L,,\n",
    )

    events = uom.load(root).events
    assert len(events) == 1
    assert events["value"].iloc[0] == 9.0


def test_meal_keeps_carbs_and_original_text(tmp_path):
    root = _make_raw(
        tmp_path,
        nutrition=(
            "meal_ts,meal_type,meal_tag,carbs_g,prot_g,fat_g,fibre_g\n"
            "02/10/2023 12:00,Lunch,Huel+Peanutbutter,19,50,32,10\n"
        ),
    )

    meal = uom.load(root).events.iloc[0]
    assert meal["event_type"] == "meal"
    assert meal["value"] == 19.0
    assert meal["unit"] == "g"
    assert meal["text"] == "Lunch; Huel+Peanutbutter"


def test_meal_without_carbs_keeps_the_event(tmp_path):
    """탄수화물 g이 비어 있어도 '먹었다'는 사실은 남긴다. value만 NaN이다."""
    root = _make_raw(
        tmp_path,
        nutrition=(
            "meal_ts,meal_type,meal_tag,carbs_g,prot_g,fat_g,fibre_g\n"
            "02/10/2023 17:15,Dinner,,,,,\n"
        ),
    )

    meal = uom.load(root).events.iloc[0]
    assert meal["event_type"] == "meal"
    assert pd.isna(meal["value"])
    assert meal["text"] == "Dinner"


def test_missing_optional_tables_are_not_an_error(tmp_path):
    """2303은 기저·볼러스·식사가 전부 없고, 2310은 식사 파일이 없다."""
    root = _make_raw(tmp_path)

    result = uom.load(root)
    assert result.events.empty
    validate_event_frame(result.events)


def test_missing_raw_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        uom.load(tmp_path / "nope")


# --- 실데이터 통합 -----------------------------------------------------------


@requires_raw
def test_real_data_satisfies_the_schema_contract():
    result = uom.load()
    validate_cgm_frame(result.cgm)
    validate_event_frame(result.events)


@requires_raw
def test_real_data_yields_only_the_eight_mdi_patients():
    result = uom.load()

    assert set(result.patients["modality"]) == {"MDI"}
    assert result.patients["patient_id"].tolist() == [
        uom.patient_id(p) for p in uom.MDI_PATIENTS
    ]


@requires_raw
def test_real_data_pump_patients_are_reachable_when_asked():
    """기본값에서 빠질 뿐 못 읽는 것은 아니다."""
    result = uom.load(modalities={"PUMP_OPEN"})

    assert set(result.patients["modality"]) == {"PUMP_OPEN"}
    assert len(result.patients) == 4


@requires_raw
def test_real_data_has_both_sensor_cadences():
    """MDI 8명 안에서도 5분(2313)과 15분이 섞여 있다. 하나로 가정하면 안 된다."""
    intervals = uom.load().patients["sensor_interval_min"]

    assert set(intervals) == {5, 15}
    assert (intervals == 5).sum() == 1


@requires_raw
def test_real_data_stays_inside_the_study_period():
    cgm = uom.load().cgm
    assert cgm["timestamp"].min() >= uom.STUDY_START
    assert cgm["timestamp"].max() <= uom.STUDY_END
