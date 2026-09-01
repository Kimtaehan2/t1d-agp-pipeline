"""ShanghaiT1DM 로더 테스트.

합성 데이터로 파싱 규칙을 검증하고, 원본이 있을 때만 실데이터 통합 테스트를 돈다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocess import MMOL_TO_MGDL
from src.schema import validate_cgm_frame, validate_event_frame
from src.loaders import shanghai_t1dm as sh

RAW_AVAILABLE = sh.DEFAULT_RAW_DIR.is_dir()
requires_raw = pytest.mark.skipif(
    not RAW_AVAILABLE, reason=f"원본이 없다: {sh.DEFAULT_RAW_DIR}"
)


def _raw(**overrides) -> pd.DataFrame:
    """원본 컬럼을 그대로 흉내 낸 2행짜리 DataFrame."""
    base = {
        sh.COL_DATE: [pd.Timestamp("2021-05-04 10:33"), pd.Timestamp("2021-05-04 10:48")],
        sh.COL_CGM: [156.6, 151.2],
        sh.COL_CBG: [None, None],
        sh.COL_KETONE: [None, None],
        sh.COL_DIET_EN: [None, None],
        sh.COL_DIET_ZH: [None, None],
        sh.COL_INSULIN_SC: [None, None],
        sh.COL_ORAL: [None, None],
        sh.COL_CSII_BOLUS: [None, None],
        sh.COL_CSII_BASAL: [None, None],
        sh.COL_INSULIN_IV: [None, None],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _events(**overrides) -> pd.DataFrame:
    return sh._extract_events(_raw(**overrides), "shanghai_t1dm_9999")


# --- 파일명 / 파일 탐색 ------------------------------------------------------


def test_parse_filename_groups_visits_under_one_patient():
    """같은 환자의 여러 내원이 하나의 patient_id로 묶여야 한다 (환자 단위 분할 규칙)."""
    assert sh.parse_filename("1002_0_20210504.xls") == ("shanghai_t1dm_1002", 0)
    assert sh.parse_filename("1002_1_20210521.xls") == ("shanghai_t1dm_1002", 1)
    assert sh.parse_filename("1002_2_20210909.xls") == ("shanghai_t1dm_1002", 2)


def test_parse_filename_rejects_unexpected_format():
    with pytest.raises(ValueError, match="파일명"):
        sh.parse_filename("patient_A.xlsx")


def test_find_files_skips_excel_lock_files(tmp_path):
    (tmp_path / "1001_0_20210730.xlsx").touch()
    (tmp_path / "~$1001_0_20210730.xlsx").touch()
    (tmp_path / "notes.txt").touch()

    names = [p.name for p in sh.find_files(tmp_path)]
    assert names == ["1001_0_20210730.xlsx"]


def test_find_files_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        sh.find_files(tmp_path)


def test_find_files_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        sh.find_files(tmp_path / "nope")


# --- CGM 추출 ---------------------------------------------------------------


def test_cgm_passes_through_mgdl():
    out = sh._extract_cgm(_raw(), "shanghai_t1dm_9999")
    assert out["glucose_mgdl"].tolist() == [156.6, 151.2]


def test_cgm_converts_mmol_source():
    """단위를 하드코딩하지 않았는지 확인 — mmol/L 값이 들어오면 변환돼야 한다."""
    out = sh._extract_cgm(_raw(**{sh.COL_CGM: [8.7, 8.4]}), "shanghai_t1dm_9999")
    assert out["glucose_mgdl"].tolist() == pytest.approx(
        [8.7 * MMOL_TO_MGDL, 8.4 * MMOL_TO_MGDL]
    )


def test_cgm_drops_rows_without_reading():
    out = sh._extract_cgm(_raw(**{sh.COL_CGM: [156.6, None]}), "shanghai_t1dm_9999")
    assert len(out) == 1


# --- 이벤트 추출 -------------------------------------------------------------


def test_meal_placeholder_is_not_an_event():
    ev = _events(
        **{
            sh.COL_DIET_EN: ["data not available", None],
            sh.COL_DIET_ZH: ["未记录", None],
        }
    )
    assert "meal" not in set(ev["event_type"])


def test_meal_prefers_english_text():
    ev = _events(
        **{
            sh.COL_DIET_EN: ["Wonton 250 g", None],
            sh.COL_DIET_ZH: ["馄饨250g", None],
        }
    )
    meals = ev[ev["event_type"] == "meal"]
    assert len(meals) == 1
    assert meals["text"].iloc[0] == "Wonton 250 g"


def test_meal_falls_back_to_chinese_text():
    ev = _events(
        **{
            sh.COL_DIET_EN: ["data not available", None],
            sh.COL_DIET_ZH: ["馄饨250g", None],
        }
    )
    meals = ev[ev["event_type"] == "meal"]
    assert len(meals) == 1
    assert meals["text"].iloc[0] == "馄饨250g"


def test_meal_value_is_nan_because_carbs_are_unknown():
    ev = _events(**{sh.COL_DIET_EN: ["Wonton 250 g", None]})
    assert ev[ev["event_type"] == "meal"]["value"].isna().all()


def test_insulin_sc_dose_is_parsed_from_text():
    ev = _events(**{sh.COL_INSULIN_SC: ["Novolin R, 5 IU", "insulin degludec, 10 IU"]})
    sc = ev[ev["event_type"] == "insulin_sc"].sort_values("timestamp")
    assert sc["value"].tolist() == [5.0, 10.0]
    assert sc["unit"].tolist() == ["IU", "IU"]
    assert sc["text"].iloc[0] == "Novolin R, 5 IU"  # 원본 문자열 보존


def test_oral_agent_handles_nbsp_and_comma_variants():
    ev = _events(**{sh.COL_ORAL: ["\xa0voglibose 0.2 mg", "acarbose, 50 mg"]})
    oral = ev[ev["event_type"] == "oral_agent"].sort_values("timestamp")
    assert oral["value"].tolist() == [0.2, 50.0]
    assert oral["text"].iloc[0] == "voglibose 0.2 mg"  # NBSP 제거됨


def test_insulin_iv_takes_the_insulin_dose_not_the_volume():
    text = "500ml 0.9% sodium chloride,  12 IU Novolin R,  10 ml 10% potassium chloride"
    ev = _events(**{sh.COL_INSULIN_IV: [text, None]})
    iv = ev[ev["event_type"] == "insulin_iv"]
    assert iv["value"].tolist() == [12.0]


def test_pump_suspend_becomes_its_own_event():
    ev = _events(
        **{sh.COL_CSII_BASAL: ["temporarily suspend insulin delivery", 0.7]}
    )
    assert set(ev["event_type"]) == {"pump_suspend", "insulin_basal_rate"}
    assert ev[ev["event_type"] == "pump_suspend"]["value"].isna().all()
    assert ev[ev["event_type"] == "insulin_basal_rate"]["value"].tolist() == [0.7]


def test_cbg_and_ketone_keep_their_units():
    ev = _events(**{sh.COL_CBG: [156.6, None], sh.COL_KETONE: [None, 0.3]})
    by_type = ev.set_index("event_type")
    assert by_type.loc["cbg", "unit"] == "mg/dL"
    assert by_type.loc["blood_ketone", "unit"] == "mmol/L"


def test_no_events_yields_empty_but_valid_frame():
    ev = _events()
    assert ev.empty
    validate_event_frame(ev)


# --- 실데이터 통합 테스트 ----------------------------------------------------


@pytest.fixture(scope="module")
def loaded():
    return sh.load()


@requires_raw
def test_real_data_satisfies_contract(loaded):
    validate_cgm_frame(loaded.cgm)
    validate_event_frame(loaded.events)


@requires_raw
def test_real_data_has_twelve_patients_from_sixteen_files(loaded):
    assert len(loaded.visits) == 16
    assert loaded.cgm["patient_id"].nunique() == 12


@requires_raw
def test_repeat_visits_share_one_patient_id(loaded):
    """1002·1006은 3회씩 내원했지만 환자는 각 1명이다."""
    counts = loaded.visits["patient_id"].value_counts()
    assert counts["shanghai_t1dm_1002"] == 3
    assert counts["shanghai_t1dm_1006"] == 3


@requires_raw
def test_real_data_is_in_mgdl_range(loaded):
    g = loaded.cgm["glucose_mgdl"]
    # 원본 실측 범위는 39.6~475.2 mg/dL (2.2~26.4 mmol/L)
    assert 30 < g.min() < 60
    assert 400 < g.max() < 600
    assert 100 < g.median() < 200


@requires_raw
def test_real_data_source_column(loaded):
    assert set(loaded.cgm["source"].unique()) == {"shanghai_t1dm"}


@requires_raw
def test_every_patient_id_has_dataset_prefix(loaded):
    assert loaded.cgm["patient_id"].str.startswith("shanghai_t1dm_").all()
