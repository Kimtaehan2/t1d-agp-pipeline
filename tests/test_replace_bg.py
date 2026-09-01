"""Replace-BG 로더 테스트.

원본과 똑같은 컬럼 구성의 축소판 파일을 tmp에 만들어 파싱 규칙을 검증하고,
원본이 있을 때만 실데이터 통합 테스트를 돈다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.schema import validate_cgm_frame, validate_event_frame
from src.loaders import replace_bg as rb

RAW_AVAILABLE = (rb.DEFAULT_RAW_DIR / "HDeviceCGM.txt").is_file()
requires_raw = pytest.mark.skipif(
    not RAW_AVAILABLE, reason=f"원본이 없다: {rb.DEFAULT_RAW_DIR}"
)

# 원본 헤더를 그대로 옮겨 적었다. 컬럼 순서까지 동일해야 의미가 있다.
H_CGM = ("RecID|ParentHDeviceUploadsID|PtID|SiteID|DeviceDtTmDaysFromEnroll|DeviceTm"
         "|DexInternalDtTmDaysFromEnroll|DexInternalTm|RecordType|GlucoseValue")
H_BOLUS = ("RecID|ParentHDeviceUploadsID|PtID|SiteID|DeviceDtTmDaysFromEnroll|DeviceTm"
           "|BolusType|InjValue|Insulin|Normal|ExpectedNormal|Extended"
           "|ExpectedExtended|Duration|ExpectedDuration")
H_BASAL = ("RecID|ParentHDeviceUploadsID|PtID|SiteID|DeviceDtTmDaysFromEnroll|DeviceTm"
           "|BasalType|Duration|ExpectedDuration|Percnt|Rate|SuprBasalType"
           "|SuprDuration|SuprRate")
H_WIZARD = ("RecID|ParentHDeviceUploadsID|PtId|SiteID|DeviceDtTmDaysFromEnroll|DeviceTm"
            "|RecommendedCarb|RecommendedCorrection|RecommendedNet|BgInput|CarbInput"
            "|InsulinOnBoard|InsulinCarbRatio|InsulinSensitivity|BgTargetLow"
            "|BgTargetHigh|ParentHDeviceBolusID|BgTargetTarget|BgTargetRange")
H_BGM = ("RecID|ParentHDeviceUploadsID|PtID|SiteID|DeviceDtTmDaysFromEnroll|DeviceTm"
         "|RecordType|RecordSubType|GlucoseValue")
H_ROSTER = ("RecID|PtID|SiteOrig|SiteID|RandDtDaysAfterEnroll|PtStatus|TrtGroup"
            "|AgeAsOfEnrollDt")


@pytest.fixture
def mini(tmp_path):
    """원본 축소판. 경계 사례를 일부러 섞어 놨다."""

    def write(name, header, rows):
        (tmp_path / name).write_text(
            "\n".join([header, *rows]) + "\n", encoding="utf-8", newline="\r\n"
        )

    write(H_CGM and "HDeviceCGM.txt", H_CGM, [
        # 등록일 이전 기록 (min_day=0이면 버려져야 한다)
        "1|1|263|15|-6|05:35:41|-6|12:37:02|CGM|162.00",
        # 정상 5분 간격 3개
        "2|1|263|15|0|00:00:00|0|00:00:00|CGM|100.00",
        "3|1|263|15|0|00:05:00|0|00:05:00|CGM|105.00",
        "4|1|263|15|0|00:10:00|0|00:10:00|CGM|39.00",   # 저혈당 센티널
        # 완전 중복 (같은 시각, 같은 값)
        "5|2|263|15|0|00:05:00|0|00:05:00|CGM|105.00",
        # 값이 다른 중복
        "6|2|263|15|0|00:10:00|0|00:10:00|CGM|401.00",
        # 다른 환자
        "7|3|101|15|1|12:00:00|1|12:00:00|CGM|150.00",
        # 교정값 — CGM이 아니라 이벤트로 가야 한다
        "8|3|101|15|1|12:30:00|1|12:30:00|Calibration|155.00",
    ])
    write("HDeviceBolus.txt", H_BOLUS, [
        "10|1|263|15|0|08:00:00|normal|||3.50|||||",
        "11|1|263|15|0|12:00:00|Normal|||2.00|||||",          # 대문자 변형
        "12|1|263|15|0|18:00:00|dual/square|||1.50||2.50||7200000|",
        "13|1|263|15|-3|08:00:00|normal|||9.90|||||",          # 등록일 이전
    ])
    write("HDeviceBasal.txt", H_BASAL, [
        "20|1|263|15|0|00:00:00|scheduled|10800000|||.650|||",
        "21|1|263|15|0|03:00:00|suspend|3600000|||.000|||",
        "22|1|263|15|0|04:00:00|temp|3600000||50|.325|||",
    ])
    write("HDeviceWizard.txt", H_WIZARD, [
        "30|1|263|15|0|08:00:00|3.50|.00|3.50|8.38|25.00|.00|7.00|2.22|5.55|6.11|10||",
        "31|1|263|15|0|12:00:00|.00|.00|.00|6.10|.00|.00|7.00|2.22|5.55|6.11|11||",
        "32|1|263|15|-3|08:00:00|2.80|.00|2.80|7.00|20.00|.00|7.00|2.22|5.55|6.11|13||",
    ])
    write("HDeviceBGM.txt", H_BGM, [
        "40|1|263|15|0|07:55:00|BGM|manual|137.00",
        "41|1|263|15|0|07:56:00|Ketone||0.30",
        "42|1|263|15|-3|07:55:00|BGM|manual|180.00",
    ])
    write("HPtRoster.txt", H_ROSTER, [
        "50|263|15|15|81|Completed|CGM Only|44",
        "51|101|15|15|61|Completed|CGM+BGM|27",
    ])
    return tmp_path


# --- timestamp 조립 ----------------------------------------------------------


def test_build_timestamp_expands_relative_days():
    ts = rb.build_timestamp(pd.Series([0, 1]), pd.Series(["00:00:00", "12:30:45"]))
    assert ts.iloc[0] == rb.ANCHOR_DATE
    assert ts.iloc[1] == rb.ANCHOR_DATE + pd.Timedelta(days=1, hours=12,
                                                       minutes=30, seconds=45)


def test_build_timestamp_handles_negative_days():
    ts = rb.build_timestamp(pd.Series([-6]), pd.Series(["05:35:41"]))
    assert ts.iloc[0] == rb.ANCHOR_DATE - pd.Timedelta(days=6) + pd.Timedelta(
        hours=5, minutes=35, seconds=41
    )


def test_build_timestamp_returns_nat_for_unparsable_time():
    ts = rb.build_timestamp(pd.Series([0, 0]), pd.Series(["", "not a time"]))
    assert ts.isna().all()


def test_build_timestamp_preserves_time_of_day():
    """AGP는 하루 중 시각만 쓰므로 이것만은 반드시 보존돼야 한다."""
    ts = rb.build_timestamp(pd.Series([37]), pd.Series(["03:15:00"]))
    assert ts.dt.hour.iloc[0] == 3 and ts.dt.minute.iloc[0] == 15


def test_patient_id_format():
    assert rb.patient_id(263) == "replace_bg_263"


def test_wanted_pt_ids_accepts_both_forms():
    assert rb._wanted_pt_ids(["replace_bg_263", "101"]) == {263, 101}
    assert rb._wanted_pt_ids(None) is None


# --- CGM ---------------------------------------------------------------------


def test_cgm_drops_pre_enrollment_by_default(mini):
    cgm, stats = rb.load_cgm(mini, return_stats=True)
    assert stats.dropped_before_min_day == 1
    assert cgm["timestamp"].min() >= rb.ANCHOR_DATE


def test_cgm_keeps_pre_enrollment_when_min_day_is_none(mini):
    cgm, stats = rb.load_cgm(mini, min_day=None, return_stats=True)
    assert stats.dropped_before_min_day == 0
    assert cgm["timestamp"].min() < rb.ANCHOR_DATE


def test_cgm_excludes_calibration_records(mini):
    cgm, stats = rb.load_cgm(mini, return_stats=True)
    assert stats.dropped_calibration == 1
    assert 155.0 not in set(cgm["glucose_mgdl"])


def test_cgm_removes_duplicate_timestamps(mini):
    cgm, stats = rb.load_cgm(mini, return_stats=True)
    assert stats.dropped_duplicate == 2
    validate_cgm_frame(cgm)


def test_cgm_duplicate_resolution_keeps_first(mini):
    """값이 다른 중복은 먼저 온 값을 남긴다 (00:10:00은 39.0과 401.0 두 건)."""
    cgm = rb.load_cgm(mini)
    row = cgm.loc[
        (cgm["patient_id"] == "replace_bg_263")
        & (cgm["timestamp"] == rb.ANCHOR_DATE + pd.Timedelta(minutes=10))
    ]
    assert len(row) == 1
    assert row["glucose_mgdl"].iloc[0] == 39.0


def test_cgm_satisfies_schema(mini):
    validate_cgm_frame(rb.load_cgm(mini))


def test_cgm_patient_filter(mini):
    cgm = rb.load_cgm(mini, patient_ids=["replace_bg_101"])
    assert set(cgm["patient_id"]) == {"replace_bg_101"}


def test_cgm_is_sorted_lexicographically_not_numerically(tmp_path):
    """patient_id는 문자열이라 replace_bg_10이 replace_bg_2보다 앞이다.

    숫자 PtID 순으로 정렬하면 226명 규모에서 스키마 검증이 깨진다.
    """
    (tmp_path / "HDeviceCGM.txt").write_text(
        "\n".join([
            H_CGM,
            "1|1|2|15|0|00:00:00|0|00:00:00|CGM|100.00",
            "2|1|10|15|0|00:00:00|0|00:00:00|CGM|110.00",
            "3|1|100|15|0|00:00:00|0|00:00:00|CGM|120.00",
        ]) + "\n",
        encoding="utf-8", newline="\r\n",
    )
    cgm = rb.load_cgm(tmp_path)
    assert cgm["patient_id"].tolist() == [
        "replace_bg_10", "replace_bg_100", "replace_bg_2"
    ]
    validate_cgm_frame(cgm)


def test_export_keeps_patients_in_contiguous_blocks(tmp_path):
    """환자 묶음 단위로 Parquet에 흘려 쓰므로 블록이 쪼개지면 안 된다."""
    from src.schema import coerce_cgm_frame
    from src.storage import read_parquet

    rows = [H_CGM]
    for pt in (2, 10, 100):
        for minute in range(3):
            rows.append(
                f"{pt}{minute}|1|{pt}|15|0|00:0{minute}:00|0|00:0{minute}:00|CGM|100.00"
            )
    (tmp_path / "HDeviceCGM.txt").write_text(
        "\n".join(rows) + "\n", encoding="utf-8", newline="\r\n"
    )

    out = tmp_path / "cgm.parquet"
    rb.export_cgm_parquet(out, tmp_path, patients_per_batch=2)
    back = coerce_cgm_frame(read_parquet(out))

    assert len(back) == 9
    validate_cgm_frame(back)


def test_sentinel_constants_match_measured_values():
    """스펙의 40/400이 아니라 실측한 39/401이어야 한다."""
    assert rb.LOW_SENTINEL_MGDL == 39.0
    assert rb.HIGH_SENTINEL_MGDL == 401.0


# --- 이벤트 ------------------------------------------------------------------


def test_events_satisfy_schema(mini):
    validate_event_frame(rb.load_events(mini))


def test_meal_comes_from_wizard_carb_input(mini):
    ev = rb.load_events(mini)
    meals = ev[ev["event_type"] == "meal"]
    assert len(meals) == 1                    # CarbInput=0인 행은 식사가 아니다
    assert meals["value"].iloc[0] == 25.0
    assert meals["unit"].iloc[0] == "g"


def test_bolus_type_case_variants_are_normalized(mini):
    ev = rb.load_events(mini)
    bolus = ev[ev["event_type"] == "insulin_bolus"]
    assert sorted(bolus["value"]) == [1.5, 2.0, 3.5]   # -3일 기록은 제외됨
    assert set(bolus["text"]) == {"normal", "dual/square"}


def test_extended_bolus_is_separate_event_with_duration(mini):
    ev = rb.load_events(mini)
    ext = ev[ev["event_type"] == "insulin_bolus_extended"]
    assert len(ext) == 1
    assert ext["value"].iloc[0] == 2.5
    assert "duration_min=120" in ext["text"].iloc[0]   # 7,200,000 ms = 120분


def test_basal_suspend_is_not_a_rate(mini):
    ev = rb.load_events(mini)
    assert (ev["event_type"] == "pump_suspend").sum() == 1
    rates = ev[ev["event_type"] == "insulin_basal_rate"]
    assert sorted(rates["value"]) == [0.325, 0.65]
    assert rates["unit"].iloc[0] == "IU/h"


def test_bgm_and_ketone_are_split_by_record_type(mini):
    ev = rb.load_events(mini)
    cbg = ev[(ev["event_type"] == "cbg") & (ev["text"] != "calibration")]
    ketone = ev[ev["event_type"] == "blood_ketone"]
    assert cbg["value"].tolist() == [137.0]
    assert cbg["unit"].iloc[0] == "mg/dL"
    assert ketone["value"].tolist() == [0.3]
    assert ketone["unit"].iloc[0] == "mmol/L"


def test_calibration_becomes_a_tagged_cbg_event(mini):
    ev = rb.load_events(mini)
    cal = ev[(ev["event_type"] == "cbg") & (ev["text"] == "calibration")]
    assert cal["value"].tolist() == [155.0]


def test_events_respect_min_day(mini):
    default = rb.load_events(mini)
    everything = rb.load_events(mini, min_day=None)
    assert len(everything) > len(default)
    assert default["timestamp"].min() >= rb.ANCHOR_DATE


def test_roster_gets_schema_patient_id(mini):
    roster = rb.load_patients(mini)
    assert set(roster["patient_id"]) == {"replace_bg_263", "replace_bg_101"}


# --- Parquet 내보내기 --------------------------------------------------------


def test_export_round_trips_through_parquet(mini, tmp_path):
    from src.schema import coerce_cgm_frame
    from src.storage import read_parquet

    out = tmp_path / "out" / "cgm.parquet"
    stats = rb.export_cgm_parquet(out, mini, patients_per_batch=1)

    back = coerce_cgm_frame(read_parquet(out))
    expected = rb.load_cgm(mini)

    assert stats.rows_kept == len(expected)
    pd.testing.assert_frame_equal(
        back.sort_values(["patient_id", "timestamp"]).reset_index(drop=True),
        expected,
    )


# --- 실데이터 통합 테스트 ----------------------------------------------------


@pytest.fixture(scope="module")
def two_patients():
    return rb.load_cgm(patient_ids=["replace_bg_263", "replace_bg_101"])


@requires_raw
def test_real_data_satisfies_contract(two_patients):
    validate_cgm_frame(two_patients)


@requires_raw
def test_real_roster_has_226_patients():
    assert len(rb.load_patients()) == 226


@requires_raw
def test_real_data_is_in_mgdl_range(two_patients):
    g = two_patients["glucose_mgdl"]
    assert g.min() >= 39.0 and g.max() <= 401.0
    assert 100 < g.median() < 200


@requires_raw
def test_real_data_is_five_minute_cadence(two_patients):
    one = two_patients[two_patients["patient_id"] == "replace_bg_263"]
    gaps = one["timestamp"].diff().dropna().dt.total_seconds()
    within_jitter = ((gaps >= 298) & (gaps <= 302)).mean()
    assert within_jitter > 0.9
