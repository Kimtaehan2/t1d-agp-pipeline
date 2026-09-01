"""공통 스키마 계약 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from src import schema


def _cgm(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return schema.coerce_cgm_frame(
        pd.DataFrame(
            {
                "patient_id": [r[0] for r in rows],
                "timestamp": [pd.Timestamp(r[1]) for r in rows],
                "glucose_mgdl": [r[2] for r in rows],
                "source": ["test"] * len(rows),
            }
        )
    )


def test_empty_frames_satisfy_contract():
    schema.validate_cgm_frame(schema.empty_cgm_frame())
    schema.validate_event_frame(schema.empty_event_frame())


def test_coerce_enforces_column_order_and_dtypes():
    df = pd.DataFrame(
        {
            "source": ["test"],
            "glucose_mgdl": ["120"],          # 문자열로 들어와도
            "timestamp": ["2021-01-01 00:00"],
            "patient_id": ["test_1"],
        }
    )
    out = schema.coerce_cgm_frame(df)

    assert list(out.columns) == schema.CGM_COLUMNS
    assert str(out["timestamp"].dtype) == "datetime64[ns]"
    assert out["glucose_mgdl"].iloc[0] == 120.0


def test_coerce_converts_microsecond_datetime_to_ns():
    """read_excel은 pandas 3.x에서 datetime64[us]를 돌려준다."""
    df = pd.DataFrame(
        {
            "patient_id": ["test_1"],
            "timestamp": pd.Series(["2021-01-01"]).astype("datetime64[us]"),
            "glucose_mgdl": [100.0],
            "source": ["test"],
        }
    )
    assert str(schema.coerce_cgm_frame(df)["timestamp"].dtype) == "datetime64[ns]"


def test_missing_column_raises():
    with pytest.raises(schema.SchemaError, match="필수 컬럼 누락"):
        schema.coerce_cgm_frame(pd.DataFrame({"patient_id": ["a"]}))


def test_duplicate_patient_timestamp_raises():
    df = _cgm(
        [
            ("test_1", "2021-01-01 00:00", 100.0),
            ("test_1", "2021-01-01 00:00", 110.0),
        ]
    )
    with pytest.raises(schema.SchemaError, match="중복"):
        schema.validate_cgm_frame(df)


def test_same_timestamp_different_patients_is_fine():
    df = _cgm(
        [
            ("test_1", "2021-01-01 00:00", 100.0),
            ("test_2", "2021-01-01 00:00", 110.0),
        ]
    )
    schema.validate_cgm_frame(df)


def test_unsorted_raises():
    df = _cgm(
        [
            ("test_1", "2021-01-01 01:00", 100.0),
            ("test_1", "2021-01-01 00:00", 110.0),
        ]
    )
    with pytest.raises(schema.SchemaError, match="정렬"):
        schema.validate_cgm_frame(df)


def test_out_of_range_glucose_raises():
    df = _cgm([("test_1", "2021-01-01 00:00", 5000.0)])
    with pytest.raises(schema.SchemaError, match="범위"):
        schema.validate_cgm_frame(df)


def test_nan_glucose_is_allowed():
    """결측 처리 단계에서 NaN을 남기는 것이 규칙이므로 NaN 자체는 위반이 아니다."""
    df = _cgm([("test_1", "2021-01-01 00:00", float("nan"))])
    schema.validate_cgm_frame(df)


def test_unknown_event_type_raises():
    df = schema.coerce_event_frame(
        pd.DataFrame(
            {
                "patient_id": ["test_1"],
                "timestamp": [pd.Timestamp("2021-01-01")],
                "event_type": ["telepathy"],
                "value": [1.0],
                "unit": ["IU"],
                "text": [None],
                "source": ["test"],
            }
        )
    )
    with pytest.raises(schema.SchemaError, match="event_type"):
        schema.validate_event_frame(df)


def test_tz_aware_timestamp_raises():
    df = _cgm([("test_1", "2021-01-01 00:00", 100.0)])
    df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Seoul")
    with pytest.raises(schema.SchemaError):
        schema.validate_cgm_frame(df)
