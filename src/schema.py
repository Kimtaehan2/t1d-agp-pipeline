"""공통 스키마 정의 — 팀 계약. 변경 시 팀 합의가 필요하다.

모든 데이터셋 로더는 최종적으로 :data:`CGM_COLUMNS` 형태의 DataFrame을 반환한다.
이벤트(식사·인슐린·운동)는 별도 테이블로 같은 ``patient_id``, ``timestamp`` 키를 쓴다.
"""

from __future__ import annotations

import pandas as pd

# --- CGM 테이블 -------------------------------------------------------------

CGM_COLUMNS: list[str] = ["patient_id", "timestamp", "glucose_mgdl", "source"]

CGM_DTYPES: dict[str, str] = {
    "patient_id": "string",
    "timestamp": "datetime64[ns]",   # tz-naive, 로컬 시간
    "glucose_mgdl": "float64",       # mg/dL로 통일
    "source": "category",
}

# --- 전처리 완료 CGM 테이블 --------------------------------------------------
#
# 리샘플링·결측처리를 마친 뒤에는 공통 4개 컬럼에 플래그 2개가 붙는다.
# 원본 계약(CGM_COLUMNS)은 그대로 두고 여기서 확장만 한다.

PROCESSED_CGM_COLUMNS: list[str] = CGM_COLUMNS + ["is_imputed", "is_capped"]

PROCESSED_CGM_DTYPES: dict[str, str] = {
    **CGM_DTYPES,
    "is_imputed": "bool",  # 선형보간으로 채운 값
    "is_capped": "bool",   # 센서 측정범위에 걸린(검열된) 값
}

# --- 이벤트 테이블 ----------------------------------------------------------

EVENT_COLUMNS: list[str] = [
    "patient_id",
    "timestamp",
    "event_type",
    "value",
    "unit",
    "text",
    "source",
]

EVENT_DTYPES: dict[str, str] = {
    "patient_id": "string",
    "timestamp": "datetime64[ns]",
    "event_type": "category",
    "value": "float64",   # 값이 없는 이벤트(예: 식사 텍스트)는 NaN
    "unit": "string",
    "text": "string",     # 원본 문자열 보존 (파싱 실패 시 추적용)
    "source": "category",
}

#: 허용되는 이벤트 종류. 데이터셋을 추가할 때 여기에 등록한다.
EVENT_TYPES: frozenset[str] = frozenset(
    {
        "meal",               # 식사. value=탄수화물 g (모르면 NaN)
        "insulin_bolus",      # 볼러스 인슐린(즉시분), IU
        "insulin_bolus_extended",  # 연장 볼러스분, IU. 지속시간은 text에 기록
        "insulin_basal_rate", # 기저 인슐린 주입률, IU/h
        "insulin_sc",         # 피하주사 인슐린 (펜/주사기), IU
        "insulin_iv",         # 정맥 인슐린, IU
        "oral_agent",         # 경구 혈당강하제, mg
        "pump_suspend",       # 펌프 일시 중단
        "cbg",                # 자가혈당측정(지문채혈), mg/dL
        "blood_ketone",       # 혈중 케톤, mmol/L
        "exercise",           # 운동
    }
)

# --- 생리학적 범위 ----------------------------------------------------------

#: 이 범위를 벗어난 혈당은 센서/기록 오류로 보고 검증에서 걸러낸다.
GLUCOSE_MIN_MGDL = 10.0
GLUCOSE_MAX_MGDL = 1000.0


class SchemaError(ValueError):
    """DataFrame이 공통 스키마 계약을 만족하지 않을 때 발생."""


def empty_cgm_frame() -> pd.DataFrame:
    """스키마를 만족하는 빈 CGM DataFrame."""
    return coerce_cgm_frame(pd.DataFrame({c: [] for c in CGM_COLUMNS}))


def empty_processed_cgm_frame() -> pd.DataFrame:
    """스키마를 만족하는 빈 전처리 CGM DataFrame."""
    return coerce_processed_cgm_frame(
        pd.DataFrame({c: [] for c in PROCESSED_CGM_COLUMNS})
    )


def empty_event_frame() -> pd.DataFrame:
    """스키마를 만족하는 빈 이벤트 DataFrame."""
    return coerce_event_frame(pd.DataFrame({c: [] for c in EVENT_COLUMNS}))


def coerce_cgm_frame(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼 순서와 dtype을 계약에 맞춰 강제한다."""
    return _coerce(df, CGM_COLUMNS, CGM_DTYPES)


def coerce_event_frame(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼 순서와 dtype을 계약에 맞춰 강제한다."""
    return _coerce(df, EVENT_COLUMNS, EVENT_DTYPES)


def coerce_processed_cgm_frame(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼 순서와 dtype을 계약에 맞춰 강제한다."""
    return _coerce(df, PROCESSED_CGM_COLUMNS, PROCESSED_CGM_DTYPES)


def _coerce(df: pd.DataFrame, columns: list[str], dtypes: dict[str, str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(f"필수 컬럼 누락: {missing}")
    out = df.loc[:, columns].copy()
    for col, dtype in dtypes.items():
        if dtype.startswith("datetime64"):
            # read_excel은 pandas 3.x에서 datetime64[us]를 돌려주므로 [ns]로 맞춘다.
            out[col] = pd.to_datetime(out[col]).astype("datetime64[ns]")
        else:
            out[col] = out[col].astype(dtype)
    return out.reset_index(drop=True)


def validate_cgm_frame(df: pd.DataFrame) -> None:
    """CGM DataFrame이 계약을 지키는지 검사한다. 위반 시 :class:`SchemaError`."""
    _validate_cgm_like(df, CGM_COLUMNS, CGM_DTYPES)


def validate_processed_cgm_frame(df: pd.DataFrame) -> None:
    """전처리 완료 CGM DataFrame을 검사한다."""
    _validate_cgm_like(df, PROCESSED_CGM_COLUMNS, PROCESSED_CGM_DTYPES)

    # 보간했다고 표시해 놓고 값이 비어 있으면 앞뒤가 안 맞는다.
    broken = df["is_imputed"] & df["glucose_mgdl"].isna()
    if broken.any():
        raise SchemaError(f"is_imputed=True인데 glucose_mgdl이 NaN인 행 {int(broken.sum())}개")

    capped_empty = df["is_capped"] & df["glucose_mgdl"].isna()
    if capped_empty.any():
        raise SchemaError(
            f"is_capped=True인데 glucose_mgdl이 NaN인 행 {int(capped_empty.sum())}개"
        )


def _validate_cgm_like(
    df: pd.DataFrame, columns: list[str], dtypes: dict[str, str]
) -> None:
    _validate_common(df, columns, dtypes)

    if df["timestamp"].isna().any():
        raise SchemaError("timestamp에 NaT가 있다")

    g = df["glucose_mgdl"]
    bad = g.notna() & ((g < GLUCOSE_MIN_MGDL) | (g > GLUCOSE_MAX_MGDL))
    if bad.any():
        raise SchemaError(
            f"생리학적 범위({GLUCOSE_MIN_MGDL}~{GLUCOSE_MAX_MGDL} mg/dL)를 "
            f"벗어난 값 {int(bad.sum())}개"
        )

    dup = df.duplicated(subset=["patient_id", "timestamp"])
    if dup.any():
        raise SchemaError(f"(patient_id, timestamp) 중복 {int(dup.sum())}건")

    if not df.sort_values(["patient_id", "timestamp"]).index.equals(df.index):
        raise SchemaError("(patient_id, timestamp) 오름차순으로 정렬돼 있지 않다")


def validate_event_frame(df: pd.DataFrame) -> None:
    """이벤트 DataFrame이 계약을 지키는지 검사한다."""
    _validate_common(df, EVENT_COLUMNS, EVENT_DTYPES)

    if df["timestamp"].isna().any():
        raise SchemaError("timestamp에 NaT가 있다")

    unknown = set(df["event_type"].dropna().unique()) - EVENT_TYPES
    if unknown:
        raise SchemaError(f"EVENT_TYPES에 등록되지 않은 event_type: {sorted(unknown)}")


def _validate_common(df: pd.DataFrame, columns: list[str], dtypes: dict[str, str]) -> None:
    if list(df.columns) != columns:
        raise SchemaError(f"컬럼이 계약과 다르다: {list(df.columns)} != {columns}")

    for col, expected in dtypes.items():
        actual = str(df[col].dtype)
        if expected.startswith("datetime64"):
            if actual != "datetime64[ns]":
                raise SchemaError(f"{col} dtype이 datetime64[ns]가 아니다: {actual}")
            if getattr(df[col].dtype, "tz", None) is not None:
                raise SchemaError(f"{col}은 tz-naive여야 한다")
        elif actual != expected:
            raise SchemaError(f"{col} dtype이 {expected}가 아니다: {actual}")

    if df["patient_id"].isna().any():
        raise SchemaError("patient_id에 결측이 있다")
