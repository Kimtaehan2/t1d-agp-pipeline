"""ShanghaiT1DM 로더.

원본: 환자별 Excel 파일 1개 = 내원 1회. 파일명은 ``{환자번호}_{내원차수}_{시작일}``.
같은 환자가 여러 번 내원하므로(1002·1006은 각 3회) ``patient_id``는 파일이 아니라
**환자번호** 기준으로 묶는다. 시점 단위가 아닌 환자 단위 분할을 위해 중요하다.

원본 구조 실측 결과는 ``docs/schema_notes.md`` 참고.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.preprocess import MGDL, infer_glucose_unit, sensor_limits, to_mgdl
from src.schema import (
    SchemaError,
    coerce_cgm_frame,
    coerce_event_frame,
    empty_event_frame,
    validate_cgm_frame,
    validate_event_frame,
)

DATASET = "shanghai_t1dm"
DEFAULT_RAW_DIR = Path("data/raw/shanghai/Shanghai_T1DM")

#: 센서 측정범위. 값이 전부 0.1 mmol/L 배수라 하한 2.2 mmol/L = 39.6,
#: 상한 27.8 mmol/L = 500.4다. Dexcom(40/400)과 다르므로 하드코딩하면 안 된다.
#: 값의 단일 출처는 ``src.preprocess.SENSOR_LIMITS``다.
LOW_SENTINEL_MGDL, HIGH_SENTINEL_MGDL = sensor_limits(DATASET)

#: FreeStyle Libre. 전 파일 15분 고정.
CGM_INTERVAL_MIN = 15

# --- 원본 컬럼명 (16개 파일 전수 확인 결과 모두 동일) -------------------------

COL_DATE = "Date"
COL_CGM = "CGM (mg / dl)"
COL_CBG = "CBG (mg / dl)"
COL_KETONE = "Blood Ketone (mmol / L)"
COL_DIET_EN = "Dietary intake"
COL_DIET_ZH = "饮食"  # 饮食 (식사)
COL_INSULIN_SC = "Insulin dose - s.c."
COL_ORAL = "Non-insulin hypoglycemic agents"
COL_CSII_BOLUS = "CSII - bolus insulin (Novolin R, IU)"
COL_CSII_BASAL = "CSII - basal insulin (Novolin R, IU / H)"
COL_INSULIN_IV = "Insulin dose - i.v."

REQUIRED_COLUMNS = [
    COL_DATE, COL_CGM, COL_CBG, COL_KETONE, COL_DIET_EN, COL_DIET_ZH,
    COL_INSULIN_SC, COL_ORAL, COL_CSII_BOLUS, COL_CSII_BASAL, COL_INSULIN_IV,
]

# --- 결측/센티널 문자열 ------------------------------------------------------

#: 원본에서 "기록 없음"을 뜻하는 문자열들.
#: "未记录"(未记录)는 '기록 없음'의 중국어 표기.
_PLACEHOLDERS = {
    "data not available",
    "未记录",
    "",
    "-",
    "/",
    "nan",
    "none",
}

#: 펌프 일시 중단은 용량이 아니라 상태라서 숫자 컬럼에 문자열로 들어온다.
_SUSPEND = "temporarily suspend insulin delivery"

_FILE_STEM = re.compile(r"^(?P<pid>\d+)_(?P<visit>\d+)_(?P<date>\d{8})$")
_RE_INSULIN_SC = re.compile(r"^(?P<name>.+?),\s*(?P<dose>\d+(?:\.\d+)?)\s*IU$", re.I)
_RE_ORAL = re.compile(r"^(?P<name>[^,\d]+?),?\s*(?P<dose>\d+(?:\.\d+)?)\s*mg$", re.I)
_RE_IV_IU = re.compile(r"(?P<dose>\d+(?:\.\d+)?)\s*IU", re.I)


@dataclass(frozen=True)
class ShanghaiT1DM:
    """로더 결과 묶음.

    Attributes:
        cgm: 공통 스키마 CGM 테이블.
        events: 공통 스키마 이벤트 테이블.
        visits: 내원 단위 메타데이터. 환자별 내원 사이의 긴 공백을 구간으로
            나눠 처리해야 하므로(1002는 5월·9월로 4개월 떨어져 있다) 별도로 둔다.
    """

    cgm: pd.DataFrame
    events: pd.DataFrame
    visits: pd.DataFrame


def load(raw_dir: str | Path | None = None, validate: bool = True) -> ShanghaiT1DM:
    """ShanghaiT1DM 원본 Excel을 읽어 공통 스키마로 변환한다."""
    raw_dir = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    files = find_files(raw_dir)

    cgm_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    visit_rows: list[dict] = []

    for path in files:
        patient_id, visit = parse_filename(path)
        raw = _read_one(path)

        cgm = _extract_cgm(raw, patient_id)
        cgm_parts.append(cgm)
        event_parts.append(_extract_events(raw, patient_id))

        visit_rows.append(
            {
                "patient_id": patient_id,
                "visit": visit,
                "file": path.name,
                "start": cgm["timestamp"].min(),
                "end": cgm["timestamp"].max(),
                "n_cgm": len(cgm),
            }
        )

    cgm_df = _finalize_cgm(cgm_parts)
    events_df = _finalize_events(event_parts)

    visits_df = (
        pd.DataFrame(visit_rows)
        .sort_values(["patient_id", "visit"])
        .reset_index(drop=True)
    )
    visits_df["days"] = (
        visits_df["end"] - visits_df["start"]
    ).dt.total_seconds() / 86400.0

    if validate:
        validate_cgm_frame(cgm_df)
        validate_event_frame(events_df)

    return ShanghaiT1DM(cgm=cgm_df, events=events_df, visits=visits_df)


def load_cgm(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """공통 스키마 CGM 테이블만 필요할 때 쓰는 얇은 래퍼."""
    return load(raw_dir).cgm


def find_files(raw_dir: str | Path) -> list[Path]:
    """원본 Excel 경로 목록. Excel 임시 잠금 파일(``~$*``)은 제외한다."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"원본 디렉터리가 없다: {raw_dir}")

    files = [p for p in sorted(raw_dir.glob("*.xls*")) if not p.name.startswith("~$")]
    if not files:
        raise FileNotFoundError(f"{raw_dir}에 .xls/.xlsx 파일이 없다")
    return files


def parse_filename(path: str | Path) -> tuple[str, int]:
    """``1002_1_20210521.xls`` -> ``("shanghai_t1dm_1002", 1)``.

    파일명의 날짜는 실제 첫 기록 시각과 다를 수 있으므로(1010은 파일명 0915,
    첫 기록 09-14) 타임스탬프 산출에 쓰지 않는다.
    """
    stem = Path(path).stem
    m = _FILE_STEM.match(stem)
    if m is None:
        raise ValueError(f"예상과 다른 파일명 형식: {stem}")
    return f"{DATASET}_{m['pid']}", int(m["visit"])


# --- 내부 구현 ---------------------------------------------------------------


def _read_one(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{path.name}: 원본 컬럼 누락 {missing}")
    return df


def _extract_cgm(raw: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    values = pd.to_numeric(raw[COL_CGM], errors="coerce")

    # 규칙: 단위를 하드코딩하지 않고 파일마다 확인한다.
    unit = infer_glucose_unit(values)
    if unit != MGDL:
        values = to_mgdl(values, unit)

    out = pd.DataFrame(
        {
            "patient_id": patient_id,
            "timestamp": pd.to_datetime(raw[COL_DATE]),
            "glucose_mgdl": values.to_numpy(),
            "source": DATASET,
        }
    )
    return out.loc[out["timestamp"].notna() & out["glucose_mgdl"].notna()]


def _finalize_cgm(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["patient_id", "timestamp"], kind="stable")
    # 같은 환자의 내원 구간은 겹치지 않지만, 겹칠 경우 첫 값을 남긴다.
    df = df.drop_duplicates(subset=["patient_id", "timestamp"], keep="first")
    return coerce_cgm_frame(df)


def _finalize_events(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if not p.empty]
    if not parts:
        return empty_event_frame()
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["patient_id", "timestamp", "event_type"], kind="stable")
    return coerce_event_frame(df)


def _clean_text(value: object) -> str | None:
    """공백/센티널을 정리해 실제 내용이 있으면 문자열, 없으면 ``None``."""
    if value is None:
        return None
    if not isinstance(value, str) and pd.isna(value):
        return None
    text = str(value).replace("\xa0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if text.lower() in _PLACEHOLDERS or text in _PLACEHOLDERS:
        return None
    return text


def _event_rows(raw: pd.DataFrame, patient_id: str, column: str, builder) -> list[dict]:
    """한 원본 컬럼을 이벤트 행 리스트로 바꾼다. ``builder``는 dict 또는 None 반환."""
    rows: list[dict] = []
    timestamps = pd.to_datetime(raw[COL_DATE])
    for ts, value in zip(timestamps, raw[column]):
        if pd.isna(ts):
            continue
        built = builder(value)
        if built is None:
            continue
        rows.append({"patient_id": patient_id, "timestamp": ts, **built})
    return rows


def _numeric_builder(event_type: str, unit: str):
    """숫자 컬럼용. 펌프 중단 문자열은 별도 이벤트로 바꾼다."""

    def build(value: object) -> dict | None:
        text = _clean_text(value)
        if text is None:
            return None
        if text.lower() == _SUSPEND:
            return {
                "event_type": "pump_suspend",
                "value": float("nan"),
                "unit": None,
                "text": text,
            }
        number = pd.to_numeric(text, errors="coerce")
        if pd.isna(number):
            return None
        return {
            "event_type": event_type,
            "value": float(number),
            "unit": unit,
            "text": text,
        }

    return build


def _meal_builder(value: object) -> dict | None:
    text = _clean_text(value)
    if text is None:
        return None
    # 원본은 음식별 '중량(g)'만 주고 탄수화물 g은 없으므로 value는 NaN으로 둔다.
    return {"event_type": "meal", "value": float("nan"), "unit": None, "text": text}


def _insulin_sc_builder(value: object) -> dict | None:
    text = _clean_text(value)
    if text is None:
        return None
    m = _RE_INSULIN_SC.match(text)
    dose = float(m["dose"]) if m else float("nan")
    return {"event_type": "insulin_sc", "value": dose, "unit": "IU", "text": text}


def _insulin_iv_builder(value: object) -> dict | None:
    text = _clean_text(value)
    if text is None:
        return None
    m = _RE_IV_IU.search(text)
    dose = float(m["dose"]) if m else float("nan")
    return {"event_type": "insulin_iv", "value": dose, "unit": "IU", "text": text}


def _oral_builder(value: object) -> dict | None:
    text = _clean_text(value)
    if text is None:
        return None
    m = _RE_ORAL.match(text)
    dose = float(m["dose"]) if m else float("nan")
    return {"event_type": "oral_agent", "value": dose, "unit": "mg", "text": text}


def _extract_events(raw: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    rows: list[dict] = []

    # 식사: 영문 컬럼을 기본으로 쓰고, 비어 있으면 중문 컬럼으로 보완한다.
    meals: dict[pd.Timestamp, dict] = {}
    for column in (COL_DIET_EN, COL_DIET_ZH):
        for row in _event_rows(raw, patient_id, column, _meal_builder):
            meals.setdefault(row["timestamp"], row)
    rows.extend(meals.values())

    rows += _event_rows(
        raw, patient_id, COL_CSII_BOLUS, _numeric_builder("insulin_bolus", "IU")
    )
    rows += _event_rows(
        raw, patient_id, COL_CSII_BASAL, _numeric_builder("insulin_basal_rate", "IU/h")
    )
    rows += _event_rows(raw, patient_id, COL_CBG, _numeric_builder("cbg", "mg/dL"))
    rows += _event_rows(
        raw, patient_id, COL_KETONE, _numeric_builder("blood_ketone", "mmol/L")
    )
    rows += _event_rows(raw, patient_id, COL_INSULIN_SC, _insulin_sc_builder)
    rows += _event_rows(raw, patient_id, COL_INSULIN_IV, _insulin_iv_builder)
    rows += _event_rows(raw, patient_id, COL_ORAL, _oral_builder)

    if not rows:
        return empty_event_frame()

    df = pd.DataFrame(rows)
    df["source"] = DATASET
    return df
