"""T1D-UOM 로더 (University of Manchester, CC BY 4.0).

원본: 환자별 CSV 1개 = 데이터 종류 1개. ``UoM{종류}{환자번호}.csv`` 형식이고
혈당·기저·볼러스·식사·활동·수면 폴더로 나뉜다. 17명 · 2023-09 ~ 2024-09.

이 로더가 **원본 README를 그대로 믿지 않는 지점이 세 군데** 있다. 전부 실측으로
뒤집은 것이며 근거는 ``docs/schema_notes.md``에 적어 뒀다.

1. **날짜는 ``DD/MM/YYYY``다.** README 데이터 사전은 ``MM/DD/YYYY``라고 적어 놨지만
   ``22/10/2023`` 같은 값이 있고 두 번째 필드가 12를 넘는 행은 전 파일에 0건이다.
   월-일을 뒤집어 읽으면 두 필드가 모두 12 이하인 날짜에서 **조용히** 어긋나므로
   :data:`STUDY_START` / :data:`STUDY_END` 범위로 파싱 결과를 검증한다.
2. **치료 방식 컬럼이 없다.** README가 광고하는 ``Demographics/`` 폴더가 배포본에
   없어서 기저 기록의 *모양*으로 추론했다. :data:`MODALITY` 참고.
3. **폐루프(AID) 환자가 섞여 있다.** 2301·2307은 기저가 5분마다 1,000~2,000개의
   서로 다른 값으로 바뀐다. 사람이 손으로 넣는 값이 아니다. 수동관리 분석에
   섞이면 안 되므로 기본값에서 제외한다.

원본 구조 실측 결과는 ``docs/schema_notes.md`` 참고.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.preprocess import MGDL, infer_glucose_unit, sensor_limits, to_mgdl
from src.schema import (
    GLUCOSE_MAX_MGDL,
    GLUCOSE_MIN_MGDL,
    SchemaError,
    coerce_cgm_frame,
    coerce_event_frame,
    empty_event_frame,
    validate_cgm_frame,
    validate_event_frame,
)

DATASET = "t1d_uom"
DEFAULT_RAW_DIR = Path(
    "data/raw/ManchesterCSCoordinatedDiabetesStudy-V1.0.1/"
    "sharpic-ManchesterCSCoordinatedDiabetesStudy-a9e8025"
)

#: 센서 측정범위(mg/dL). 값의 단일 출처는 ``src.preprocess.SENSOR_LIMITS``다.
#:
#: 주의: **이 데이터셋은 센서가 한 종류가 아니다.** 상한이 27.8 mmol/L(=500.9)인
#: 환자와 22.2 mmol/L(=400.0)인 환자가 섞여 있다. 여기 등록된 값은 넓은 쪽이라
#: 22.2에서 검열된 환자(2309: 22.2가 207건)의 캡핑은 잡히지 않는다.
LOW_SENTINEL_MGDL, HIGH_SENTINEL_MGDL = sensor_limits(DATASET)

#: 파싱 결과 검증용 범위. 실측 관측 구간(2023-09-04 ~ 2024-09-03)을 넉넉히 감싼다.
#: 월-일이 뒤집힌 날짜는 이 범위를 크게 벗어나므로 여기서 걸린다.
STUDY_START = pd.Timestamp("2023-09-01")
STUDY_END = pd.Timestamp("2024-10-01")

#: 치료 방식. **원본에 없는 정보이고 기저 기록의 모양에서 추론한 것이다.**
#:
#:   MDI          ``insulin_kind=L``, 하루 1~2건, 서로 다른 용량 9개 이하.
#:                지속형 인슐린 1일 1회 주사.
#:   PUMP_OPEN    ``insulin_kind=R``, 하루 4~8건, 서로 다른 용량 3~9개.
#:                손으로 짜 넣은 기저 프로파일 구간.
#:   CLOSED_LOOP  ``insulin_kind=R``, 하루 120~157건(≈5분마다),
#:                서로 다른 용량 1,034~1,974개. 알고리즘이 조절한 것.
#:   UNKNOWN      기저 파일 자체가 없어 분류 불가.
MODALITY: dict[str, str] = {
    "2301": "CLOSED_LOOP",
    "2302": "MDI",
    "2303": "UNKNOWN",
    "2304": "PUMP_OPEN",
    "2305": "MDI",
    "2306": "MDI",
    "2307": "CLOSED_LOOP",
    "2308": "PUMP_OPEN",
    "2309": "PUMP_OPEN",
    "2310": "PUMP_OPEN",
    "2313": "MDI",
    "2314": "MDI",
    "2320": "UNKNOWN",
    "2401": "MDI",
    "2403": "MDI",
    "2404": "UNKNOWN",
    "2405": "MDI",
}

#: 수동관리(사람이 용량을 결정하는) 환자. 기본 대상이다.
MANUAL_MODALITIES: frozenset[str] = frozenset({"MDI", "PUMP_OPEN"})

# --- 원본 폴더/컬럼 ----------------------------------------------------------

_DIR_GLUCOSE = "Glucose Data"
_DIR_BASAL = "Insulin Data/Basal Data"
_DIR_BOLUS = "Insulin Data/Bolus Data"
_DIR_NUTRITION = "Nutrition Data"
_DIR_ACTIVITY = "Activity Data"

_TABLES: dict[str, tuple[str, str, str]] = {
    # 종류: (폴더, 파일명 접두, 타임스탬프 컬럼)
    "glucose": (_DIR_GLUCOSE, "UoMGlucose", "bg_ts"),
    "basal": (_DIR_BASAL, "UoMBasal", "basal_ts"),
    "bolus": (_DIR_BOLUS, "UoMBolus", "bolus_ts"),
    "nutrition": (_DIR_NUTRITION, "UoMNutrition", "meal_ts"),
    "activity": (_DIR_ACTIVITY, "UoMActivity", "activity_ts"),
}

#: 지속형(Long-acting) 기저는 펌프 주입률이 아니라 **주사**다. 공통 스키마에서
#: ``insulin_basal_rate``(IU/h)가 아니라 ``insulin_sc``(IU)로 가야 맞는다.
_KIND_LONG_ACTING = "L"
_KIND_RAPID = "R"


@dataclass(frozen=True)
class T1DUOM:
    """로더 결과 묶음.

    Attributes:
        cgm: 공통 스키마 CGM 테이블.
        events: 공통 스키마 이벤트 테이블.
        patients: 환자별 메타데이터(치료 방식, 센서 주기, 관측 구간, 행 수).
            치료 방식이 추론값이라 어떤 근거로 걸렀는지 남겨 둔다.
    """

    cgm: pd.DataFrame
    events: pd.DataFrame
    patients: pd.DataFrame


def load(
    raw_dir: str | Path | None = None,
    modalities: frozenset[str] | set[str] | None = None,
    validate: bool = True,
) -> T1DUOM:
    """T1D-UOM 원본 CSV를 읽어 공통 스키마로 변환한다.

    Args:
        modalities: 포함할 치료 방식. 기본값 ``None``은
            :data:`MANUAL_MODALITIES`(MDI + 개방루프 펌프)를 뜻한다.
            폐루프와 분류 불가 환자를 넣으려면 명시적으로 지정해야 한다.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"원본 디렉터리가 없다: {raw_dir}")

    wanted = frozenset(modalities) if modalities is not None else MANUAL_MODALITIES
    unknown = wanted - set(MODALITY.values())
    if unknown:
        raise ValueError(f"알 수 없는 치료 방식: {sorted(unknown)}")

    pids = [p for p in find_patients(raw_dir) if MODALITY.get(p) in wanted]
    if not pids:
        raise SchemaError(f"{sorted(wanted)}에 해당하는 환자가 없다")

    cgm_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    meta_rows: list[dict] = []

    for pid in pids:
        cgm = _extract_cgm(raw_dir, pid)
        cgm_parts.append(cgm)
        event_parts.append(_extract_events(raw_dir, pid))

        meta_rows.append(
            {
                "patient_id": patient_id(pid),
                "modality": MODALITY[pid],
                "sensor_interval_min": _sensor_interval(cgm["timestamp"]),
                "start": cgm["timestamp"].min(),
                "end": cgm["timestamp"].max(),
                "n_cgm": len(cgm),
            }
        )

    cgm_df = _finalize_cgm(cgm_parts)
    events_df = _finalize_events(event_parts)

    patients_df = pd.DataFrame(meta_rows).sort_values("patient_id").reset_index(drop=True)
    patients_df["days"] = (
        patients_df["end"] - patients_df["start"]
    ).dt.total_seconds() / 86400.0

    if validate:
        validate_cgm_frame(cgm_df)
        validate_event_frame(events_df)

    return T1DUOM(cgm=cgm_df, events=events_df, patients=patients_df)


def load_cgm(
    raw_dir: str | Path | None = None,
    modalities: frozenset[str] | set[str] | None = None,
) -> pd.DataFrame:
    """공통 스키마 CGM 테이블만 필요할 때 쓰는 얇은 래퍼."""
    return load(raw_dir, modalities).cgm


def find_patients(raw_dir: str | Path) -> list[str]:
    """혈당 파일이 있는 환자번호 목록. ``["2301", "2302", ...]``."""
    folder = Path(raw_dir) / _DIR_GLUCOSE
    if not folder.is_dir():
        raise FileNotFoundError(f"혈당 폴더가 없다: {folder}")

    pids = sorted(p.stem.replace("UoMGlucose", "") for p in folder.glob("UoMGlucose*.csv"))
    if not pids:
        raise FileNotFoundError(f"{folder}에 UoMGlucose*.csv가 없다")

    unregistered = [p for p in pids if p not in MODALITY]
    if unregistered:
        raise SchemaError(
            f"MODALITY에 등록되지 않은 환자: {unregistered}. "
            f"기저 기록 모양을 실측해 치료 방식을 판정한 뒤 추가해라."
        )
    return pids


def patient_id(pid: str) -> str:
    """``"2302"`` -> ``"t1d_uom_2302"``."""
    return f"{DATASET}_{pid}"


# --- 내부 구현 ---------------------------------------------------------------


def _read_table(raw_dir: Path, kind: str, pid: str) -> pd.DataFrame | None:
    """원본 CSV 한 장을 읽어 타임스탬프를 파싱한다. 파일이 없으면 ``None``.

    BOM(``utf-8-sig``)과 빈 꼬리 컬럼(``UoMBasal2301.csv``에 2개)이 섞여 있어
    둘 다 여기서 정리한다.
    """
    folder, prefix, tscol = _TABLES[kind]
    path = raw_dir / folder / f"{prefix}{pid}.csv"
    if not path.is_file():
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    if tscol not in df.columns:
        raise SchemaError(f"{path.name}: 타임스탬프 컬럼 {tscol}이 없다")

    # 규칙: 원본 README가 MM/DD라고 적어 놨어도 day-first로 읽는다.
    df[tscol] = pd.to_datetime(df[tscol], dayfirst=True, errors="coerce")
    df = df.loc[df[tscol].notna()]

    # 월-일이 뒤집혔거나 원본이 손상된 행을 여기서 버린다. 연구 기간 밖 기록은
    # 실제로 존재한다(2302 볼러스 2023-01, 2403 볼러스 2023-03, 2314 식사 2024-12).
    inside = df[tscol].between(STUDY_START, STUDY_END)
    df = df.loc[inside]

    return df.sort_values(tscol).reset_index(drop=True)


def _sensor_interval(timestamps: pd.Series) -> int:
    """센서 측정 주기(분). 최빈 간격으로 본다. 이 데이터셋은 5분과 15분이 섞여 있다."""
    gaps = timestamps.diff().dt.total_seconds().div(60).round()
    gaps = gaps[gaps > 0]
    return int(gaps.mode().iloc[0]) if not gaps.empty else 0


def _extract_cgm(raw_dir: Path, pid: str) -> pd.DataFrame:
    raw = _read_table(raw_dir, "glucose", pid)
    if raw is None or raw.empty:
        raise SchemaError(f"{pid}: 혈당 파일이 비어 있다")

    values = pd.to_numeric(raw["value"], errors="coerce")

    # 규칙: 단위를 하드코딩하지 않고 파일마다 확인한다.
    # 이 데이터셋은 전 환자 mmol/L이지만(중앙값 6.3~9.8) 확인은 한다.
    unit = infer_glucose_unit(values)
    if unit != MGDL:
        values = to_mgdl(values, unit)

    out = pd.DataFrame(
        {
            "patient_id": patient_id(pid),
            "timestamp": raw["bg_ts"],
            "glucose_mgdl": values.to_numpy(),
            "source": DATASET,
        }
    )
    out = out.loc[out["glucose_mgdl"].notna()]

    # 생리학적으로 불가능한 값은 버린다. 2307에 0.1 mmol/L(=1.8 mg/dL)가 있다.
    plausible = out["glucose_mgdl"].between(GLUCOSE_MIN_MGDL, GLUCOSE_MAX_MGDL)
    return out.loc[plausible]


def _finalize_cgm(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["patient_id", "timestamp"], kind="stable")
    df = df.drop_duplicates(subset=["patient_id", "timestamp"], keep="first")
    return coerce_cgm_frame(df)


def _finalize_events(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return empty_event_frame()
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["patient_id", "timestamp", "event_type"], kind="stable")
    return coerce_event_frame(df)


def _events_frame(
    pid: str,
    timestamps: pd.Series,
    event_type: str,
    values,
    unit: str | None,
    text=None,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "patient_id": patient_id(pid),
            "timestamp": pd.Series(timestamps).reset_index(drop=True),
            "event_type": event_type,
            "value": pd.Series(values, dtype="float64").reset_index(drop=True),
            "unit": unit,
            "text": pd.Series(text).reset_index(drop=True) if text is not None else None,
            "source": DATASET,
        }
    )
    return df.loc[df["timestamp"].notna()]


def _join_text(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """여러 원본 컬럼을 ``"Lunch; Huel+Peanutbutter"`` 형태 한 줄로 합친다.

    원본 문자열을 버리지 않고 ``text``에 남겨 두기 위한 것이다. 전부 비어 있으면
    ``None``으로 둔다.
    """
    present = [c for c in columns if c in df.columns]
    if not present:
        return pd.Series([None] * len(df), index=df.index, dtype="object")

    joined = (
        df[present]
        .astype("string")
        .fillna("")
        .apply(lambda row: "; ".join(v for v in row if v.strip()), axis=1)
    )
    return joined.replace("", None)


def _extract_events(raw_dir: Path, pid: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    # --- 식사 -------------------------------------------------------------
    # 자기보고다. 시각이 정시로 반올림된 기록이 많고 탄수화물 g는 대체로 채워져
    # 있다(후보 창 기준 96~100%). meal_type/meal_tag는 text에 붙여 보존한다.
    meals = _read_table(raw_dir, "nutrition", pid)
    if meals is not None and not meals.empty:
        carbs = pd.to_numeric(meals.get("carbs_g"), errors="coerce")
        text = _join_text(meals, ["meal_type", "meal_tag"])
        parts.append(_events_frame(pid, meals["meal_ts"], "meal", carbs, "g", text))

    # --- 볼러스 -----------------------------------------------------------
    # 원본에 볼러스 종류(식사분/교정분)가 없다. 구분해서 쓸 수 없다.
    bolus = _read_table(raw_dir, "bolus", pid)
    if bolus is not None and not bolus.empty:
        dose = pd.to_numeric(bolus["bolus_dose"], errors="coerce")
        keep = dose.notna()
        if keep.any():
            parts.append(
                _events_frame(
                    pid, bolus.loc[keep, "bolus_ts"], "insulin_bolus", dose.loc[keep], "IU"
                )
            )

    # --- 기저 -------------------------------------------------------------
    # insulin_kind에 따라 이벤트 종류가 갈린다. L(지속형)은 하루 1~2회 주사라
    # 주입률이 아니라 피하주사이고, R(속효성)은 펌프 기저 주입률이다.
    basal = _read_table(raw_dir, "basal", pid)
    if basal is not None and not basal.empty:
        dose = pd.to_numeric(basal["basal_dose"], errors="coerce")
        kind = basal.get("insulin_kind", pd.Series("", index=basal.index)).astype("str")

        is_long = kind.str.strip().str.upper() == _KIND_LONG_ACTING
        for mask, event_type, unit in (
            (is_long & dose.notna(), "insulin_sc", "IU"),
            (~is_long & dose.notna(), "insulin_basal_rate", "IU/h"),
        ):
            if mask.any():
                parts.append(
                    _events_frame(
                        pid,
                        basal.loc[mask, "basal_ts"],
                        event_type,
                        dose.loc[mask],
                        unit,
                        kind.loc[mask],
                    )
                )

    if not parts:
        return empty_event_frame()
    return pd.concat(parts, ignore_index=True)


__all__ = [
    "DATASET",
    "DEFAULT_RAW_DIR",
    "MANUAL_MODALITIES",
    "MODALITY",
    "STUDY_END",
    "STUDY_START",
    "T1DUOM",
    "find_patients",
    "load",
    "load_cgm",
    "patient_id",
]
