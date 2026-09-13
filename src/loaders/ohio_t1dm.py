"""OhioT1DM 로더 (Ohio University, Marling & Bunescu 2018/2020).

원본: 환자당 XML 두 장(``{id}-ws-training.xml``, ``{id}-ws-testing.xml``). 2018
코호트 6명과 2020 코호트 6명, 환자당 약 8주. ``<patient>`` 아래에 데이터 종류별
섹션이 있고 각 섹션은 ``<event .../>`` 목록이다.

이 데이터셋의 가치는 CGM이 아니라 **자기보고 이벤트 로그**다. 운동(강도·지속시간),
저혈당 사건, 질병, 스트레스, 수면이 참가자 입력으로 들어 있다. T1D-UOM에는 이것이
하나도 없어서 ``EXERCISE_DELAYED_HYPO`` 같은 규칙을 정답 없이 돌려야 했다.

두 가지를 알고 써야 한다.

- **날짜가 익명화로 밀려 있다.** 연도가 2021~2027로 흩어져 있고 환자마다 다르다.
  Replace-BG와 마찬가지로 달력상 절대 날짜와 요일은 의미가 없다. 환자 내부의
  시간 간격과 하루 중 시각만 쓴다. 형식은 ``DD-MM-YYYY HH:MM:SS``다.
- **12명 전원 펌프 사용자다.** ``basal``과 ``temp_basal``이 전원에게 있다. MDI는
  없다. 540과 567은 자기보고 이벤트가 0건이면서 ``temp_basal``이 유독 많아
  (540: 57일에 287건) 자동 모드 펌프일 가능성이 있다. — 확인 필요

train/test 분할은 원본의 예측 대회 규약이지 우리 관심사가 아니므로, 한 환자의 두
파일을 이어 붙여 하나의 ``patient_id``로 만든다.

원본 구조 실측 결과는 ``docs/schema_notes.md`` 참고.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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

DATASET = "ohio_t1dm"
DEFAULT_RAW_DIR = Path("data/raw/OhioT1DM/OhioT1DM")

#: 센서 측정범위(mg/dL). 값의 단일 출처는 ``src.preprocess.SENSOR_LIMITS``다.
LOW_SENTINEL_MGDL, HIGH_SENTINEL_MGDL = sensor_limits(DATASET)

#: 전 환자 5분 간격.
CGM_INTERVAL_MIN = 5

_TS_FORMAT = "%d-%m-%Y %H:%M:%S"

COHORTS = ("2018", "2020")
_SPLITS = (("train", "training"), ("test", "testing"))

#: 볼러스 ``type``별 이벤트 종류. square 계열은 시간에 걸쳐 주입되므로 연장 볼러스다.
_BOLUS_KIND = {
    "normal": "insulin_bolus",
    "normal dual": "insulin_bolus",
    "square": "insulin_bolus_extended",
    "square dual": "insulin_bolus_extended",
}


@dataclass(frozen=True)
class OhioT1DM:
    """로더 결과 묶음.

    Attributes:
        cgm: 공통 스키마 CGM 테이블.
        events: 공통 스키마 이벤트 테이블.
        patients: 환자별 메타데이터(코호트, 인슐린 종류, 체중, 관측 구간, 행 수).
    """

    cgm: pd.DataFrame
    events: pd.DataFrame
    patients: pd.DataFrame


def load(raw_dir: str | Path | None = None, validate: bool = True) -> OhioT1DM:
    """OhioT1DM 원본 XML을 읽어 공통 스키마로 변환한다."""
    raw_dir = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"원본 디렉터리가 없다: {raw_dir}")

    cgm_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    meta_rows: list[dict] = []

    for cohort, pid, files in find_patients(raw_dir):
        roots = [ET.parse(f).getroot() for f in files]
        attrs = roots[0].attrib

        cgm = _finalize_one(pd.concat([_extract_cgm(r, pid) for r in roots]))
        cgm_parts.append(cgm)
        event_parts.extend(_extract_events(r, pid) for r in roots)

        meta_rows.append(
            {
                "patient_id": patient_id(pid),
                "cohort": cohort,
                "insulin_type": attrs.get("insulin_type"),
                # 12명 전원 99다. 익명화된 값으로 보이며 쓰지 않는다.
                "weight": pd.to_numeric(attrs.get("weight"), errors="coerce"),
                "start": cgm["timestamp"].min(),
                "end": cgm["timestamp"].max(),
                "n_cgm": len(cgm),
            }
        )

    cgm_df = coerce_cgm_frame(
        pd.concat(cgm_parts, ignore_index=True)
        .sort_values(["patient_id", "timestamp"], kind="stable")
    )
    events_df = _finalize_events(event_parts)

    patients_df = pd.DataFrame(meta_rows).sort_values("patient_id").reset_index(drop=True)
    patients_df["days"] = (
        patients_df["end"] - patients_df["start"]
    ).dt.total_seconds() / 86400.0

    if validate:
        validate_cgm_frame(cgm_df)
        validate_event_frame(events_df)

    return OhioT1DM(cgm=cgm_df, events=events_df, patients=patients_df)


def load_cgm(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """공통 스키마 CGM 테이블만 필요할 때 쓰는 얇은 래퍼."""
    return load(raw_dir).cgm


def find_patients(raw_dir: str | Path) -> list[tuple[str, str, list[Path]]]:
    """``[(코호트, 환자번호, [train.xml, test.xml]), ...]``. train 파일 기준으로 찾는다."""
    raw_dir = Path(raw_dir)
    out = []
    for cohort in COHORTS:
        train_dir = raw_dir / cohort / "train"
        if not train_dir.is_dir():
            continue
        for train in sorted(train_dir.glob("*-ws-training.xml")):
            pid = train.name.split("-")[0]
            files = [train]
            test = raw_dir / cohort / "test" / f"{pid}-ws-testing.xml"
            if test.is_file():
                files.append(test)
            out.append((cohort, pid, files))
    if not out:
        raise FileNotFoundError(f"{raw_dir}에 *-ws-training.xml이 없다")
    return out


def patient_id(pid: str) -> str:
    """``"559"`` -> ``"ohio_t1dm_559"``."""
    return f"{DATASET}_{pid}"


# --- 내부 구현 ---------------------------------------------------------------


def _ts(values) -> pd.Series:
    """``DD-MM-YYYY HH:MM:SS`` 파싱. 빈 문자열과 깨진 값은 NaT."""
    return pd.to_datetime(pd.Series(values, dtype="object"), format=_TS_FORMAT,
                          errors="coerce")


def _section(root: ET.Element, tag: str) -> pd.DataFrame:
    """섹션 하나를 ``<event>`` 속성 표로. 없거나 비어 있으면 빈 표."""
    sec = root.find(tag)
    if sec is None or len(sec) == 0:
        return pd.DataFrame()
    df = pd.DataFrame([e.attrib for e in sec]).apply(lambda c: c.str.strip())
    # 원본은 값이 없을 때 빈 문자열이나 공백 하나를 넣는다.
    return df.mask(df == "")


def _extract_cgm(root: ET.Element, pid: str) -> pd.DataFrame:
    raw = _section(root, "glucose_level")
    if raw.empty:
        raise SchemaError(f"{pid}: glucose_level 섹션이 비어 있다")

    values = pd.to_numeric(raw["value"], errors="coerce")

    # 규칙: 단위를 하드코딩하지 않고 파일마다 확인한다. 이 데이터셋은 mg/dL이다.
    unit = infer_glucose_unit(values)
    if unit != MGDL:
        values = to_mgdl(values, unit)

    out = pd.DataFrame(
        {
            "patient_id": patient_id(pid),
            "timestamp": _ts(raw["ts"]),
            "glucose_mgdl": values.to_numpy(),
            "source": DATASET,
        }
    )
    out = out.loc[out["timestamp"].notna() & out["glucose_mgdl"].notna()]
    plausible = out["glucose_mgdl"].between(GLUCOSE_MIN_MGDL, GLUCOSE_MAX_MGDL)
    return out.loc[plausible]


def _finalize_one(cgm: pd.DataFrame) -> pd.DataFrame:
    """train/test를 이어 붙인 뒤 정렬·중복 제거. 두 파일의 경계는 겹치지 않는다."""
    cgm = cgm.sort_values("timestamp", kind="stable")
    return cgm.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)


def _finalize_events(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return empty_event_frame()
    df = pd.concat(parts, ignore_index=True)
    # 원본이 같은 이벤트를 train과 test 파일 양쪽에 싣는 경우가 있다(544 운동 등).
    # 두 파일을 이어 붙였으므로 행 전체가 같으면 하나만 남긴다.
    df = df.drop_duplicates(keep="first")
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


def _duration_min(begin: pd.Series, end: pd.Series) -> pd.Series:
    """``ts_begin``/``ts_end`` 차이(분). 끝이 없으면 NaN."""
    return (_ts(end) - _ts(begin)).dt.total_seconds() / 60.0


def _tag(prefix: str, series: pd.Series | None) -> pd.Series:
    """``"intensity=5"`` 꼴. 값이 없거나 컬럼 자체가 없으면 빈 문자열."""
    if series is None:
        return pd.Series(dtype="string")
    s = series.astype("string")
    return (prefix + "=" + s).fillna("")


def _join(*cols: pd.Series | None) -> pd.Series:
    """빈 조각을 빼고 ``"; "``로 잇는다. 전부 비면 ``None``. 없는 컬럼은 건너뛴다."""
    cols = [c for c in cols if c is not None and len(c)]
    df = pd.concat([c.astype("string").fillna("") for c in cols], axis=1)
    joined = df.apply(lambda r: "; ".join(v for v in r if v), axis=1)
    return joined.replace("", None)


def _extract_events(root: ET.Element, pid: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    # --- 식사 ---------------------------------------------------------------
    meal = _section(root, "meal")
    if not meal.empty:
        parts.append(_events_frame(
            pid, _ts(meal["ts"]), "meal",
            pd.to_numeric(meal["carbs"], errors="coerce"), "g", meal.get("type"),
        ))

    # --- 볼러스 -------------------------------------------------------------
    # normal은 즉시 주입, square는 ts_begin~ts_end에 걸쳐 주입한다. dual은 둘의
    # 조합인데 원본이 normal 분과 square 분을 각각 한 행으로 나눠 준다.
    # bwz_carb_input(볼러스 계산기에 넣은 탄수화물)은 2018 코호트에만 있다.
    bolus = _section(root, "bolus")
    if not bolus.empty:
        kind = bolus["type"].str.lower().map(_BOLUS_KIND)
        unknown = bolus.loc[kind.isna(), "type"].dropna().unique()
        if len(unknown):
            raise SchemaError(f"{pid}: 알 수 없는 bolus type {sorted(unknown)}")
        dose = pd.to_numeric(bolus["dose"], errors="coerce")
        duration = _duration_min(bolus["ts_begin"], bolus["ts_end"]).round(0)
        carb = bolus["bwz_carb_input"] if "bwz_carb_input" in bolus else pd.Series(
            np.nan, index=bolus.index)

        for event_type in ("insulin_bolus", "insulin_bolus_extended"):
            m = (kind == event_type) & dose.notna()
            if not m.any():
                continue
            text = _join(
                bolus.loc[m, "type"],
                _tag("duration_min", duration.loc[m].astype("Int64")) if event_type.endswith("extended") else pd.Series("", index=bolus.index[m]),
                _tag("bwz_carb_input", carb.loc[m]),
            )
            parts.append(_events_frame(
                pid, _ts(bolus.loc[m, "ts_begin"]), event_type, dose.loc[m], "IU", text,
            ))

    # --- 기저 / 임시 기저 / 펌프 중단 --------------------------------------
    basal = _section(root, "basal")
    if not basal.empty:
        rate = pd.to_numeric(basal["value"], errors="coerce")
        parts.append(_events_frame(
            pid, _ts(basal["ts"]), "insulin_basal_rate", rate, "IU/h", "scheduled",
        ))

    # temp_basal은 프로파일 위에 덮어쓰는 임시 주입률이다. 0.0이면 사실상 중단이라
    # pump_suspend로 보낸다. 572건 중 371건이 0.0이다.
    temp = _section(root, "temp_basal")
    if not temp.empty:
        rate = pd.to_numeric(temp["value"], errors="coerce")
        duration = _duration_min(temp["ts_begin"], temp["ts_end"]).round(0)
        text = _join(pd.Series("temp_basal", index=temp.index),
                     _tag("duration_min", duration.astype("Int64")))
        is_zero = rate.fillna(-1) == 0
        if is_zero.any():
            parts.append(_events_frame(
                pid, _ts(temp.loc[is_zero, "ts_begin"]), "pump_suspend", np.nan, None,
                text.loc[is_zero],
            ))
        keep = rate.notna() & ~is_zero
        if keep.any():
            parts.append(_events_frame(
                pid, _ts(temp.loc[keep, "ts_begin"]), "insulin_basal_rate",
                rate.loc[keep], "IU/h", text.loc[keep],
            ))

    # --- 자가혈당 -------------------------------------------------------------
    fs = _section(root, "finger_stick")
    if not fs.empty:
        parts.append(_events_frame(
            pid, _ts(fs["ts"]), "cbg", pd.to_numeric(fs["value"], errors="coerce"), "mg/dL",
        ))

    # --- 자기보고 이벤트 ------------------------------------------------------
    # 여기가 이 데이터셋을 쓰는 이유다. exercise는 강도(1~10)와 지속시간(분)이
    # 있고 type/competitive는 전 행 비어 있다. hypo_event는 시각만 있다.
    ex = _section(root, "exercise")
    if not ex.empty:
        parts.append(_events_frame(
            pid, _ts(ex["ts"]), "exercise",
            pd.to_numeric(ex["duration"], errors="coerce"), "min",
            _join(_tag("intensity", ex.get("intensity")), _tag("type", ex.get("type"))),
        ))

    hypo = _section(root, "hypo_event")
    if not hypo.empty:
        parts.append(_events_frame(pid, _ts(hypo["ts"]), "hypo_event", np.nan, None))

    ill = _section(root, "illness")
    if not ill.empty:
        parts.append(_events_frame(
            pid, _ts(ill["ts_begin"]), "illness", np.nan, None,
            _join(ill.get("type"), ill.get("description"), _tag("end", ill.get("ts_end"))),
        ))

    stress = _section(root, "stressors")
    if not stress.empty:
        parts.append(_events_frame(
            pid, _ts(stress["ts"]), "stressor", np.nan, None,
            _join(stress.get("type"), stress.get("description")),
        ))

    # sleep은 2018 코호트에서 ts_begin/ts_end가 뒤바뀐 행이 있어 둘 중 이른 쪽을
    # 시작으로 본다.
    sleep = _section(root, "sleep")
    if not sleep.empty:
        b, e = _ts(sleep["ts_begin"]), _ts(sleep["ts_end"])
        start = pd.concat([b, e], axis=1).min(axis=1)
        hours = (b - e).abs().dt.total_seconds() / 3600.0
        parts.append(_events_frame(
            pid, start, "sleep", hours, "h", _tag("quality", sleep.get("quality")),
        ))

    if not parts:
        return empty_event_frame()
    return pd.concat(parts, ignore_index=True)


__all__ = [
    "COHORTS",
    "DATASET",
    "DEFAULT_RAW_DIR",
    "OhioT1DM",
    "find_patients",
    "load",
    "load_cgm",
    "patient_id",
]
