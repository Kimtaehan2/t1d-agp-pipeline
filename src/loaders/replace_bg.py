"""Replace-BG 로더 (주력 데이터셋).

원본은 JAEB 배포 형식의 파이프(`|`) 구분 텍스트다. 226명 · 26주 · Dexcom G4 CGM
1,480만 행. `HDeviceCGM.txt` 하나가 837MB라 전체를 한 번에 DataFrame으로 올리지
않고 청크로 읽는다. 전체 변환은 :func:`export_cgm_parquet`로 Parquet에 흘려 쓴다.

**날짜가 익명화돼 있다.** 원본에는 절대 날짜가 없고 등록일로부터의 상대 일수
(`DeviceDtTmDaysFromEnroll`, 음수 가능)와 시각(`DeviceTm`)만 있다. 공통 스키마의
``timestamp``를 채우려면 기준일이 필요하므로 :data:`ANCHOR_DATE`를 더해 펼친다.
따라서 **달력상의 절대 날짜와 요일은 의미가 없다.** 의미가 있는 것은 환자 내부의
시간 간격과 하루 중 시각(time-of-day)뿐이며, AGP는 후자만 쓰므로 문제가 없다.

원본 구조 실측 결과는 ``docs/schema_notes.md`` 참고.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import sensor_limits
from src.schema import (
    SchemaError,
    coerce_cgm_frame,
    coerce_event_frame,
    empty_event_frame,
)
from src.storage import write_parquet

DATASET = "replace_bg"
DEFAULT_RAW_DIR = Path("data/raw/Replace-BG Dataset/Data Tables")

#: 익명화된 상대 일수를 절대 시각으로 펼칠 때 쓰는 기준일(= 등록일 D0).
#: 값 자체에 의미는 없다. 연구 수행 시기(2015~2016)에 맞춰 골랐을 뿐이다.
ANCHOR_DATE = pd.Timestamp("2015-01-01")

#: Dexcom G4가 측정 범위 밖을 기록할 때 쓰는 센티널. **40/400이 아니다.**
#: 실측 빈도: 39가 28,885건(주변 값의 6배), 401이 48,935건(주변 값의 40배).
#: 값의 단일 출처는 ``src.preprocess.SENSOR_LIMITS``다.
LOW_SENTINEL_MGDL, HIGH_SENTINEL_MGDL = sensor_limits(DATASET)

#: Dexcom G4 측정 주기(분). 실측 간격의 92%가 정확히 300초, 99%가 298~302초다.
CGM_INTERVAL_MIN = 5

_CGM_FILE = "HDeviceCGM.txt"
_BOLUS_FILE = "HDeviceBolus.txt"
_BASAL_FILE = "HDeviceBasal.txt"
_WIZARD_FILE = "HDeviceWizard.txt"
_BGM_FILE = "HDeviceBGM.txt"
_ROSTER_FILE = "HPtRoster.txt"

_SEP = "|"

#: 볼러스 계산기(HDeviceWizard) 테이블의 혈당 관련 값은 mmol/L로 기록돼 있다.
#: 226명 중 mg/dL로 보이는 환자는 0명이었다(BgTargetLow 중앙값 전원 25 미만).
_WIZARD_GLUCOSE_IS_MMOL = True

_MS_PER_MIN = 60_000


@dataclass(frozen=True)
class ReplaceBGStats:
    """변환 과정에서 버린 행을 추적한다. 조용히 사라지는 데이터가 없게 한다."""

    rows_read: int = 0
    dropped_calibration: int = 0
    dropped_before_min_day: int = 0
    dropped_bad_timestamp: int = 0
    dropped_duplicate: int = 0
    rows_kept: int = 0

    def summary(self) -> str:
        return (
            f"읽은 행 {self.rows_read:,} → 남은 행 {self.rows_kept:,} "
            f"(교정값 {self.dropped_calibration:,}, "
            f"기준일 이전 {self.dropped_before_min_day:,}, "
            f"시각 파싱 실패 {self.dropped_bad_timestamp:,}, "
            f"중복 {self.dropped_duplicate:,} 제외)"
        )


# --- 공통 헬퍼 ---------------------------------------------------------------


def build_timestamp(
    days_from_enroll,
    time_of_day,
    anchor: pd.Timestamp = ANCHOR_DATE,
) -> pd.Series:
    """상대 일수 + 시각을 절대 timestamp로 펼친다.

    파싱할 수 없는 값은 ``NaT``로 남긴다. 호출 측에서 걸러낸다.
    """
    days = pd.to_numeric(days_from_enroll, errors="coerce")
    tod = pd.to_timedelta(
        pd.Series(time_of_day).astype("str").str.strip(), errors="coerce"
    )
    ts = anchor + pd.to_timedelta(days, unit="D").to_numpy() + tod.to_numpy()
    return pd.Series(ts).astype("datetime64[ns]")


def patient_id(pt_id) -> str:
    """``263`` -> ``"replace_bg_263"``."""
    return f"{DATASET}_{int(pt_id)}"


def _patient_id_series(pt_ids: pd.Series) -> pd.Series:
    return DATASET + "_" + pt_ids.astype("int64").astype("str")


def _lexicographic_rank(pt: np.ndarray) -> np.ndarray:
    """숫자 PtID 배열을 문자열 ``patient_id``의 사전순 순위로 바꾼다."""
    uniq = np.unique(pt)
    labels = np.array([patient_id(u) for u in uniq])
    rank_of_uniq = np.argsort(np.argsort(labels))
    return rank_of_uniq[np.searchsorted(uniq, pt)]


def _resolve(raw_dir: str | Path | None, filename: str) -> Path:
    base = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    path = base / filename
    if not path.is_file():
        raise FileNotFoundError(f"원본 파일이 없다: {path}")
    return path


def _wanted_pt_ids(patient_ids: list[str] | None) -> set[int] | None:
    """``["replace_bg_263"]`` -> ``{263}``. ``None``이면 전체."""
    if patient_ids is None:
        return None
    out = set()
    for pid in patient_ids:
        text = str(pid)
        out.add(int(text.rsplit("_", 1)[-1] if text.startswith(DATASET) else text))
    return out


# --- CGM ---------------------------------------------------------------------


def _scan_cgm(
    raw_dir: str | Path | None,
    min_day: int | None,
    patient_ids: list[str] | None,
    chunksize: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ReplaceBGStats]:
    """CGM 파일을 한 번 훑어 (PtID, timestamp, glucose)를 압축 배열로 모은다.

    문자열 ``patient_id``는 이 단계에서 만들지 않는다. 1,480만 개의 파이썬 문자열은
    1GB 가까이 먹기 때문에, 실제로 내보낼 때 환자 단위로만 만든다.
    """
    path = _resolve(raw_dir, _CGM_FILE)
    wanted = _wanted_pt_ids(patient_ids)

    pt_parts: list[np.ndarray] = []
    ts_parts: list[np.ndarray] = []
    gl_parts: list[np.ndarray] = []
    n_read = n_calib = n_early = n_bad_ts = 0

    reader = pd.read_csv(
        path,
        sep=_SEP,
        usecols=["PtID", "DeviceDtTmDaysFromEnroll", "DeviceTm",
                 "RecordType", "GlucoseValue"],
        dtype={
            "PtID": "int32",
            "DeviceDtTmDaysFromEnroll": "int32",
            "DeviceTm": "str",
            "RecordType": "str",
            "GlucoseValue": "float32",
        },
        chunksize=chunksize,
    )

    for chunk in reader:
        n_read += len(chunk)

        # 환자 필터를 먼저 걸어야 통계가 "선택한 환자 기준"으로 일관된다.
        if wanted is not None:
            chunk = chunk.loc[chunk["PtID"].isin(wanted)]
        if chunk.empty:
            continue

        is_cgm = chunk["RecordType"] == "CGM"
        n_calib += int((~is_cgm).sum())
        chunk = chunk.loc[is_cgm]
        if chunk.empty:
            continue

        if min_day is not None:
            keep = chunk["DeviceDtTmDaysFromEnroll"] >= min_day
            n_early += int((~keep).sum())
            chunk = chunk.loc[keep]
            if chunk.empty:
                continue

        ts = build_timestamp(
            chunk["DeviceDtTmDaysFromEnroll"], chunk["DeviceTm"]
        ).to_numpy()
        ok = ~pd.isna(ts)
        n_bad_ts += int((~ok).sum())

        pt_parts.append(chunk["PtID"].to_numpy()[ok])
        ts_parts.append(ts[ok].astype("datetime64[ns]"))
        gl_parts.append(chunk["GlucoseValue"].to_numpy()[ok])

    if not pt_parts:
        empty_i = np.empty(0, dtype="int32")
        return (
            empty_i,
            np.empty(0, dtype="datetime64[ns]"),
            np.empty(0, dtype="float32"),
            ReplaceBGStats(
                rows_read=n_read,
                dropped_calibration=n_calib,
                dropped_before_min_day=n_early,
                dropped_bad_timestamp=n_bad_ts,
            ),
        )

    pt = np.concatenate(pt_parts)
    ts = np.concatenate(ts_parts)
    gl = np.concatenate(gl_parts)
    del pt_parts, ts_parts, gl_parts

    # (patient_id, timestamp) 오름차순 정렬 후 중복 제거.
    # patient_id는 문자열이라 정렬 기준이 숫자순이 아니라 **사전순**이다.
    # (replace_bg_10 < replace_bg_100 < replace_bg_2). 공통 스키마 검증이
    # 사전순을 요구하므로 숫자 PtID로 정렬하면 안 된다.
    order = np.lexsort((ts, _lexicographic_rank(pt)))
    pt, ts, gl = pt[order], ts[order], gl[order]

    if len(pt) > 1:
        dup = (pt[1:] == pt[:-1]) & (ts[1:] == ts[:-1])
        keep = np.concatenate(([True], ~dup))
        n_dup = int((~keep).sum())
        pt, ts, gl = pt[keep], ts[keep], gl[keep]
    else:
        n_dup = 0

    stats = ReplaceBGStats(
        rows_read=n_read,
        dropped_calibration=n_calib,
        dropped_before_min_day=n_early,
        dropped_bad_timestamp=n_bad_ts,
        dropped_duplicate=n_dup,
        rows_kept=len(pt),
    )
    return pt, ts, gl, stats


def _frame_from_arrays(pt: np.ndarray, ts: np.ndarray, gl: np.ndarray) -> pd.DataFrame:
    return coerce_cgm_frame(
        pd.DataFrame(
            {
                "patient_id": _patient_id_series(pd.Series(pt)),
                "timestamp": ts,
                "glucose_mgdl": gl.astype("float64"),
                "source": DATASET,
            }
        )
    )


def load_cgm(
    raw_dir: str | Path | None = None,
    min_day: int | None = 0,
    patient_ids: list[str] | None = None,
    chunksize: int = 2_000_000,
    return_stats: bool = False,
):
    """CGM을 공통 스키마 DataFrame으로 읽는다.

    Args:
        min_day: 이 일수 이전(등록일 기준) 기록을 버린다. 기본 0 = 등록일 이후만.
            원본에는 등록 전 기기 이력이 7.4% 섞여 있고 최대 595일 전까지 거슬러
            올라간다. ``None``이면 전부 남긴다.
        patient_ids: ``["replace_bg_263", ...]`` 또는 원본 숫자 ID. 전체를 메모리에
            올리면 1GB를 넘으므로 탐색할 때는 몇 명만 지정해서 쓴다.

    Warning:
        ``patient_ids=None``이면 1,400만 행짜리 DataFrame이 만들어진다. 전체 변환은
        :func:`export_cgm_parquet`를 쓰는 편이 낫다.
    """
    pt, ts, gl, stats = _scan_cgm(raw_dir, min_day, patient_ids, chunksize)
    df = _frame_from_arrays(pt, ts, gl)
    return (df, stats) if return_stats else df


def export_cgm_parquet(
    out_path: str | Path,
    raw_dir: str | Path | None = None,
    min_day: int | None = 0,
    patient_ids: list[str] | None = None,
    chunksize: int = 2_000_000,
    patients_per_batch: int = 10,
) -> ReplaceBGStats:
    """CGM 전체를 Parquet 한 파일로 흘려 쓴다.

    문자열 ``patient_id`` 컬럼을 환자 묶음 단위로만 만들어 메모리를 억제한다.
    """
    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()

    pt, ts, gl, stats = _scan_cgm(raw_dir, min_day, patient_ids, chunksize)
    if len(pt) == 0:
        write_parquet(_frame_from_arrays(pt, ts, gl), out_path)
        return stats

    # 환자별 행이 이미 연속 블록으로 모여 있으므로, 값이 바뀌는 지점이 경계다.
    starts = np.flatnonzero(np.concatenate(([True], pt[1:] != pt[:-1])))
    n_patients = len(starts)
    bounds = list(starts) + [len(pt)]

    for i in range(0, n_patients, patients_per_batch):
        lo = bounds[i]
        hi = bounds[min(i + patients_per_batch, n_patients)]
        batch = _frame_from_arrays(pt[lo:hi], ts[lo:hi], gl[lo:hi])
        write_parquet(batch, out_path, append=i > 0)

    return stats


def load_patients(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """환자 명단(HPtRoster). ``patient_id``를 공통 스키마 형식으로 붙여 준다."""
    df = pd.read_csv(_resolve(raw_dir, _ROSTER_FILE), sep=_SEP)
    df["patient_id"] = _patient_id_series(df["PtID"])
    return df


# --- 이벤트 ------------------------------------------------------------------


def _event_frame(
    pt_ids: pd.Series,
    timestamps: pd.Series,
    event_type: str,
    values,
    unit: str | None,
    text=None,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "patient_id": _patient_id_series(pt_ids.reset_index(drop=True)),
            "timestamp": pd.Series(timestamps).reset_index(drop=True),
            "event_type": event_type,
            "value": pd.Series(values, dtype="float64").reset_index(drop=True),
            "unit": unit,
            "text": (
                pd.Series(text).reset_index(drop=True) if text is not None else None
            ),
            "source": DATASET,
        }
    )
    return df.loc[df["timestamp"].notna()]


def _read_table(raw_dir, filename: str, pt_col: str) -> pd.DataFrame:
    df = pd.read_csv(_resolve(raw_dir, filename), sep=_SEP, low_memory=False)
    if pt_col not in df.columns:
        raise SchemaError(f"{filename}: 환자 ID 컬럼 {pt_col}이 없다")
    return df


def _filter_days(df: pd.DataFrame, min_day: int | None) -> pd.DataFrame:
    if min_day is None:
        return df
    days = pd.to_numeric(df["DeviceDtTmDaysFromEnroll"], errors="coerce")
    return df.loc[days >= min_day]


def load_events(
    raw_dir: str | Path | None = None,
    min_day: int | None = 0,
    patient_ids: list[str] | None = None,
) -> pd.DataFrame:
    """식사·인슐린·자가혈당·케톤 이벤트를 공통 스키마로 모은다.

    출처:
        - ``HDeviceWizard.CarbInput`` → ``meal`` (탄수화물 g)
        - ``HDeviceBolus.Normal`` → ``insulin_bolus`` (IU)
        - ``HDeviceBolus.Extended`` → ``insulin_bolus_extended`` (IU)
        - ``HDeviceBasal`` → ``insulin_basal_rate`` (IU/h) / ``pump_suspend``
        - ``HDeviceBGM`` → ``cbg`` (mg/dL) / ``blood_ketone`` (mmol/L)
        - ``HDeviceCGM``의 ``RecordType=Calibration`` → ``cbg`` (text=``calibration``)
    """
    wanted = _wanted_pt_ids(patient_ids)
    parts: list[pd.DataFrame] = []

    def keep_patients(df: pd.DataFrame, col: str) -> pd.DataFrame:
        return df if wanted is None else df.loc[df[col].isin(wanted)]

    # --- 식사 (볼러스 계산기의 탄수화물 입력) ---
    wiz = _filter_days(keep_patients(_read_table(raw_dir, _WIZARD_FILE, "PtId"), "PtId"),
                       min_day)
    carbs = pd.to_numeric(wiz["CarbInput"], errors="coerce")
    meals = wiz.loc[carbs > 0]
    if not meals.empty:
        parts.append(
            _event_frame(
                meals["PtId"],
                build_timestamp(meals["DeviceDtTmDaysFromEnroll"], meals["DeviceTm"]),
                "meal",
                carbs.loc[meals.index],
                "g",
            )
        )

    # --- 볼러스 ---
    bolus = _filter_days(keep_patients(_read_table(raw_dir, _BOLUS_FILE, "PtID"), "PtID"),
                         min_day)
    bolus_ts = build_timestamp(bolus["DeviceDtTmDaysFromEnroll"], bolus["DeviceTm"])
    # BolusType은 'normal'과 'Normal'이 섞여 있어 소문자로 정규화한다.
    bolus_type = bolus["BolusType"].astype("str").str.strip().str.lower()

    normal = pd.to_numeric(bolus["Normal"], errors="coerce")
    has_normal = normal.notna()
    if has_normal.any():
        parts.append(
            _event_frame(
                bolus.loc[has_normal, "PtID"],
                bolus_ts.loc[has_normal.to_numpy()],
                "insulin_bolus",
                normal.loc[has_normal],
                "IU",
                bolus_type.loc[has_normal],
            )
        )

    extended = pd.to_numeric(bolus["Extended"], errors="coerce")
    has_ext = extended.notna()
    if has_ext.any():
        duration_min = pd.to_numeric(bolus["Duration"], errors="coerce") / _MS_PER_MIN
        text = (
            bolus_type.loc[has_ext]
            + "; duration_min="
            + duration_min.loc[has_ext].round(0).astype("str")
        )
        parts.append(
            _event_frame(
                bolus.loc[has_ext, "PtID"],
                bolus_ts.loc[has_ext.to_numpy()],
                "insulin_bolus_extended",
                extended.loc[has_ext],
                "IU",
                text,
            )
        )

    # --- 기저 주입률 / 펌프 중단 ---
    basal = _filter_days(keep_patients(_read_table(raw_dir, _BASAL_FILE, "PtID"), "PtID"),
                         min_day)
    basal_ts = build_timestamp(basal["DeviceDtTmDaysFromEnroll"], basal["DeviceTm"])
    basal_type = basal["BasalType"].astype("str").str.strip().str.lower()

    is_suspend = basal_type == "suspend"
    if is_suspend.any():
        parts.append(
            _event_frame(
                basal.loc[is_suspend, "PtID"],
                basal_ts.loc[is_suspend.to_numpy()],
                "pump_suspend",
                np.nan,
                None,
                basal_type.loc[is_suspend],
            )
        )

    rate = pd.to_numeric(basal["Rate"], errors="coerce")
    has_rate = rate.notna() & ~is_suspend
    if has_rate.any():
        parts.append(
            _event_frame(
                basal.loc[has_rate, "PtID"],
                basal_ts.loc[has_rate.to_numpy()],
                "insulin_basal_rate",
                rate.loc[has_rate],
                "IU/h",
                basal_type.loc[has_rate],
            )
        )

    # --- 자가혈당 / 케톤 ---
    bgm = _filter_days(keep_patients(_read_table(raw_dir, _BGM_FILE, "PtID"), "PtID"),
                       min_day)
    bgm_ts = build_timestamp(bgm["DeviceDtTmDaysFromEnroll"], bgm["DeviceTm"])
    bgm_value = pd.to_numeric(bgm["GlucoseValue"], errors="coerce")
    bgm_sub = bgm["RecordSubType"].astype("str").str.strip().str.lower()

    is_bgm = (bgm["RecordType"] == "BGM") & bgm_value.notna()
    if is_bgm.any():
        parts.append(
            _event_frame(
                bgm.loc[is_bgm, "PtID"],
                bgm_ts.loc[is_bgm.to_numpy()],
                "cbg",
                bgm_value.loc[is_bgm],
                "mg/dL",
                bgm_sub.loc[is_bgm].replace({"nan": None}),
            )
        )

    is_ketone = (bgm["RecordType"] == "Ketone") & bgm_value.notna()
    if is_ketone.any():
        parts.append(
            _event_frame(
                bgm.loc[is_ketone, "PtID"],
                bgm_ts.loc[is_ketone.to_numpy()],
                "blood_ketone",
                bgm_value.loc[is_ketone],
                "mmol/L",
            )
        )

    # --- CGM 교정값 (지문채혈) ---
    parts.append(_load_calibrations(raw_dir, min_day, wanted))

    parts = [p for p in parts if not p.empty]
    if not parts:
        return empty_event_frame()

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["patient_id", "timestamp", "event_type"], kind="stable")
    return coerce_event_frame(out)


def _load_calibrations(raw_dir, min_day: int | None, wanted: set[int] | None):
    """``HDeviceCGM``에서 ``RecordType=Calibration``만 뽑는다 (14만 건)."""
    parts = []
    reader = pd.read_csv(
        _resolve(raw_dir, _CGM_FILE),
        sep=_SEP,
        usecols=["PtID", "DeviceDtTmDaysFromEnroll", "DeviceTm",
                 "RecordType", "GlucoseValue"],
        dtype={"PtID": "int32", "DeviceDtTmDaysFromEnroll": "int32",
               "DeviceTm": "str", "RecordType": "str", "GlucoseValue": "float32"},
        chunksize=2_000_000,
    )
    for chunk in reader:
        cal = chunk.loc[chunk["RecordType"] == "Calibration"]
        if wanted is not None:
            cal = cal.loc[cal["PtID"].isin(wanted)]
        cal = _filter_days(cal, min_day)
        if cal.empty:
            continue
        parts.append(
            _event_frame(
                cal["PtID"],
                build_timestamp(cal["DeviceDtTmDaysFromEnroll"], cal["DeviceTm"]),
                "cbg",
                cal["GlucoseValue"],
                "mg/dL",
                "calibration",
            )
        )
    if not parts:
        return empty_event_frame()
    return pd.concat(parts, ignore_index=True)
