"""단위 변환 · 리샘플링 · 결측 처리 · 센서 캡핑 플래그.

파이프라인 순서(:func:`preprocess_cgm`):

1. **그리드 스냅** — 원본 타임스탬프를 가장 가까운 5분 격자로 반올림한다.
   Replace-BG는 5분 간격이지만 기기 시계가 흔들려 정확히 300초는 92%뿐이고,
   Shanghai는 15분 간격에 격자 정렬조차 안 돼 있다(16:43, 16:58, 17:13 …).
2. **격자 채우기** — 환자별 첫 관측 ~ 마지막 관측을 5분 격자로 빈틈없이 편다.
   관측이 없는 슬롯은 NaN으로 남는다.
3. **짧은 공백만 보간** — 30분 이내 공백은 선형보간, 30분 초과는 손대지 않는다.
   채운 값은 ``is_imputed=True``로 구분한다.
4. **캡핑 플래그** — 센서 측정범위에 걸린 값이 연속으로 반복되는 구간을
   ``is_capped=True``로 표시한다. 값 자체는 바꾸지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import (
    PROCESSED_CGM_COLUMNS,
    coerce_processed_cgm_frame,
    empty_processed_cgm_frame,
)

# --- 단위 -------------------------------------------------------------------

#: mmol/L → mg/dL 변환 계수
MMOL_TO_MGDL = 18.018

#: 중앙값이 이 값보다 작으면 mmol/L, 크면 mg/dL로 판단한다.
#: (mmol/L의 중앙값은 보통 10 근처, mg/dL은 150 근처라 25는 안전한 경계다.)
_UNIT_SPLIT = 25.0

MGDL = "mg/dL"
MMOL = "mmol/L"

# --- 리샘플링/결측 기본값 ----------------------------------------------------

#: 공통 리샘플링 격자(분).
GRID_MINUTES = 5

#: 이 시간 **이내**의 공백만 선형보간한다. 초과분은 NaN으로 남긴다.
MAX_INTERPOLATION_GAP_MINUTES = 30

# --- 센서 측정범위 -----------------------------------------------------------
#
# 데이터셋마다 센티널 값이 다르다. 전부 실측한 값이며 근거는
# docs/schema_notes.md에 적어 뒀다. 로더는 여기서 값을 가져다 쓴다.
#
#   replace_bg     : Dexcom G4. 39가 28,885건, 401이 48,935건으로 주변 값의
#                    6~40배. 스펙에 적힌 40/400이 아니다.
#   shanghai_t1dm  : 값이 전부 0.1 mmol/L 배수라 하한 2.2 mmol/L = 39.6,
#                    상한 27.8 mmol/L = 500.4. 실측 최대는 475.2로 상한 미도달.
#   t1d_uom        : 값이 전부 0.1 mmol/L 배수다. 하한 2.2 mmol/L = 39.6,
#                    상한 27.8 mmol/L = 500.9. MDI 환자 116,625행 기준으로
#                    27.8이 23건인데 27.7은 1건이라 상한 쌓임이 뚜렷하다.
#                    **센서가 한 종류가 아니다.** 상한이 22.2 mmol/L(=400.0)인
#                    환자도 있다(2309: 22.2가 207건). 여기 값은 넓은 쪽이라
#                    22.2에서 검열된 환자의 캡핑은 잡히지 않는다.
#   simulated      : 정확히 40.0/400.0으로 클리핑돼 있다.
SENSOR_LIMITS: dict[str, tuple[float, float]] = {
    "replace_bg": (39.0, 401.0),
    "shanghai_t1dm": (39.6, 500.4),
    "t1d_uom": (39.6, 500.9),
    "simulated": (40.0, 400.0),
}

#: 캡핑으로 인정할 최소 연속 길이. 스펙의 "연속으로 반복되는 구간"을 따라 2다.
#: 1로 두면 측정범위에 닿은 모든 값(검열된 값)을 표시한다.
MIN_CAPPING_RUN = 2

# --- AGP 창 유효성 -----------------------------------------------------------

MIN_AGP_DAYS = 10
RECOMMENDED_AGP_DAYS = 14
MIN_AGP_COVERAGE = 0.70


class UnitInferenceError(ValueError):
    """혈당 단위를 판별할 수 없을 때 발생."""


# ============================================================================
# 단위
# ============================================================================


def infer_glucose_unit(values: pd.Series | np.ndarray) -> str:
    """혈당 값의 단위를 중앙값으로 판별한다.

    데이터셋마다 단위가 다르므로 로더는 하드코딩 대신 이 함수로 확인한다.
    """
    s = pd.Series(values, dtype="float64").dropna()
    if s.empty:
        raise UnitInferenceError("유효한 혈당 값이 없어 단위를 판별할 수 없다")

    median = float(s.median())
    if median <= 0:
        raise UnitInferenceError(f"혈당 중앙값이 비정상이다: {median}")
    return MMOL if median < _UNIT_SPLIT else MGDL


def to_mgdl(values: pd.Series | np.ndarray, unit: str | None = None) -> pd.Series:
    """혈당 값을 mg/dL로 변환한다.

    Args:
        values: 원본 혈당 값.
        unit: ``"mg/dL"`` 또는 ``"mmol/L"``. ``None``이면 자동 판별한다.
    """
    s = pd.Series(values, dtype="float64").reset_index(drop=True)
    unit = unit or infer_glucose_unit(s)
    if unit == MGDL:
        return s
    if unit == MMOL:
        return s * MMOL_TO_MGDL
    raise ValueError(f"알 수 없는 단위: {unit!r}")


def sensor_limits(source: str) -> tuple[float, float]:
    """데이터셋 이름으로 센서 측정범위(하한, 상한)를 찾는다."""
    try:
        return SENSOR_LIMITS[source]
    except KeyError:
        raise KeyError(
            f"'{source}'의 센서 측정범위가 등록돼 있지 않다. "
            f"원본을 실측해 SENSOR_LIMITS에 추가해라. 등록된 것: "
            f"{sorted(SENSOR_LIMITS)}"
        ) from None


# ============================================================================
# 런(run) 계산 — 결측 보간과 캡핑 플래그가 공유한다
# ============================================================================


def _run_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``True``가 연속된 구간의 (시작, 끝+1) 인덱스 쌍을 돌려준다."""
    if mask.size == 0:
        return np.empty(0, dtype="int64"), np.empty(0, dtype="int64")
    changed = np.flatnonzero(np.concatenate(([True], mask[1:] != mask[:-1])))
    starts = changed
    ends = np.concatenate((changed[1:], [mask.size]))
    keep = mask[starts]
    return starts[keep], ends[keep]


# ============================================================================
# 리샘플링
# ============================================================================


def snap_to_grid(
    timestamps: pd.Series, grid_minutes: int = GRID_MINUTES
) -> pd.Series:
    """타임스탬프를 가장 가까운 격자점으로 반올림한다."""
    return timestamps.dt.round(f"{grid_minutes}min")


def resample_patient(
    timestamps: pd.Series,
    glucose: pd.Series,
    grid_minutes: int = GRID_MINUTES,
) -> tuple[pd.DatetimeIndex, np.ndarray, int]:
    """한 환자를 격자로 편다.

    Returns:
        (격자 시각, 혈당 배열, 같은 슬롯으로 뭉친 관측 수).
        같은 격자점에 관측이 둘 이상 떨어지면 평균을 쓴다.
    """
    snapped = snap_to_grid(timestamps, grid_minutes)
    observed = pd.Series(glucose.to_numpy(), index=snapped.to_numpy())
    observed = observed[observed.index.notna()]

    if observed.empty:
        return pd.DatetimeIndex([], dtype="datetime64[ns]"), np.empty(0), 0

    collapsed = observed.groupby(level=0).mean()
    n_collisions = len(observed) - len(collapsed)

    grid = pd.date_range(
        collapsed.index.min(),
        collapsed.index.max(),
        freq=f"{grid_minutes}min",
    )
    return grid, collapsed.reindex(grid).to_numpy(dtype="float64"), n_collisions


# ============================================================================
# 결측 처리
# ============================================================================


def interpolate_short_gaps(
    values: np.ndarray,
    grid_minutes: int = GRID_MINUTES,
    max_gap_minutes: int = MAX_INTERPOLATION_GAP_MINUTES,
) -> tuple[np.ndarray, np.ndarray]:
    """짧은 공백만 선형보간한다.

    공백의 길이는 **양 끝 관측 사이의 시간**으로 잰다. 빈 슬롯이 L개면 공백은
    ``(L+1) × grid_minutes`` 분이다. 예를 들어 5분 격자에서 빈 슬롯 5개는
    30분 공백이라 보간 대상이지만, 6개는 35분이라 손대지 않는다.

    앞뒤가 막히지 않은 구간(시계열 맨 앞/뒤의 결측)은 외삽이 되므로 보간하지 않는다.

    Returns:
        (보간된 값, ``is_imputed`` 플래그)
    """
    missing = np.isnan(values)
    imputed = np.zeros(values.shape, dtype=bool)
    if not missing.any():
        return values.copy(), imputed

    max_run = max_gap_minutes // grid_minutes - 1
    filled = (
        pd.Series(values)
        .interpolate(method="linear", limit_area="inside")
        .to_numpy()
    )

    starts, ends = _run_bounds(missing)
    for start, end in zip(starts, ends):
        if end - start > max_run:
            continue          # 30분 초과 공백은 절대 보간하지 않는다
        if start == 0 or end == values.size:
            continue          # 양끝이 막히지 않은 구간은 외삽이라 제외
        imputed[start:end] = True

    out = values.copy()
    out[imputed] = filled[imputed]
    return out, imputed


# ============================================================================
# 센서 캡핑
# ============================================================================


def flag_capping(
    values: np.ndarray,
    low: float,
    high: float,
    min_run: int = MIN_CAPPING_RUN,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """센서 측정범위에 걸린 값이 ``min_run`` 이상 연속되는 구간을 표시한다.

    값을 고치지는 않는다. 실제 값을 알 수 없는 구간이라는 표시만 남긴다.
    """
    flags = np.zeros(values.shape, dtype=bool)
    if values.size == 0:
        return flags

    for limit in (low, high):
        at_limit = np.isclose(values, limit, atol=tolerance, equal_nan=False)
        starts, ends = _run_bounds(at_limit)
        for start, end in zip(starts, ends):
            if end - start >= min_run:
                flags[start:end] = True
    return flags


# ============================================================================
# 파이프라인
# ============================================================================


def preprocess_cgm(
    df: pd.DataFrame,
    grid_minutes: int = GRID_MINUTES,
    max_gap_minutes: int = MAX_INTERPOLATION_GAP_MINUTES,
    min_capping_run: int = MIN_CAPPING_RUN,
    limits: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """공통 스키마 CGM 프레임을 리샘플링·결측처리·캡핑 플래그까지 처리한다.

    Args:
        df: :data:`~src.schema.CGM_COLUMNS` 형태의 프레임. 환자가 여러 명이어도 된다.
        limits: ``(하한, 상한)``. ``None``이면 각 행의 ``source``로 찾는다.

    Returns:
        :data:`~src.schema.PROCESSED_CGM_COLUMNS` 형태의 프레임.
    """
    if df.empty:
        return empty_processed_cgm_frame()

    parts: list[pd.DataFrame] = []
    for (pid, source), sub in df.groupby(["patient_id", "source"], observed=True):
        if sub.empty:
            continue
        low, high = limits if limits is not None else sensor_limits(str(source))

        sub = sub.sort_values("timestamp", kind="stable")
        grid, values, _ = resample_patient(
            sub["timestamp"], sub["glucose_mgdl"], grid_minutes
        )
        if grid.size == 0:
            continue

        values, imputed = interpolate_short_gaps(values, grid_minutes, max_gap_minutes)
        capped = flag_capping(values, low, high, min_capping_run)

        parts.append(
            pd.DataFrame(
                {
                    "patient_id": pid,
                    "timestamp": grid,
                    "glucose_mgdl": values,
                    "source": source,
                    "is_imputed": imputed,
                    "is_capped": capped,
                }
            )
        )

    if not parts:
        return empty_processed_cgm_frame()

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["patient_id", "timestamp"], kind="stable")
    return coerce_processed_cgm_frame(out)


def preprocess_parquet(
    in_path,
    out_path,
    patients_per_batch: int = 10,
    **kwargs,
) -> dict[str, float]:
    """interim Parquet을 읽어 전처리한 뒤 processed Parquet으로 흘려 쓴다.

    문자열 ``patient_id``는 환자 묶음 단위로만 만든다. 1,500만 행짜리 문자열
    컬럼을 한 번에 들고 있으면 1GB 가까이 먹기 때문이다.
    """
    from pathlib import Path

    from src.schema import coerce_cgm_frame
    from src.storage import read_parquet, write_parquet

    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()

    df = read_parquet(in_path)
    df["patient_id"] = df["patient_id"].astype("category")
    df["source"] = df["source"].astype("category")
    df = df.sort_values(["patient_id", "timestamp"], kind="stable").reset_index(
        drop=True
    )

    codes = df["patient_id"].cat.codes.to_numpy()
    starts = np.flatnonzero(np.concatenate(([True], codes[1:] != codes[:-1])))
    bounds = list(starts) + [len(df)]
    n_patients = len(starts)

    totals: dict[str, float] = {}
    written = 0
    for i in range(0, n_patients, patients_per_batch):
        lo = bounds[i]
        hi = bounds[min(i + patients_per_batch, n_patients)]
        batch = preprocess_cgm(coerce_cgm_frame(df.iloc[lo:hi]), **kwargs)
        if batch.empty:
            continue
        write_parquet(batch, out_path, append=written > 0)
        written += 1

        stats = summarize(batch)
        for key in ("n_slots", "n_present", "n_imputed", "n_capped"):
            totals[key] = totals.get(key, 0) + stats[key]

    totals["n_patients"] = n_patients
    if totals.get("n_slots"):
        totals["coverage"] = totals["n_present"] / totals["n_slots"]
        totals["pct_imputed"] = 100 * totals["n_imputed"] / totals["n_slots"]
        totals["pct_capped"] = 100 * totals["n_capped"] / totals["n_slots"]
    return totals


# ============================================================================
# AGP 창 유효성
# ============================================================================


def window_coverage(
    processed: pd.DataFrame,
    include_imputed: bool = True,
    grid_minutes: int = GRID_MINUTES,
) -> pd.DataFrame:
    """환자별 관찰 길이와 데이터 확보율을 계산한다.

    Args:
        include_imputed: 보간으로 채운 값을 "확보한 데이터"로 칠지 여부.
            AGP 지표 계산에는 보간값도 쓰므로 기본값은 True다.

    Returns:
        ``patient_id, start, end, days, n_slots, n_present, coverage,
        is_valid_agp, meets_recommended_days`` 컬럼의 프레임.
    """
    if processed.empty:
        return pd.DataFrame(
            columns=["patient_id", "start", "end", "days", "n_slots", "n_present",
                     "coverage", "is_valid_agp", "meets_recommended_days"]
        )

    present = processed["glucose_mgdl"].notna()
    if not include_imputed:
        present = present & ~processed["is_imputed"]

    rows = []
    for pid, sub in processed.assign(_present=present).groupby(
        "patient_id", observed=True
    ):
        start, end = sub["timestamp"].min(), sub["timestamp"].max()
        n_slots = len(sub)
        n_present = int(sub["_present"].sum())
        days = (end - start).total_seconds() / 86400.0
        coverage = n_present / n_slots if n_slots else 0.0
        rows.append(
            {
                "patient_id": pid,
                "start": start,
                "end": end,
                "days": days,
                "n_slots": n_slots,
                "n_present": n_present,
                "coverage": coverage,
                "is_valid_agp": days >= MIN_AGP_DAYS and coverage >= MIN_AGP_COVERAGE,
                "meets_recommended_days": days >= RECOMMENDED_AGP_DAYS,
            }
        )

    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


def summarize(processed: pd.DataFrame) -> dict[str, float]:
    """전처리 결과를 한눈에 볼 수 있는 요약값."""
    n = len(processed)
    if n == 0:
        return {"n_slots": 0}
    present = processed["glucose_mgdl"].notna()
    return {
        "n_slots": n,
        "n_patients": processed["patient_id"].nunique(),
        "n_present": int(present.sum()),
        "coverage": float(present.mean()),
        "n_imputed": int(processed["is_imputed"].sum()),
        "pct_imputed": float(processed["is_imputed"].mean() * 100),
        "n_capped": int(processed["is_capped"].sum()),
        "pct_capped": float(processed["is_capped"].mean() * 100),
    }


__all__ = [
    "GRID_MINUTES",
    "MAX_INTERPOLATION_GAP_MINUTES",
    "MGDL",
    "MMOL",
    "MMOL_TO_MGDL",
    "MIN_AGP_COVERAGE",
    "MIN_AGP_DAYS",
    "MIN_CAPPING_RUN",
    "PROCESSED_CGM_COLUMNS",
    "RECOMMENDED_AGP_DAYS",
    "SENSOR_LIMITS",
    "UnitInferenceError",
    "flag_capping",
    "infer_glucose_unit",
    "interpolate_short_gaps",
    "preprocess_cgm",
    "preprocess_parquet",
    "resample_patient",
    "sensor_limits",
    "snap_to_grid",
    "summarize",
    "to_mgdl",
    "window_coverage",
]
