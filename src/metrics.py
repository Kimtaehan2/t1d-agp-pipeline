"""AGP 지표 계산.

**전부 순수 코드다. 지표 계산에 LLM을 쓰지 않는다.**

핵심은 :func:`metrics_for_window` 하나이고, 창을 어떻게 자르느냐만 다른 두 래퍼가
그 위에 얹힌다.

- :func:`recent_metrics` — 환자당 **최근 14일** 창 1개. 앱/LLM 레이어에 넘기는 값.
  전체 기간으로 계산하면 옛 데이터가 현재 상태를 희석한다. Replace-BG 실측으로는
  TIR 70% 목표 달성 판정이 226명 중 29명에서, CV 36% 판정이 56명에서 뒤집혔다.
- :func:`sliding_metrics` — 창을 뒤에서부터 겹쳐 가며 여러 개. 모델 학습용.
  최근 14일만 쓰면 환자당 샘플이 1개뿐이라 학습이 안 된다(226개). stride 7일이면
  6,702개가 나온다. 환자 단위 train/test 분리는 이 표를 쓰는 쪽에서 지켜야 한다.

두 함수 모두 :data:`~src.schema.PROCESSED_CGM_COLUMNS` 형태의 전처리 완료 프레임을
받는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocess import GRID_MINUTES, MIN_AGP_COVERAGE, MIN_AGP_DAYS
from src.schema import PROCESSED_CGM_COLUMNS, SchemaError

# --- 지표 정의 (팀 스펙) -----------------------------------------------------

#: TIR 목표 범위 (mg/dL, 양끝 포함)
TIR_LOW = 70.0
TIR_HIGH = 180.0

#: TBR 기준. level 2는 더 심한 저혈당.
TBR_BELOW = 70.0
TBR_LEVEL2_BELOW = 54.0

#: TAR 기준. level 2는 더 심한 고혈당.
#: 스펙 표에는 level 2가 없지만 국제 합의 AGP에 포함돼 있어 팀 합의로 추가했다.
TAR_ABOVE = 180.0
TAR_LEVEL2_ABOVE = 250.0

#: GMI = 3.31 + 0.02392 × 평균혈당(mg/dL)
GMI_INTERCEPT = 3.31
GMI_SLOPE = 0.02392

#: 지표별 목표. (비교 방향, 기준값) — 스펙의 목표 열을 그대로 옮겼다.
TARGETS: dict[str, tuple[str, float]] = {
    "tir": (">=", 70.0),
    "tar": ("<=", 25.0),
    "tar_level2": ("<=", 5.0),
    "tbr": ("<=", 4.0),
    "tbr_level2": ("<=", 1.0),
    "cv": ("<=", 36.0),
}

#: 기본 창 길이(일). 스펙의 권장값이자 국제 합의 기준.
DEFAULT_WINDOW_DAYS = 14

#: 슬라이딩 창의 기본 이동 폭(일).
DEFAULT_STRIDE_DAYS = 7

METRIC_COLUMNS: list[str] = [
    "patient_id",
    "source",
    "window_start",
    "window_end",
    "days",
    "n_expected",
    "n_present",
    "coverage",
    "pct_imputed",
    "pct_capped",
    "mean_glucose",
    "sd_glucose",
    "tir",
    "tar",
    "tar_level2",
    "tbr",
    "tbr_level2",
    "gmi",
    "cv",
    "is_valid",
    "invalid_reason",
]


# ============================================================================
# 순수 수치 계산
# ============================================================================


def glucose_metrics(values: np.ndarray | pd.Series) -> dict[str, float]:
    """혈당 배열에서 AGP 지표를 계산한다. NaN은 무시한다.

    캡핑된 값(측정범위에 걸린 값)도 포함해서 센다. 39 mg/dL은 실제로 저혈당이고
    401 mg/dL은 실제로 고혈당이므로, 빼면 TIR이 부풀려진다.
    """
    v = np.asarray(values, dtype="float64")
    v = v[~np.isnan(v)]

    if v.size == 0:
        return {
            key: float("nan")
            for key in ("mean_glucose", "sd_glucose", "tir", "tar", "tar_level2",
                        "tbr", "tbr_level2", "gmi", "cv")
        }

    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")

    return {
        "mean_glucose": mean,
        "sd_glucose": sd,
        "tir": 100.0 * float(np.mean((v >= TIR_LOW) & (v <= TIR_HIGH))),
        "tar": 100.0 * float(np.mean(v > TAR_ABOVE)),
        "tar_level2": 100.0 * float(np.mean(v > TAR_LEVEL2_ABOVE)),
        "tbr": 100.0 * float(np.mean(v < TBR_BELOW)),
        "tbr_level2": 100.0 * float(np.mean(v < TBR_LEVEL2_BELOW)),
        "gmi": GMI_INTERCEPT + GMI_SLOPE * mean,
        "cv": 100.0 * sd / mean if mean else float("nan"),
    }


def evaluate_targets(metrics: pd.DataFrame | dict) -> pd.DataFrame | dict:
    """각 지표가 목표를 충족하는지 판정한다. ``{지표}_ok`` 컬럼을 만든다."""
    if isinstance(metrics, dict):
        return {
            f"{name}_ok": _meets(metrics.get(name), op, target)
            for name, (op, target) in TARGETS.items()
        }

    out = metrics.copy()
    for name, (op, target) in TARGETS.items():
        if name in out.columns:
            out[f"{name}_ok"] = [_meets(v, op, target) for v in out[name]]
    return out


def _meets(value, op: str, target: float):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NA
    return value >= target if op == ">=" else value <= target


# ============================================================================
# 창 단위 계산
# ============================================================================


def metrics_for_window(
    timestamps: np.ndarray,
    glucose: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
    is_imputed: np.ndarray | None = None,
    is_capped: np.ndarray | None = None,
    grid_minutes: int = GRID_MINUTES,
    min_days: float = MIN_AGP_DAYS,
    min_coverage: float = MIN_AGP_COVERAGE,
) -> dict:
    """반열린 구간 ``[start, end)`` 하나에 대한 지표.

    ``timestamps``는 오름차순 정렬돼 있어야 한다(전처리 결과는 항상 정렬돼 있다).

    확보율의 분모는 프레임에 실제로 들어 있는 행 수가 아니라 **창 길이로부터
    계산한 격자 슬롯 수**다. 창 끝이 데이터 범위를 벗어나도 결측으로 잡히게
    하기 위해서다.
    """
    lo = int(np.searchsorted(timestamps, np.datetime64(start), side="left"))
    hi = int(np.searchsorted(timestamps, np.datetime64(end), side="left"))

    window_values = glucose[lo:hi]
    days = (end - start).total_seconds() / 86400.0
    n_expected = int(round((end - start).total_seconds() / (grid_minutes * 60)))
    present = ~np.isnan(window_values)
    n_present = int(present.sum())
    coverage = n_present / n_expected if n_expected else 0.0

    def _share(flags: np.ndarray | None) -> float:
        if flags is None or n_present == 0:
            return float("nan")
        return 100.0 * float(np.sum(flags[lo:hi] & present)) / n_present

    reason = None
    if days < min_days:
        reason = f"관찰기간 {days:.1f}일 < {min_days}일"
    elif coverage < min_coverage:
        reason = f"확보율 {coverage:.1%} < {min_coverage:.0%}"

    return {
        "window_start": start,
        "window_end": end,
        "days": days,
        "n_expected": n_expected,
        "n_present": n_present,
        "coverage": coverage,
        "pct_imputed": _share(is_imputed),
        "pct_capped": _share(is_capped),
        **glucose_metrics(window_values),
        "is_valid": reason is None,
        "invalid_reason": reason,
    }


# ============================================================================
# 환자 단위 래퍼
# ============================================================================


def _check_processed(processed: pd.DataFrame) -> None:
    missing = [c for c in PROCESSED_CGM_COLUMNS if c not in processed.columns]
    if missing:
        raise SchemaError(
            f"전처리 완료 프레임이 아니다. 누락된 컬럼: {missing}. "
            f"src.preprocess.preprocess_cgm을 먼저 돌려라."
        )


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in METRIC_COLUMNS})


def _window_ends(
    last: pd.Timestamp,
    first: pd.Timestamp,
    window: pd.Timedelta,
    stride: pd.Timedelta,
    grid: pd.Timedelta,
    limit: int | None,
) -> list[pd.Timestamp]:
    """창의 끝점들을 **최신 것부터** 만든다.

    끝점을 ``마지막 관측 + 격자 1칸``으로 잡아 반열린 구간이 마지막 관측을
    포함하게 한다. 이렇게 하면 슬라이딩 창의 첫 번째가 최근 창과 정확히 같아진다.
    """
    ends: list[pd.Timestamp] = []
    end = last + grid
    while end - window >= first:
        ends.append(end)
        if limit is not None and len(ends) >= limit:
            break
        end = end - stride
    return ends


def _metrics_table(
    processed: pd.DataFrame,
    window_days: float,
    stride_days: float | None,
    max_windows: int | None,
    grid_minutes: int,
    min_days: float,
    min_coverage: float,
    drop_invalid: bool,
) -> pd.DataFrame:
    _check_processed(processed)
    if processed.empty:
        return _empty_table()

    window = pd.Timedelta(days=window_days)
    stride = pd.Timedelta(days=stride_days if stride_days else window_days)
    grid = pd.Timedelta(minutes=grid_minutes)

    rows: list[dict] = []
    for (pid, source), sub in processed.groupby(
        ["patient_id", "source"], observed=True
    ):
        sub = sub.sort_values("timestamp", kind="stable")
        timestamps = sub["timestamp"].to_numpy()
        glucose = sub["glucose_mgdl"].to_numpy(dtype="float64")
        imputed = sub["is_imputed"].to_numpy(dtype=bool)
        capped = sub["is_capped"].to_numpy(dtype=bool)

        first = sub["timestamp"].iloc[0]
        last = sub["timestamp"].iloc[-1]

        for end in _window_ends(last, first, window, stride, grid, max_windows):
            row = metrics_for_window(
                timestamps,
                glucose,
                end - window,
                end,
                is_imputed=imputed,
                is_capped=capped,
                grid_minutes=grid_minutes,
                min_days=min_days,
                min_coverage=min_coverage,
            )
            row["patient_id"] = pid
            row["source"] = source
            rows.append(row)

    if not rows:
        return _empty_table()

    table = pd.DataFrame(rows).loc[:, METRIC_COLUMNS]
    if drop_invalid:
        table = table.loc[table["is_valid"]]
    return table.sort_values(["patient_id", "window_start"]).reset_index(drop=True)


def recent_metrics(
    processed: pd.DataFrame,
    window_days: float = DEFAULT_WINDOW_DAYS,
    grid_minutes: int = GRID_MINUTES,
    min_days: float = MIN_AGP_DAYS,
    min_coverage: float = MIN_AGP_COVERAGE,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """환자별 **가장 최근** 창 하나의 지표. 앱/LLM 레이어에 넘기는 값.

    Args:
        drop_invalid: 유효성 미달 창을 빼고 돌려준다(스펙 기본). ``False``로 두면
            ``is_valid``·``invalid_reason``으로 탈락 사유를 볼 수 있다.
    """
    return _metrics_table(
        processed,
        window_days=window_days,
        stride_days=None,
        max_windows=1,
        grid_minutes=grid_minutes,
        min_days=min_days,
        min_coverage=min_coverage,
        drop_invalid=drop_invalid,
    )


def sliding_metrics(
    processed: pd.DataFrame,
    window_days: float = DEFAULT_WINDOW_DAYS,
    stride_days: float = DEFAULT_STRIDE_DAYS,
    grid_minutes: int = GRID_MINUTES,
    min_days: float = MIN_AGP_DAYS,
    min_coverage: float = MIN_AGP_COVERAGE,
    drop_invalid: bool = True,
    max_windows: int | None = None,
) -> pd.DataFrame:
    """환자별로 창을 겹쳐 가며 계산한다. 모델 학습 데이터용.

    창은 마지막 관측에서 시작해 과거로 ``stride_days``씩 물러난다. 따라서 각
    환자의 첫 행은 :func:`recent_metrics` 결과와 같다.

    Warning:
        한 환자에서 나온 창들은 서로 겹치고 상관도 높다. **train/test는 반드시
        환자 단위로 나눠야 한다.** 창 단위 랜덤 분할은 누수다.
    """
    return _metrics_table(
        processed,
        window_days=window_days,
        stride_days=stride_days,
        max_windows=max_windows,
        grid_minutes=grid_minutes,
        min_days=min_days,
        min_coverage=min_coverage,
        drop_invalid=drop_invalid,
    )


__all__ = [
    "DEFAULT_STRIDE_DAYS",
    "DEFAULT_WINDOW_DAYS",
    "GMI_INTERCEPT",
    "GMI_SLOPE",
    "METRIC_COLUMNS",
    "TARGETS",
    "TAR_ABOVE",
    "TAR_LEVEL2_ABOVE",
    "TBR_BELOW",
    "TBR_LEVEL2_BELOW",
    "TIR_HIGH",
    "TIR_LOW",
    "evaluate_targets",
    "glucose_metrics",
    "metrics_for_window",
    "recent_metrics",
    "sliding_metrics",
]
