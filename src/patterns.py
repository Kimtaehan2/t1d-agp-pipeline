"""규칙 기반 패턴 탐지.

**ML을 쓰지 않는다.** 임계값은 데이터에서 학습하는 것이 아니라 국제 합의 지침에서
오고, 정답 라벨도 없다. 규칙으로 라벨을 만들어 그 라벨을 맞히는 모델을 학습시키는
것은 순환이다. ML은 ``src/models/``의 예측 모델에서 쓴다.

**원인을 단정하지 않는다.** 이 모듈은 "무엇이 관측됐는가"만 낸다. 예를 들어 새벽에
혈당이 오르는 것이 새벽 현상인지 기저 인슐린 부족인지 판단하지 않고, "03–07시 평균
상승폭이 34 mg/dL"이라는 사실만 기록한다. 원인 설명과 권고는 지침을 근거로
LLM/RAG 레이어가 한다. 이 파일에 원인이나 조치를 넣지 마라.

출력은 발견 사항(finding) 한 줄 = 한 패턴이고, LLM이 그대로 인용할 수 있도록
측정값·기준값·근거 문장을 같이 담는다. 규칙의 정의문은 :func:`pattern_catalog`로
따로 꺼낼 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import (
    DEFAULT_WINDOW_DAYS,
    TAR_ABOVE,
    TBR_BELOW,
    recent_metrics,
)
from src.preprocess import GRID_MINUTES
from src.schema import PROCESSED_CGM_COLUMNS, SchemaError

# ============================================================================
# 임계값
# ============================================================================
#
# [국제 합의] 표시가 있는 값은 AGP 국제 합의에 명시된 목표치라 우리가 정할 여지가
# 없다. [팀 결정] 표시는 확정된 수치 정의가 없어 우리가 고른 값이다. 바꾸려면
# 여기만 고치면 된다.

#: 야간 구간. [팀 결정]
NIGHT_START_HOUR = 0
NIGHT_END_HOUR = 6

#: 창(14일) 안에서 야간 저혈당이 이 일수 이상 나타나면 패턴으로 본다. [팀 결정]
NIGHT_HYPO_MIN_DAYS = 3

#: 야간 저혈당으로 세기 위한 최소 지속 시간(분). 0이면 한 번이라도 70 미만이면
#: 그날 밤을 저혈당으로 센다. 15로 두면 국제 합의의 '저혈당 이벤트' 정의와 같아진다.
#: [팀 결정]
NIGHT_HYPO_MIN_DURATION_MIN = 0

#: 새벽 상승 구간과 판정 임계(mg/dL). [팀 결정]
#:
#: 구간 양끝의 30분 평균을 비교한다(03:00–03:30 vs 06:30–07:00).
#:
#: 주의: 고전적인 새벽 현상 정의는 고정 시각이 아니라 **야간 최저점**에서
#: 기상 전까지의 상승을 본다. 여기서는 03:00 고정 앵커를 쓰므로 최저점이 다른
#: 시각에 오는 환자는 상승폭이 과소평가된다. Replace-BG 실측으로 03–07시 평균
#: 상승폭은 중앙값 −12.6 mg/dL(즉 대체로 하강)이고 30 이상인 환자는 219명 중
#: 7명이다. 최저점 기준으로 바꾸면 수치가 달라진다. — 확인 필요
DAWN_START_HOUR = 3
DAWN_END_HOUR = 7
DAWN_ANCHOR_MINUTES = 30
DAWN_RISE_THRESHOLD = 30.0

#: 식후 관찰 시간과 상승폭 임계(mg/dL). [팀 결정]
POSTPRANDIAL_HOURS = 2
POSTPRANDIAL_RISE_THRESHOLD = 80.0
#: 식사 중 이 비율 이상이 임계를 넘어야 패턴으로 본다. [팀 결정]
#:
#: Replace-BG 실측(최근 14일 창, 8,166끼) 기준으로 식후 2시간 상승폭 분포는
#: 중앙값 34 / 75분위 67 / 90분위 102 mg/dL이고, 80 이상인 식사는 전체의 18.4%다.
#: 환자별 '80 이상인 식사 비율'은 중앙값 17.1%, 75분위 23.7%, **최댓값 50.0%**라
#: 임계를 50%로 두면 219명 중 1명만 걸려 규칙이 사실상 죽는다. 반대로 20% 근처로
#: 두면 중앙값 환자가 걸려 '패턴'이라 부를 수 없다. 75분위 위인 30%로 잡아
#: 상위 11%(25명)를 '반복적으로 크게 오르는' 환자로 본다.
POSTPRANDIAL_MIN_SHARE = 30.0
#: 평가 가능한 식사가 이보다 적으면 판정하지 않는다. [팀 결정]
POSTPRANDIAL_MIN_MEALS = 5

#: 시간대 집중을 볼 때 하루를 나누는 단위(시간). [팀 결정]
HOUR_BIN_HOURS = 2
#: 구간이 환자 전체 대비 이 배수 이상이어야 '몰린다'고 본다. [팀 결정]
CONCENTRATION_RATIO = 2.0
#: 구간의 절대 기준은 AGP 목표치를 그대로 쓴다. [국제 합의]
CONCENTRATION_MIN_TBR = 4.0
CONCENTRATION_MIN_TAR = 25.0

FINDING_COLUMNS: list[str] = [
    "patient_id",
    "source",
    "window_start",
    "window_end",
    "pattern",
    "value",
    "threshold",
    "unit",
    "n_events",
    "n_total",
    "detail",
]


@dataclass(frozen=True)
class PatternDefinition:
    """규칙 하나의 정의. LLM 레이어가 근거로 인용할 수 있게 문장으로 적는다."""

    key: str
    name: str
    definition: str
    basis: str


PATTERN_DEFINITIONS: dict[str, PatternDefinition] = {
    "nocturnal_hypoglycemia": PatternDefinition(
        "nocturnal_hypoglycemia",
        "야간 저혈당",
        f"{NIGHT_START_HOUR:02d}:00–{NIGHT_END_HOUR:02d}:00에 혈당이 "
        f"{TBR_BELOW:.0f} mg/dL 미만인 날이 14일 중 {NIGHT_HYPO_MIN_DAYS}일 이상",
        "팀 결정 (구간·일수). 저혈당 기준 70 mg/dL은 국제 합의",
    ),
    "excess_hypoglycemia": PatternDefinition(
        "excess_hypoglycemia",
        "전반적 저혈당 과다",
        "TBR(70 mg/dL 미만 시간 비율)이 4% 초과",
        "국제 합의 AGP 목표치",
    ),
    "low_time_in_range": PatternDefinition(
        "low_time_in_range",
        "목표범위 미달",
        "TIR(70–180 mg/dL 시간 비율)이 70% 미만",
        "국제 합의 AGP 목표치",
    ),
    "high_variability": PatternDefinition(
        "high_variability",
        "높은 변동성",
        "CV(표준편차/평균)가 36% 초과",
        "국제 합의 AGP 목표치",
    ),
    "dawn_rise": PatternDefinition(
        "dawn_rise",
        "새벽 상승",
        f"{DAWN_START_HOUR:02d}:00–{DAWN_END_HOUR:02d}:00 사이 평균 상승폭이 "
        f"{DAWN_RISE_THRESHOLD:.0f} mg/dL 이상",
        "팀 결정. 원인(새벽 현상/기저 부족)은 판단하지 않는다",
    ),
    "postprandial_spike": PatternDefinition(
        "postprandial_spike",
        "식후 스파이크",
        f"식사 후 {POSTPRANDIAL_HOURS}시간 이내 상승폭이 "
        f"{POSTPRANDIAL_RISE_THRESHOLD:.0f} mg/dL 이상인 식사가 "
        f"{POSTPRANDIAL_MIN_SHARE:.0f}% 이상",
        "팀 결정",
    ),
    "hypo_concentration": PatternDefinition(
        "hypo_concentration",
        "저혈당 집중 시간대",
        f"{HOUR_BIN_HOURS}시간 구간의 TBR이 환자 전체의 "
        f"{CONCENTRATION_RATIO:.0f}배 이상이면서 {CONCENTRATION_MIN_TBR:.0f}% 초과",
        "팀 결정 (배수). 절대 기준은 국제 합의 목표치",
    ),
    "hyper_concentration": PatternDefinition(
        "hyper_concentration",
        "고혈당 집중 시간대",
        f"{HOUR_BIN_HOURS}시간 구간의 TAR이 환자 전체의 "
        f"{CONCENTRATION_RATIO:.0f}배 이상이면서 {CONCENTRATION_MIN_TAR:.0f}% 초과",
        "팀 결정 (배수). 절대 기준은 국제 합의 목표치",
    ),
}


def pattern_catalog() -> pd.DataFrame:
    """규칙 정의 표. ``pattern`` 컬럼으로 발견 사항과 조인해 쓴다."""
    return pd.DataFrame(
        [
            {"pattern": d.key, "name": d.name, "definition": d.definition,
             "basis": d.basis}
            for d in PATTERN_DEFINITIONS.values()
        ]
    )


# ============================================================================
# 규칙
# ============================================================================


def _finding(pattern: str, value, threshold, unit, detail, n_events=np.nan,
             n_total=np.nan) -> dict:
    return {
        "pattern": pattern,
        "value": float(value) if value is not None else np.nan,
        "threshold": float(threshold),
        "unit": unit,
        "n_events": n_events,
        "n_total": n_total,
        "detail": detail,
    }


def _metric_findings(metrics: dict) -> list[dict]:
    """지표 임계 기반 패턴 3종. 목표치가 국제 합의로 정해져 있어 논란이 없다."""
    out = []

    tbr = metrics.get("tbr")
    if tbr is not None and not pd.isna(tbr) and tbr > 4.0:
        out.append(_finding(
            "excess_hypoglycemia", tbr, 4.0, "%",
            f"TBR {tbr:.1f}% (목표 4% 이하)",
        ))

    tir = metrics.get("tir")
    if tir is not None and not pd.isna(tir) and tir < 70.0:
        out.append(_finding(
            "low_time_in_range", tir, 70.0, "%",
            f"TIR {tir:.1f}% (목표 70% 이상)",
        ))

    cv = metrics.get("cv")
    if cv is not None and not pd.isna(cv) and cv > 36.0:
        out.append(_finding(
            "high_variability", cv, 36.0, "%",
            f"CV {cv:.1f}% (목표 36% 이하)",
        ))

    return out


def _nocturnal_hypoglycemia(
    timestamps: pd.Series,
    glucose: np.ndarray,
    grid_minutes: int,
) -> list[dict]:
    hours = timestamps.dt.hour.to_numpy()
    dates = timestamps.dt.normalize().to_numpy()
    night = (hours >= NIGHT_START_HOUR) & (hours < NIGHT_END_HOUR)
    if not night.any():
        return []

    min_slots = max(1, int(np.ceil(NIGHT_HYPO_MIN_DURATION_MIN / grid_minutes)))
    present = ~np.isnan(glucose)
    low = present & (glucose < TBR_BELOW)

    nights_observed = 0
    nights_with_hypo = 0
    for date in np.unique(dates[night]):
        block = night & (dates == date)
        if not (present & block).any():
            continue
        nights_observed += 1
        if _longest_run(low[block]) >= min_slots:
            nights_with_hypo += 1

    if nights_with_hypo < NIGHT_HYPO_MIN_DAYS:
        return []

    return [_finding(
        "nocturnal_hypoglycemia", nights_with_hypo, NIGHT_HYPO_MIN_DAYS, "일",
        f"{NIGHT_START_HOUR:02d}:00–{NIGHT_END_HOUR:02d}:00에 "
        f"{TBR_BELOW:.0f} mg/dL 미만이 관측된 밤 {nights_with_hypo}일 "
        f"/ 야간 데이터가 있는 밤 {nights_observed}일",
        n_events=nights_with_hypo, n_total=nights_observed,
    )]


def _longest_run(mask: np.ndarray) -> int:
    """``True``가 연속된 최대 길이."""
    if mask.size == 0 or not mask.any():
        return 0
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def _dawn_rise(timestamps: pd.Series, glucose: np.ndarray) -> list[dict]:
    minutes = (timestamps.dt.hour * 60 + timestamps.dt.minute).to_numpy()
    dates = timestamps.dt.normalize().to_numpy()

    start_lo = DAWN_START_HOUR * 60
    start_hi = start_lo + DAWN_ANCHOR_MINUTES
    end_hi = DAWN_END_HOUR * 60
    end_lo = end_hi - DAWN_ANCHOR_MINUTES

    early = (minutes >= start_lo) & (minutes < start_hi)
    late = (minutes >= end_lo) & (minutes < end_hi)

    rises = []
    for date in np.unique(dates):
        same = dates == date
        a = glucose[same & early]
        b = glucose[same & late]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if a.size and b.size:
            rises.append(float(b.mean() - a.mean()))

    if not rises:
        return []

    mean_rise = float(np.mean(rises))
    if mean_rise < DAWN_RISE_THRESHOLD:
        return []

    return [_finding(
        "dawn_rise", mean_rise, DAWN_RISE_THRESHOLD, "mg/dL",
        f"{DAWN_START_HOUR:02d}:00–{DAWN_END_HOUR:02d}:00 평균 상승폭 "
        f"{mean_rise:+.1f} mg/dL ({len(rises)}일 관측)",
        n_total=len(rises),
    )]


def _postprandial_spike(
    timestamps: pd.Series,
    glucose: np.ndarray,
    meal_times: np.ndarray,
) -> list[dict]:
    if meal_times.size == 0:
        return []

    ts = timestamps.to_numpy()
    horizon = np.timedelta64(POSTPRANDIAL_HOURS * 60, "m")
    rises = []

    for meal in meal_times:
        base_idx = int(np.searchsorted(ts, meal, side="right")) - 1
        if base_idx < 0:
            continue
        baseline = glucose[base_idx]
        if np.isnan(baseline):
            continue
        hi = int(np.searchsorted(ts, meal + horizon, side="right"))
        after = glucose[base_idx + 1:hi]
        after = after[~np.isnan(after)]
        if after.size == 0:
            continue
        rises.append(float(after.max() - baseline))

    if len(rises) < POSTPRANDIAL_MIN_MEALS:
        return []

    arr = np.array(rises)
    n_spikes = int((arr >= POSTPRANDIAL_RISE_THRESHOLD).sum())
    share = 100.0 * n_spikes / arr.size
    if share < POSTPRANDIAL_MIN_SHARE:
        return []

    return [_finding(
        "postprandial_spike", share, POSTPRANDIAL_MIN_SHARE, "%",
        f"식후 {POSTPRANDIAL_HOURS}시간 이내 상승폭이 "
        f"{POSTPRANDIAL_RISE_THRESHOLD:.0f} mg/dL 이상인 식사 "
        f"{n_spikes}회 / {arr.size}회 ({share:.0f}%), "
        f"평균 상승폭 {arr.mean():+.0f} mg/dL",
        n_events=n_spikes, n_total=int(arr.size),
    )]


def _time_of_day_concentration(
    timestamps: pd.Series, glucose: np.ndarray
) -> list[dict]:
    present = ~np.isnan(glucose)
    if not present.any():
        return []

    values = glucose[present]
    overall_tbr = 100.0 * float(np.mean(values < TBR_BELOW))
    overall_tar = 100.0 * float(np.mean(values > TAR_ABOVE))

    bins = (timestamps.dt.hour.to_numpy() // HOUR_BIN_HOURS)[present]
    out = []

    for b in np.unique(bins):
        chunk = values[bins == b]
        if chunk.size == 0:
            continue
        lo_h = int(b) * HOUR_BIN_HOURS
        hi_h = lo_h + HOUR_BIN_HOURS
        band = f"{lo_h:02d}:00–{hi_h:02d}:00"

        tbr = 100.0 * float(np.mean(chunk < TBR_BELOW))
        if tbr > CONCENTRATION_MIN_TBR and tbr >= CONCENTRATION_RATIO * overall_tbr:
            out.append(_finding(
                "hypo_concentration", tbr, CONCENTRATION_MIN_TBR, "%",
                f"{band} 구간 TBR {tbr:.1f}% (환자 전체 {overall_tbr:.1f}%)",
                n_total=int(chunk.size),
            ))

        tar = 100.0 * float(np.mean(chunk > TAR_ABOVE))
        if tar > CONCENTRATION_MIN_TAR and tar >= CONCENTRATION_RATIO * overall_tar:
            out.append(_finding(
                "hyper_concentration", tar, CONCENTRATION_MIN_TAR, "%",
                f"{band} 구간 TAR {tar:.1f}% (환자 전체 {overall_tar:.1f}%)",
                n_total=int(chunk.size),
            ))

    return out


# ============================================================================
# 진입점
# ============================================================================


def detect_patterns(
    processed: pd.DataFrame,
    events: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
    window_days: float = DEFAULT_WINDOW_DAYS,
    grid_minutes: int = GRID_MINUTES,
) -> pd.DataFrame:
    """환자별 최근 창에서 패턴을 찾는다.

    Args:
        processed: 전처리 완료 CGM 프레임.
        events: 이벤트 프레임. 없으면 식후 스파이크는 건너뛴다.
        metrics: 이미 계산해 둔 :func:`~src.metrics.recent_metrics` 결과.
            없으면 안에서 계산한다. 창은 이 표의 ``window_start``/``window_end``를
            따르므로 지표와 패턴이 항상 같은 구간을 본다.

    Returns:
        발견 사항 표. 패턴이 하나도 없으면 빈 표를 돌려준다. 유효 창이 없는 환자
        (관찰기간·확보율 미달)는 애초에 등장하지 않는다.
    """
    missing = [c for c in PROCESSED_CGM_COLUMNS if c not in processed.columns]
    if missing:
        raise SchemaError(
            f"전처리 완료 프레임이 아니다. 누락된 컬럼: {missing}. "
            f"src.preprocess.preprocess_cgm을 먼저 돌려라."
        )

    if metrics is None:
        metrics = recent_metrics(processed, window_days=window_days,
                                 grid_minutes=grid_minutes)
    if metrics.empty:
        return _empty_findings()

    meals_by_patient = _meal_times(events)
    by_patient = {pid: sub for pid, sub in processed.groupby("patient_id",
                                                            observed=True)}

    rows: list[dict] = []
    for row in metrics.itertuples(index=False):
        sub = by_patient.get(row.patient_id)
        if sub is None:
            continue

        sub = sub.sort_values("timestamp", kind="stable")
        in_window = (sub["timestamp"] >= row.window_start) & (
            sub["timestamp"] < row.window_end
        )
        sub = sub.loc[in_window]
        if sub.empty:
            continue

        timestamps = sub["timestamp"].reset_index(drop=True)
        glucose = sub["glucose_mgdl"].to_numpy(dtype="float64")

        meals = meals_by_patient.get(row.patient_id, np.empty(0, dtype="datetime64[ns]"))
        if meals.size:
            meals = meals[
                (meals >= np.datetime64(row.window_start))
                & (meals < np.datetime64(row.window_end))
            ]

        findings = [
            *_metric_findings(row._asdict()),
            *_nocturnal_hypoglycemia(timestamps, glucose, grid_minutes),
            *_dawn_rise(timestamps, glucose),
            *_postprandial_spike(timestamps, glucose, meals),
            *_time_of_day_concentration(timestamps, glucose),
        ]

        for finding in findings:
            finding.update(
                patient_id=row.patient_id,
                source=row.source,
                window_start=row.window_start,
                window_end=row.window_end,
            )
            rows.append(finding)

    if not rows:
        return _empty_findings()

    table = pd.DataFrame(rows).loc[:, FINDING_COLUMNS]
    return table.sort_values(["patient_id", "pattern"]).reset_index(drop=True)


def _empty_findings() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in FINDING_COLUMNS})


def _meal_times(events: pd.DataFrame | None) -> dict[str, np.ndarray]:
    if events is None or events.empty:
        return {}
    meals = events.loc[events["event_type"] == "meal"]
    if meals.empty:
        return {}
    return {
        pid: np.sort(sub["timestamp"].to_numpy())
        for pid, sub in meals.groupby("patient_id", observed=True)
    }


def summarize_patterns(findings: pd.DataFrame, n_patients: int | None = None) -> pd.DataFrame:
    """패턴별로 몇 명에게서 나왔는지 센다."""
    if findings.empty:
        return pd.DataFrame(columns=["pattern", "name", "n_patients", "pct_patients"])

    counts = (
        findings.groupby("pattern", observed=True)["patient_id"]
        .nunique()
        .rename("n_patients")
        .reset_index()
    )
    counts["name"] = counts["pattern"].map(
        {k: v.name for k, v in PATTERN_DEFINITIONS.items()}
    )
    if n_patients:
        counts["pct_patients"] = 100.0 * counts["n_patients"] / n_patients
    return counts.sort_values("n_patients", ascending=False).reset_index(drop=True)


__all__ = [
    "FINDING_COLUMNS",
    "PATTERN_DEFINITIONS",
    "PatternDefinition",
    "detect_patterns",
    "pattern_catalog",
    "summarize_patterns",
]
