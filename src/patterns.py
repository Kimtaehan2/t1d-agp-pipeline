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
from src.preprocess import GRID_MINUTES, MIN_AGP_COVERAGE, MIN_AGP_DAYS
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

# --- 2026-09 문헌 조사(ATTD 2019 · KDA 2025 · ADA · ISPAD 2022 · AGP 해석 문헌)로
# --- 추가한 규칙. 출처 수가 6~7개인 것은 MVP, 이벤트 로그가 있어야 정확한 것은
# --- Phase 2다. 조사 보고서의 권고 enum과 여기 키의 대응은 PATTERN_DEFINITIONS에.

#: 데이터 충분성. 14일 중 활성 데이터 70% 미만이면 다른 규칙을 돌리지 않고 이것만
#: 낸다. [국제 합의 — ATTD/KDA/ADA/JKD 일치]
LOW_COVERAGE_THRESHOLD = 100.0 * MIN_AGP_COVERAGE

#: 공복(아침) 고혈당의 대리 구간. 식사 로그 없이 "아침 식전"을 잡아야 하므로 시각
#: 고정이다. 06:00–07:00 평균을 그날의 공복값으로 보고, 창 안 일별 중앙값이 임계를
#: 넘으면 패턴이다. [팀 결정 (구간). 임계 130은 KDA 공복 목표 80–130의 상한]
FASTING_START_HOUR = 6
FASTING_END_HOUR = 7
FASTING_HYPER_THRESHOLD = 130.0
#: 공복값이 있는 날이 이보다 적으면 판정하지 않는다. [팀 결정]
FASTING_MIN_DAYS = 7

#: 급격한 하강. 15분 동안 3 mg/dL/min 이상 떨어지면 한 에피소드. [팀 결정 (횟수).
#: 3 mg/dL/min은 Dexcom '빠르게 하강(↓↓)' 화살표 기준]
RAPID_DROP_RATE = 3.0
RAPID_DROP_SPAN_MIN = 15
RAPID_DROP_MIN_EPISODES = 3

#: 저혈당 에피소드의 최소 지속시간(분). [국제 합의 — ATTD 저혈당 event 정의]
HYPO_EPISODE_MIN_MINUTES = 15
#: 이보다 길게 이어지면 지속 저혈당. [ATTD 2019 · ISPAD 2022: >120분]
PROLONGED_HYPO_MINUTES = 120

#: sick-day 케톤 위험. 270 mg/dL(15 mmol/L) 초과가 2시간 이상 이어지면 한 에피소드.
#: [DAFNE: >15 mmol/L 2시간 지속 = 펌프 실패 신호]. 창당 최소 횟수는 [팀 결정].
SICKDAY_GLUCOSE_THRESHOLD = 270.0
SICKDAY_MIN_MINUTES = 120
SICKDAY_MIN_EPISODES = 2

#: 운동 후 지연성 저혈당. 운동 종료 후 이 시간 창 안에 저혈당 에피소드가 시작하면
#: 그 운동은 '뒤따르는 저혈당이 있는 운동'이다. 앞 2시간은 급성 저혈당이라 뺀다.
#: [팀 결정 (창·횟수). ISPAD 2022 운동 챕터: 운동 후 지연 발생, 특히 야간]
EXERCISE_DELAY_START_HOURS = 2
EXERCISE_DELAY_END_HOURS = 24
EXERCISE_DELAYED_MIN_SESSIONS = 2

#: 발견 사항의 출력 순서. 조사한 출처(ATTD 2019, IDC 9단계, Dexcom 3단계, JKD 2020)가
#: 일관되게 권고하는 "데이터 충분성 → 저혈당 → 야간 저혈당 → 고혈당 → 변동성 →
#: 새벽/공복" 해석 순서다. 저혈당을 먼저 다룬다.
PATTERN_ORDER: list[str] = [
    "low_data_coverage",
    "excess_hypoglycemia",
    "excess_hypoglycemia_level2",
    "prolonged_hypoglycemia",
    "nocturnal_hypoglycemia",
    "exercise_delayed_hypoglycemia",
    "hypo_concentration",
    "low_time_in_range",
    "excess_hyperglycemia",
    "excess_hyperglycemia_level2",
    "sickday_ketone_risk",
    "postprandial_spike",
    "fasting_hyperglycemia",
    "hyper_concentration",
    "high_variability",
    "rapid_drop",
    "dawn_rise",
]

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
    "low_data_coverage": PatternDefinition(
        "low_data_coverage",
        "데이터 불충분",
        f"14일 창의 활성 데이터가 {LOW_COVERAGE_THRESHOLD:.0f}% 미만이거나 "
        f"관찰기간이 {MIN_AGP_DAYS}일 미만. 다른 패턴은 판정하지 않는다",
        "국제 합의 (ATTD 2019 · KDA 2025 · ADA · JKD). 조사 보고서 LOW_DATA_COVERAGE",
    ),
    "excess_hypoglycemia_level2": PatternDefinition(
        "excess_hypoglycemia_level2",
        "심한 저혈당 과다",
        "TBR level 2(54 mg/dL 미만 시간 비율)가 1% 초과",
        "국제 합의 AGP 목표치. 조사 보고서 HYPO_OVERALL_L2",
    ),
    "prolonged_hypoglycemia": PatternDefinition(
        "prolonged_hypoglycemia",
        "지속 저혈당",
        f"{TBR_BELOW:.0f} mg/dL 미만이 {PROLONGED_HYPO_MINUTES}분 넘게 이어진 에피소드가 1회 이상",
        "ATTD 2019 · ISPAD 2022 (>120분). 조사 보고서 PROLONGED_HYPO",
    ),
    "exercise_delayed_hypoglycemia": PatternDefinition(
        "exercise_delayed_hypoglycemia",
        "운동 후 지연성 저혈당",
        f"운동 종료 {EXERCISE_DELAY_START_HOURS}~{EXERCISE_DELAY_END_HOURS}시간 뒤에 "
        f"{TBR_BELOW:.0f} mg/dL 미만 {HYPO_EPISODE_MIN_MINUTES}분 이상이 시작된 운동이 "
        f"{EXERCISE_DELAYED_MIN_SESSIONS}회 이상. 운동 로그가 있어야 한다",
        "팀 결정 (창·횟수). ISPAD 2022 운동 챕터. 조사 보고서 EXERCISE_DELAYED_HYPO",
    ),
    "excess_hyperglycemia": PatternDefinition(
        "excess_hyperglycemia",
        "고혈당 과다",
        "TAR(180 mg/dL 초과 시간 비율)이 25% 초과",
        "국제 합의 AGP 목표치. 조사 보고서 HYPER_SUSTAINED",
    ),
    "excess_hyperglycemia_level2": PatternDefinition(
        "excess_hyperglycemia_level2",
        "심한 고혈당 과다",
        "TAR level 2(250 mg/dL 초과 시간 비율)가 5% 초과",
        "국제 합의 AGP 목표치. 조사 보고서 HYPER_SUSTAINED",
    ),
    "sickday_ketone_risk": PatternDefinition(
        "sickday_ketone_risk",
        "케톤 위험 지속 고혈당",
        f"{SICKDAY_GLUCOSE_THRESHOLD:.0f} mg/dL 초과가 {SICKDAY_MIN_MINUTES}분 이상 이어진 "
        f"에피소드가 {SICKDAY_MIN_EPISODES}회 이상. 케톤 측정 권고 대상",
        "DAFNE (>15 mmol/L 2시간). 횟수는 팀 결정. 조사 보고서 SICKDAY_KETONE_RISK",
    ),
    "fasting_hyperglycemia": PatternDefinition(
        "fasting_hyperglycemia",
        "공복/아침 고혈당",
        f"{FASTING_START_HOUR:02d}:00–{FASTING_END_HOUR:02d}:00 평균의 일별 중앙값이 "
        f"{FASTING_HYPER_THRESHOLD:.0f} mg/dL 초과 ({FASTING_MIN_DAYS}일 이상 관측)",
        "팀 결정 (구간). 임계는 KDA 2025 공복 목표 80–130. 조사 보고서 FASTING_MORNING_HYPER",
    ),
    "rapid_drop": PatternDefinition(
        "rapid_drop",
        "급격한 혈당 하강",
        f"{RAPID_DROP_SPAN_MIN}분 동안 {RAPID_DROP_RATE:.0f} mg/dL/min 이상 하강한 "
        f"에피소드가 {RAPID_DROP_MIN_EPISODES}회 이상",
        "팀 결정 (횟수). 속도 기준은 Dexcom 추세 화살표. 조사 보고서 RAPID_DROP",
    ),
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

    tbr2 = metrics.get("tbr_level2")
    if tbr2 is not None and not pd.isna(tbr2) and tbr2 > 1.0:
        out.append(_finding(
            "excess_hypoglycemia_level2", tbr2, 1.0, "%",
            f"TBR level 2 {tbr2:.1f}% (목표 1% 이하)",
        ))

    tar = metrics.get("tar")
    if tar is not None and not pd.isna(tar) and tar > 25.0:
        out.append(_finding(
            "excess_hyperglycemia", tar, 25.0, "%",
            f"TAR {tar:.1f}% (목표 25% 이하)",
        ))

    tar2 = metrics.get("tar_level2")
    if tar2 is not None and not pd.isna(tar2) and tar2 > 5.0:
        out.append(_finding(
            "excess_hyperglycemia_level2", tar2, 5.0, "%",
            f"TAR level 2 {tar2:.1f}% (목표 5% 이하)",
        ))

    return out


def _low_data_coverage(metrics: dict) -> list[dict]:
    """유효성 미달 창. 다른 규칙은 돌리지 않는다 — 해석 순서의 1단계."""
    coverage = 100.0 * float(metrics.get("coverage", float("nan")))
    days = float(metrics.get("days", float("nan")))
    reason = metrics.get("invalid_reason") or ""
    return [_finding(
        "low_data_coverage", coverage, LOW_COVERAGE_THRESHOLD, "%",
        f"활성 데이터 {coverage:.0f}%, 관찰기간 {days:.1f}일 ({reason})",
    )]


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


def _episodes(
    timestamps: pd.Series,
    mask: np.ndarray,
    grid_minutes: int,
    min_minutes: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """``mask``가 연속으로 참인 구간 중 ``min_minutes`` 이상인 것.

    결측 슬롯은 참이 아니므로 자연히 구간을 끊는다. 반환은 (시작, 끝, 분).
    """
    if mask.size == 0 or not mask.any():
        return []
    changed = np.flatnonzero(np.concatenate(([True], mask[1:] != mask[:-1])))
    starts = changed[mask[changed]]
    ends = np.concatenate((changed[1:], [mask.size]))[mask[changed]]
    out = []
    for a, b in zip(starts, ends):
        minutes = int(b - a) * grid_minutes
        if minutes >= min_minutes:
            out.append((timestamps.iloc[a], timestamps.iloc[b - 1], minutes))
    return out


def _hypo_episodes(timestamps, glucose, grid_minutes):
    low = (~np.isnan(glucose)) & (glucose < TBR_BELOW)
    return _episodes(timestamps, low, grid_minutes, HYPO_EPISODE_MIN_MINUTES)


def _prolonged_hypoglycemia(timestamps, glucose, grid_minutes) -> list[dict]:
    episodes = _hypo_episodes(timestamps, glucose, grid_minutes)
    long = [e for e in episodes if e[2] > PROLONGED_HYPO_MINUTES]
    if not long:
        return []
    longest = max(e[2] for e in long)
    return [_finding(
        "prolonged_hypoglycemia", longest, PROLONGED_HYPO_MINUTES, "분",
        f"{TBR_BELOW:.0f} mg/dL 미만이 {PROLONGED_HYPO_MINUTES}분 넘게 이어진 에피소드 "
        f"{len(long)}회 (최장 {longest}분) / 저혈당 에피소드 {len(episodes)}회",
        n_events=len(long), n_total=len(episodes),
    )]


def _sickday_ketone_risk(timestamps, glucose, grid_minutes) -> list[dict]:
    high = (~np.isnan(glucose)) & (glucose > SICKDAY_GLUCOSE_THRESHOLD)
    episodes = _episodes(timestamps, high, grid_minutes, SICKDAY_MIN_MINUTES)
    if len(episodes) < SICKDAY_MIN_EPISODES:
        return []
    longest = max(e[2] for e in episodes)
    return [_finding(
        "sickday_ketone_risk", len(episodes), SICKDAY_MIN_EPISODES, "회",
        f"{SICKDAY_GLUCOSE_THRESHOLD:.0f} mg/dL 초과가 {SICKDAY_MIN_MINUTES}분 이상 이어진 "
        f"에피소드 {len(episodes)}회 (최장 {longest}분)",
        n_events=len(episodes),
    )]


def _fasting_hyperglycemia(timestamps: pd.Series, glucose: np.ndarray) -> list[dict]:
    hours = timestamps.dt.hour.to_numpy()
    dates = timestamps.dt.normalize().to_numpy()
    window = (hours >= FASTING_START_HOUR) & (hours < FASTING_END_HOUR)
    present = ~np.isnan(glucose)

    daily = []
    for date in np.unique(dates[window]):
        block = window & (dates == date) & present
        if block.any():
            daily.append(float(glucose[block].mean()))
    if len(daily) < FASTING_MIN_DAYS:
        return []

    median = float(np.median(daily))
    if median <= FASTING_HYPER_THRESHOLD:
        return []
    return [_finding(
        "fasting_hyperglycemia", median, FASTING_HYPER_THRESHOLD, "mg/dL",
        f"{FASTING_START_HOUR:02d}:00–{FASTING_END_HOUR:02d}:00 평균의 일별 중앙값 "
        f"{median:.0f} mg/dL ({len(daily)}일 관측, 목표 {FASTING_HYPER_THRESHOLD:.0f} 이하)",
        n_total=len(daily),
    )]


def _rapid_drop(timestamps: pd.Series, glucose: np.ndarray, grid_minutes: int) -> list[dict]:
    span = max(1, RAPID_DROP_SPAN_MIN // grid_minutes)
    if glucose.size <= span:
        return []
    delta = glucose[span:] - glucose[:-span]
    # 격자가 연속일 때만 의미가 있다. 슬롯 사이 간격이 창 길이와 다르면 건너뛴다.
    ts = timestamps.to_numpy()
    gap = (ts[span:] - ts[:-span]) / np.timedelta64(1, "m")
    fast = (~np.isnan(delta)) & (gap == span * grid_minutes) & (
        delta <= -RAPID_DROP_RATE * RAPID_DROP_SPAN_MIN
    )
    # 연속으로 걸린 슬롯은 한 에피소드로 센다.
    n = len(_episodes(timestamps.iloc[span:].reset_index(drop=True), fast, grid_minutes, 0))
    if n < RAPID_DROP_MIN_EPISODES:
        return []
    steepest = float(-delta[fast].min() / RAPID_DROP_SPAN_MIN)
    return [_finding(
        "rapid_drop", n, RAPID_DROP_MIN_EPISODES, "회",
        f"{RAPID_DROP_SPAN_MIN}분에 {RAPID_DROP_RATE * RAPID_DROP_SPAN_MIN:.0f} mg/dL 이상 "
        f"하강 {n}회 (최대 {steepest:.1f} mg/dL/min)",
        n_events=n,
    )]


def _exercise_delayed_hypoglycemia(
    timestamps: pd.Series,
    glucose: np.ndarray,
    grid_minutes: int,
    sessions: np.ndarray,
) -> list[dict]:
    """``sessions``는 (시작, 지속시간 분) 배열. 지속시간을 모르면 0으로 본다."""
    if sessions.size == 0:
        return []
    episodes = _hypo_episodes(timestamps, glucose, grid_minutes)
    starts = np.array([np.datetime64(e[0]) for e in episodes], dtype="datetime64[ns]")

    followed = 0
    nocturnal = 0
    for start, minutes in sessions:
        end = np.datetime64(pd.Timestamp(start)) + np.timedelta64(int(minutes), "m")
        lo = end + np.timedelta64(EXERCISE_DELAY_START_HOURS, "h")
        hi = end + np.timedelta64(EXERCISE_DELAY_END_HOURS, "h")
        hits = starts[(starts >= lo) & (starts < hi)] if starts.size else starts
        if hits.size:
            followed += 1
            first = pd.Timestamp(hits.min())
            if NIGHT_START_HOUR <= first.hour < NIGHT_END_HOUR:
                nocturnal += 1

    if followed < EXERCISE_DELAYED_MIN_SESSIONS:
        return []
    share = 100.0 * followed / len(sessions)
    return [_finding(
        "exercise_delayed_hypoglycemia", followed, EXERCISE_DELAYED_MIN_SESSIONS, "회",
        f"운동 {len(sessions)}회 중 {followed}회({share:.0f}%)는 종료 "
        f"{EXERCISE_DELAY_START_HOURS}~{EXERCISE_DELAY_END_HOURS}시간 뒤 저혈당이 뒤따름 "
        f"(그중 야간 시작 {nocturnal}회)",
        n_events=followed, n_total=len(sessions),
    )]


# ============================================================================
# 진입점
# ============================================================================


def detect_patterns(
    processed: pd.DataFrame,
    events: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
    window_days: float = DEFAULT_WINDOW_DAYS,
    grid_minutes: int = GRID_MINUTES,
    include_low_coverage: bool = True,
) -> pd.DataFrame:
    """환자별 최근 창에서 패턴을 찾는다.

    Args:
        processed: 전처리 완료 CGM 프레임.
        events: 이벤트 프레임. 없으면 식후 스파이크는 건너뛴다.
        metrics: 이미 계산해 둔 :func:`~src.metrics.recent_metrics` 결과.
            없으면 안에서 계산한다. 창은 이 표의 ``window_start``/``window_end``를
            따르므로 지표와 패턴이 항상 같은 구간을 본다.

        include_low_coverage: 유효성 미달 창(관찰기간·확보율 미달)에 대해
            ``low_data_coverage`` 발견 사항을 낸다. 해석 순서의 1단계라 기본 켠다.
            ``False``면 옛 동작대로 그 환자는 표에 등장하지 않는다.

    Returns:
        발견 사항 표. 패턴이 하나도 없으면 빈 표를 돌려준다. 환자 안에서는
        :data:`PATTERN_ORDER`(저혈당 우선) 순으로 정렬된다.
    """
    missing = [c for c in PROCESSED_CGM_COLUMNS if c not in processed.columns]
    if missing:
        raise SchemaError(
            f"전처리 완료 프레임이 아니다. 누락된 컬럼: {missing}. "
            f"src.preprocess.preprocess_cgm을 먼저 돌려라."
        )

    if metrics is None:
        metrics = recent_metrics(processed, window_days=window_days,
                                 grid_minutes=grid_minutes,
                                 drop_invalid=not include_low_coverage)
    if metrics.empty:
        return _empty_findings()

    meals_by_patient = _meal_times(events)
    exercise_by_patient = _exercise_sessions(events)
    by_patient = {pid: sub for pid, sub in processed.groupby("patient_id",
                                                            observed=True)}

    rows: list[dict] = []
    for row in metrics.itertuples(index=False):
        sub = by_patient.get(row.patient_id)
        if sub is None:
            continue

        # 해석 순서 1단계: 데이터가 모자라면 그 사실만 알리고 나머지는 판정하지 않는다.
        if not getattr(row, "is_valid", True):
            if include_low_coverage:
                for finding in _low_data_coverage(row._asdict()):
                    finding.update(patient_id=row.patient_id, source=row.source,
                                   window_start=row.window_start, window_end=row.window_end)
                    rows.append(finding)
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

        sessions = exercise_by_patient.get(row.patient_id, np.empty((0, 2), dtype=object))
        if sessions.size:
            in_win = (sessions[:, 0] >= row.window_start) & (sessions[:, 0] < row.window_end)
            sessions = sessions[in_win]

        findings = [
            *_metric_findings(row._asdict()),
            *_prolonged_hypoglycemia(timestamps, glucose, grid_minutes),
            *_nocturnal_hypoglycemia(timestamps, glucose, grid_minutes),
            *_exercise_delayed_hypoglycemia(timestamps, glucose, grid_minutes, sessions),
            *_sickday_ketone_risk(timestamps, glucose, grid_minutes),
            *_postprandial_spike(timestamps, glucose, meals),
            *_fasting_hyperglycemia(timestamps, glucose),
            *_time_of_day_concentration(timestamps, glucose),
            *_rapid_drop(timestamps, glucose, grid_minutes),
            *_dawn_rise(timestamps, glucose),
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
    rank = {key: i for i, key in enumerate(PATTERN_ORDER)}
    table["_rank"] = table["pattern"].map(rank).fillna(len(rank))
    table = table.sort_values(["patient_id", "_rank", "pattern"], kind="stable")
    return table.drop(columns="_rank").reset_index(drop=True)


def _empty_findings() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in FINDING_COLUMNS})


def _exercise_sessions(events: pd.DataFrame | None) -> dict[str, np.ndarray]:
    """환자별 (시작 시각, 지속시간 분) 배열. 지속시간이 없으면 0."""
    if events is None or events.empty:
        return {}
    ex = events.loc[events["event_type"] == "exercise"]
    if ex.empty:
        return {}
    out = {}
    for pid, sub in ex.groupby("patient_id", observed=True):
        sub = sub.sort_values("timestamp")
        minutes = pd.to_numeric(sub["value"], errors="coerce").fillna(0).to_numpy()
        arr = np.empty((len(sub), 2), dtype=object)
        arr[:, 0] = list(sub["timestamp"])
        arr[:, 1] = minutes
        out[pid] = arr
    return out


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
    "PATTERN_ORDER",
    "PatternDefinition",
    "detect_patterns",
    "pattern_catalog",
    "summarize_patterns",
]
