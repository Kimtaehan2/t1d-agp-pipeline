"""T1D-UOM 리플레이 적합성 분석 스크립트.

판정 보고서(``docs/t1d_uom/판정보고서.md``)의 숫자를 만든 코드다. 파이프라인
본체가 아니라 **일회성 분석**이므로 ``src/``가 이 파일을 import하지 않는다.
원본 읽기는 ``src.loaders.t1d_uom``에 맡기고 여기서는 창·이벤트·패턴만 센다.

    python scripts/explore_t1d_uom.py windows
    python scripts/explore_t1d_uom.py events
    python scripts/explore_t1d_uom.py patterns
    python scripts/explore_t1d_uom.py all
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.loaders.t1d_uom import MODALITY, load
from src.metrics import TAR_ABOVE, TBR_BELOW

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "t1d_uom"

# --- 창 유효성 기준 ----------------------------------------------------------

WINDOW_DAYS = 14
COVERAGE_MIN = 0.70
VALID_DAYS_MIN = 10

# --- 이벤트 밀도 기준 --------------------------------------------------------

MEAL_PER_DAY_MIN = 2.0
BOLUS_PER_DAY_MIN = 2.0
PAIRS_PER_WINDOW_MIN = 20
#: 식사와 볼러스를 한 쌍으로 볼 최대 시간차(분).
PAIR_WINDOW_MIN = 120

# --- 패턴 임계값 -------------------------------------------------------------
#
# 기준선 값 그대로다. 데이터에 맞춰 조정하지 않는다.

HYPO_MIN_MINUTES = 15
NOCTURNAL_MIN_DAYS = 3
NIGHT_END_HOUR = 6
CV_HIGH = 36.0
TIR_MIN, TBR_MAX = 70.0, 4.0


def _load():
    data = load()
    meta = data.patients.set_index("patient_id")
    return data.cgm, data.events, meta


# ============================================================================
# 창 유효성
# ============================================================================


def _daily_coverage(ts: pd.Series, step: int) -> pd.Series:
    """일자별 '실제 관측이 들어온 슬롯 / 하루 슬롯 수'.

    분모는 **환자 자신의 센서 주기**다. 5분 격자로 고정하면 15분 센서 환자는
    최대 0.33밖에 못 나와서, 데이터 품질이 아니라 기기 주기 때문에 탈락한다.
    보간은 하지 않는다. 실제 값이 들어온 슬롯만 센다.
    """
    slots = ts.dt.floor(f"{step}min").drop_duplicates()
    per_day = slots.groupby(slots.dt.date).size()
    full = pd.Series(
        0, index=pd.date_range(ts.min().date(), ts.max().date(), freq="D").date
    )
    full.update(per_day)
    return (full / (1440 / step)).clip(upper=1.0)


def stage_windows(cgm, events, meta) -> pd.DataFrame:
    rows = []
    for pid, sub in cgm.groupby("patient_id", observed=True):
        step = int(meta.loc[pid, "sensor_interval_min"])
        cov_native = _daily_coverage(sub["timestamp"], step)
        cov_5min = _daily_coverage(sub["timestamp"], 5)
        days = pd.Index(cov_native.index)

        for i in range(len(days) - WINDOW_DAYS + 1):
            sel = days[i : i + WINDOW_DAYS]
            c = cov_native.loc[sel]
            rows.append(
                {
                    "patient_id": pid,
                    "modality": meta.loc[pid, "modality"],
                    "sensor_interval_min": step,
                    "window_start": sel[0],
                    "window_end": sel[-1],
                    "coverage": round(float(c.mean()), 4),
                    "coverage_5min_grid": round(float(cov_5min.loc[sel].mean()), 4),
                    "valid_days": int((c >= COVERAGE_MIN).sum()),
                }
            )

    df = pd.DataFrame(rows)
    df["passes"] = (df["coverage"] >= COVERAGE_MIN) & (df["valid_days"] >= VALID_DAYS_MIN)
    df.to_csv(OUT / "window_candidates.csv", index=False)

    print(f"창 {len(df)}개 중 통과 {int(df['passes'].sum())}개")
    print(
        df.groupby(["patient_id", "modality", "sensor_interval_min"])
        .agg(창=("passes", "size"), 통과=("passes", "sum"),
             coverage최대=("coverage", "max"), _5분격자최대=("coverage_5min_grid", "max"))
        .to_string()
    )
    return df


# ============================================================================
# 이벤트 밀도
# ============================================================================


def stage_events(cgm, events, meta, windows: pd.DataFrame) -> pd.DataFrame:
    win = windows[windows["passes"]]
    meals_all = events.query("event_type == 'meal'")
    bolus_all = events.query("event_type == 'insulin_bolus'")

    rows, deltas = [], []
    for pid, sub in win.groupby("patient_id", observed=True):
        meals = meals_all[meals_all["patient_id"] == pid]
        bolus = bolus_all[bolus_all["patient_id"] == pid]

        for w in sub.itertuples(index=False):
            lo = pd.Timestamp(w.window_start)
            hi = pd.Timestamp(w.window_end) + pd.Timedelta(days=1)
            m = meals[meals["timestamp"].between(lo, hi)]
            b = bolus[bolus["timestamp"].between(lo, hi)]

            n_pairs, d = _pair_meals_to_boluses(m, b)
            deltas.extend(d)

            seen = set()
            if len(m):
                seen |= set(m["timestamp"].dt.date)
            if len(b):
                seen |= set(b["timestamp"].dt.date)

            rows.append(
                {
                    "patient_id": pid,
                    "modality": w.modality,
                    "window_start": w.window_start,
                    "meals_per_day": len(m) / WINDOW_DAYS,
                    "bolus_per_day": len(b) / WINDOW_DAYS,
                    "carbs_present_pct": (
                        100 * m["value"].notna().mean() if len(m) else 0.0
                    ),
                    "meal_bolus_pairs": n_pairs,
                    "empty_day_pct": 100 * (WINDOW_DAYS - len(seen)) / WINDOW_DAYS,
                }
            )

    df = pd.DataFrame(rows)
    df["replay_ok"] = (df["meals_per_day"] >= MEAL_PER_DAY_MIN) & (
        df["bolus_per_day"] >= BOLUS_PER_DAY_MIN
    )
    df.to_csv(OUT / "event_density.csv", index=False)

    per_pat = (
        df.groupby(["patient_id", "modality"])
        .agg(
            창=("meals_per_day", "size"),
            식사_일=("meals_per_day", "median"),
            볼러스_일=("bolus_per_day", "median"),
            탄수화물_pct=("carbs_present_pct", "median"),
            식사볼러스쌍=("meal_bolus_pairs", "median"),
            빈날_pct=("empty_day_pct", "median"),
            리플레이가능창=("replay_ok", "sum"),
        )
        .round(2)
    )
    print(per_pat.to_string())
    print()
    print(f"CGM·이벤트 둘 다 통과한 창: {int(df['replay_ok'].sum())}/{len(df)}")

    s = pd.Series(deltas)
    if len(s):
        print(
            f"식사→볼러스 시간차 {len(s)}쌍 (분, 음수=볼러스가 먼저): "
            f"p10={s.quantile(.1):.0f} 중앙값={s.median():.0f} p90={s.quantile(.9):.0f} | "
            f"±15분 이내 {100 * (s.abs() <= 15).mean():.1f}%"
        )
    return df


def _pair_meals_to_boluses(meals: pd.DataFrame, bolus: pd.DataFrame):
    """식사마다 가장 가까운 볼러스를 찾는다. ±2시간을 넘으면 짝이 없는 것으로 본다."""
    if meals.empty or bolus.empty:
        return 0, []
    bt = bolus["timestamp"].to_numpy()
    out = []
    for t in meals["timestamp"]:
        d = (bt - t.to_datetime64()) / np.timedelta64(1, "m")
        nearest = d[np.abs(d).argmin()]
        if abs(nearest) <= PAIR_WINDOW_MIN:
            out.append(float(nearest))
    return len(out), out


# ============================================================================
# 시연 유형
# ============================================================================


def stage_patterns(cgm, events, meta, windows, density) -> pd.DataFrame:
    win = windows[windows["passes"]]
    dense = set(zip(density.loc[density["replay_ok"], "patient_id"],
                    density.loc[density["replay_ok"], "window_start"]))

    rows = []
    for pid, sub in win.groupby("patient_id", observed=True):
        g = cgm[cgm["patient_id"] == pid]
        step = int(meta.loc[pid, "sensor_interval_min"])
        for w in sub.itertuples(index=False):
            lo = pd.Timestamp(w.window_start)
            hi = pd.Timestamp(w.window_end) + pd.Timedelta(days=1)
            rows.append(
                {
                    "patient_id": pid,
                    "modality": w.modality,
                    "window_start": w.window_start,
                    "replay_ok": (pid, w.window_start) in dense,
                    **_window_metrics(g[g["timestamp"].between(lo, hi)], step),
                }
            )

    df = pd.DataFrame(rows)
    df["NOCTURNAL_HYPO"] = df["nocturnal_hypo_days"] >= NOCTURNAL_MIN_DAYS
    df["HIGH_VARIABILITY"] = df["cv"] > CV_HIGH
    df["IN_TARGET"] = (
        (df["tir"] >= TIR_MIN) & (df["tbr"] <= TBR_MAX) & (df["cv"] <= CV_HIGH)
    )
    df.round(2).to_csv(OUT / "pattern_candidates.csv", index=False)

    for label, d in [("CGM 통과 창", df), ("이벤트까지 통과한 창", df[df["replay_ok"]])]:
        print(f"\n=== {label} (n={len(d)}) ===")
        for pat in ["NOCTURNAL_HYPO", "HIGH_VARIABILITY", "IN_TARGET"]:
            hit = d[d[pat]]
            print(f"  {pat:17s} 창 {len(hit):4d}  환자 {hit['patient_id'].nunique()}명")

    print("\n=== 유형별 대표 후보 (이벤트까지 통과한 창) ===")
    d = df[df["replay_ok"]]
    for pat, key in [("NOCTURNAL_HYPO", "nocturnal_hypo_days"),
                     ("HIGH_VARIABILITY", "cv"), ("IN_TARGET", "tir")]:
        pick = d[d[pat]].sort_values(key, ascending=False).drop_duplicates("patient_id")
        print(f"\n{pat}:")
        for r in pick.head(3).itertuples(index=False):
            print(f"  {r.patient_id} {r.window_start}  CV={r.cv:.1f} "
                  f"TIR={r.tir:.1f} TBR={r.tbr:.1f} 야간저혈당={r.nocturnal_hypo_days}일")
    return df


def _window_metrics(g: pd.DataFrame, step: int) -> dict:
    v = g["glucose_mgdl"]
    night = g[g["timestamp"].dt.hour < NIGHT_END_HOUR]

    nights = set()
    if len(night):
        low = (night["glucose_mgdl"] < TBR_BELOW).to_numpy()
        ts = night["timestamp"].reset_index(drop=True)
        run = pd.Series(low).ne(pd.Series(low).shift()).cumsum()
        for _, idx in pd.Series(range(len(low))).groupby(run):
            seg = idx[low[idx]]
            if len(seg) == 0:
                continue
            span = (ts.iloc[seg.iloc[-1]] - ts.iloc[seg.iloc[0]]).total_seconds() / 60 + step
            if span >= HYPO_MIN_MINUTES:
                nights.add(ts.iloc[seg.iloc[0]].date())

    return {
        "cv": 100 * v.std() / v.mean(),
        "tir": 100 * v.between(TBR_BELOW, TAR_ABOVE).mean(),
        "tbr": 100 * v.lt(TBR_BELOW).mean(),
        "nocturnal_hypo_days": len(nights),
    }


# ============================================================================


def main(stage: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cgm, events, meta = _load()
    print(f"환자 {len(meta)}명 · CGM {len(cgm):,}행 · 이벤트 {len(events):,}건\n")

    windows = stage_windows(cgm, events, meta)
    if stage == "windows":
        return
    print()
    density = stage_events(cgm, events, meta, windows)
    if stage == "events":
        return
    stage_patterns(cgm, events, meta, windows, density)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
