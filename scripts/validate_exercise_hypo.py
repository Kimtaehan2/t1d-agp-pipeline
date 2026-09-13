"""EXERCISE_DELAYED_HYPO 규칙을 OhioT1DM 자기보고 운동 로그로 검증한다.

규칙이 "켜지는가"만 보면 검증이 아니다. 저혈당이 잦은 환자는 운동과 무관하게 운동
뒤에도 저혈당이 있을 것이므로, **운동 뒤 시간대의 저혈당 발생률이 그 외 시간대보다
실제로 높은지**를 재야 한다. 세 가지를 잰다.

1. 상대위험(RR): 운동 종료 후 2~24시간 안에 있는 시간(post) 대 나머지 시간(other)의
   저혈당 에피소드 시작률 비. 환자별 + 전체 합산.
2. 참가자가 직접 보고한 ``hypo_event``가 post 시간대에 몰리는지. post 시간대가 전체
   관측 시간에서 차지하는 비율이 귀무가설 하의 기대치다.
3. 창 정의 민감도: 2~12시간, 2~24시간, 야간(00–06시)만.

산출: ``docs/ohio_t1dm/exercise_hypo_validation.csv``와 보고서 표.

    python scripts/validate_exercise_hypo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.loaders.ohio_t1dm import load
from src.patterns import (
    EXERCISE_DELAY_END_HOURS,
    EXERCISE_DELAY_START_HOURS,
    NIGHT_END_HOUR,
    NIGHT_START_HOUR,
    _hypo_episodes,
    detect_patterns,
)
from src.preprocess import GRID_MINUTES, preprocess_cgm
from src.metrics import sliding_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ohio_t1dm"

WINDOWS = {
    "2-24h": (2, 24),
    "2-12h": (2, 12),
    "6-15h": (6, 15),      # 문헌에서 지연성 저혈당이 가장 흔하다고 보는 구간
}


def _post_mask(grid: pd.Series, sessions: pd.DataFrame, lo_h: int, hi_h: int) -> np.ndarray:
    """격자 슬롯마다 '어떤 운동의 종료 후 [lo, hi)시간 안인가'."""
    t = grid.to_numpy()
    mask = np.zeros(len(t), dtype=bool)
    for s in sessions.itertuples(index=False):
        end = np.datetime64(s.timestamp) + np.timedelta64(int(s.value or 0), "m")
        mask |= (t >= end + np.timedelta64(lo_h, "h")) & (t < end + np.timedelta64(hi_h, "h"))
    return mask


def _rate(n_events: int, n_slots: int) -> float:
    """1,000시간당 에피소드 시작 수."""
    hours = n_slots * GRID_MINUTES / 60.0
    return 1000.0 * n_events / hours if hours else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    proc = preprocess_cgm(data.cgm)
    ex_all = data.events.query("event_type == 'exercise'")
    hypo_all = data.events.query("event_type == 'hypo_event'")

    rows = []
    for pid, sub in proc.groupby("patient_id", observed=True):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        ts = sub["timestamp"]
        g = sub["glucose_mgdl"].to_numpy(dtype="float64")
        present = ~np.isnan(g)
        sessions = ex_all[ex_all["patient_id"] == pid]
        reported = hypo_all[hypo_all["patient_id"] == pid]["timestamp"].to_numpy()

        episodes = _hypo_episodes(ts, g, GRID_MINUTES)
        starts = np.array([np.datetime64(e[0]) for e in episodes], dtype="datetime64[ns]")
        night = (ts.dt.hour.to_numpy() >= NIGHT_START_HOUR) & (ts.dt.hour.to_numpy() < NIGHT_END_HOUR)

        base = {
            "patient_id": pid,
            "days": round((ts.max() - ts.min()).total_seconds() / 86400, 1),
            "n_exercise": len(sessions),
            "n_hypo_episodes": len(episodes),
            "n_reported_hypo": len(reported),
        }
        if len(sessions) == 0:
            rows.append({**base, "window": "-"})
            continue

        for name, (lo, hi) in WINDOWS.items():
            post = _post_mask(ts, sessions, lo, hi)
            for scope, scope_mask in (("all", np.ones(len(ts), bool)), ("night", night)):
                m_post = post & scope_mask & present
                m_other = ~post & scope_mask & present
                idx_post = set(np.flatnonzero(m_post))
                idx_other = set(np.flatnonzero(m_other))
                start_idx = np.searchsorted(ts.to_numpy(), starts)
                n_post = int(sum(i in idx_post for i in start_idx))
                n_other = int(sum(i in idx_other for i in start_idx))
                r_post, r_other = _rate(n_post, m_post.sum()), _rate(n_other, m_other.sum())

                # 자기보고 저혈당이 post 시간대에 떨어지는 비율 vs post 시간대의 시간 점유율
                rep_idx = np.searchsorted(ts.to_numpy(), reported)
                rep_idx = rep_idx[rep_idx < len(ts)]
                rep_in_post = int(sum(post[i] and scope_mask[i] for i in rep_idx))
                rep_in_scope = int(sum(scope_mask[i] for i in rep_idx))
                share_time = m_post.sum() / max(1, (m_post.sum() + m_other.sum()))

                # 운동 단위: 뒤따르는 저혈당이 있는 운동의 비율
                followed = 0
                for s in sessions.itertuples(index=False):
                    end = np.datetime64(s.timestamp) + np.timedelta64(int(s.value or 0), "m")
                    w = (starts >= end + np.timedelta64(lo, "h")) & (starts < end + np.timedelta64(hi, "h"))
                    if scope == "night":
                        w &= np.array([NIGHT_START_HOUR <= pd.Timestamp(x).hour < NIGHT_END_HOUR for x in starts]) if starts.size else w
                    followed += bool(w.any())

                rows.append({
                    **base, "window": name, "scope": scope,
                    "hours_post": round(m_post.sum() * GRID_MINUTES / 60, 1),
                    "hours_other": round(m_other.sum() * GRID_MINUTES / 60, 1),
                    "episodes_post": n_post, "episodes_other": n_other,
                    "rate_post_per_1000h": round(r_post, 2),
                    "rate_other_per_1000h": round(r_other, 2),
                    "relative_risk": round(r_post / r_other, 2) if r_other else float("nan"),
                    "sessions_followed": followed,
                    "sessions_followed_pct": round(100 * followed / len(sessions), 1),
                    "reported_hypo_in_post": rep_in_post,
                    "reported_hypo_in_scope": rep_in_scope,
                    "reported_share_pct": round(100 * rep_in_post / rep_in_scope, 1) if rep_in_scope else float("nan"),
                    "expected_share_pct": round(100 * share_time, 1),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exercise_hypo_validation.csv", index=False)

    # --- 요약 출력 -----------------------------------------------------------
    main_w = df[(df["window"] == "2-24h") & (df["scope"] == "all")]
    print("=== 운동 종료 2~24시간 (전체 시간대) ===")
    cols = ["patient_id", "n_exercise", "n_hypo_episodes", "episodes_post", "episodes_other",
            "rate_post_per_1000h", "rate_other_per_1000h", "relative_risk",
            "sessions_followed_pct", "reported_share_pct", "expected_share_pct"]
    print(main_w[cols].to_string(index=False))

    print("\n=== 합산 상대위험 (창 × 범위) ===")
    for (w, sc), d in df[df["window"] != "-"].groupby(["window", "scope"]):
        hp, ho = d["hours_post"].sum(), d["hours_other"].sum()
        ep, eo = int(d["episodes_post"].sum()), int(d["episodes_other"].sum())
        rp, ro = 1000 * ep / hp, 1000 * eo / ho
        rep_p, rep_s = int(d["reported_hypo_in_post"].sum()), int(d["reported_hypo_in_scope"].sum())
        print(f"  {w:6s} {sc:5s}  post {ep:3d}건/{hp:7.0f}h={rp:5.2f}  other {eo:3d}건/{ho:7.0f}h={ro:5.2f}"
              f"  RR={rp / ro:4.2f}  | 자기보고 저혈당 post 비율 {100 * rep_p / rep_s:4.1f}% (시간 점유율 {100 * hp / (hp + ho):4.1f}%)")

    # --- 규칙 자체가 슬라이딩 14일 창에서 얼마나 켜지는가 ---------------------
    print("\n=== 규칙 발화 (슬라이딩 14일, stride 7) ===")
    metrics = sliding_metrics(proc, stride_days=7)
    found = detect_patterns(proc, events=data.events, metrics=metrics)
    fired = found[found["pattern"] == "exercise_delayed_hypoglycemia"]
    n_win = len(metrics)
    print(f"  창 {n_win}개 중 {len(fired)}개 발화, 환자 {fired['patient_id'].nunique()}명")
    for r in fired.itertuples(index=False):
        print(f"    {r.patient_id} {r.window_start.date()}  {r.detail}")


if __name__ == "__main__":
    main()
