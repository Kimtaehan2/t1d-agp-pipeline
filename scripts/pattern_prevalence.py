"""문헌 조사가 권고한 패턴 enum이 실제 데이터에서 얼마나 나타나는지 센다.

세 데이터셋(T1D-UOM 수동관리 12명, OhioT1DM 12명, Replace-BG 226명)에 대해

- **최근 14일 창** (환자당 1개, 제품이 실제로 쓰는 창): 패턴별로 켜진 환자 비율
- **슬라이딩 14일 창, stride 7일**: 패턴별로 켜진 창 비율 (표본 수 확보용)

을 낸다. 규칙이 한 번도 안 켜지거나 전원 켜지면 그 자체가 결과다 — 임계값이
이 인구에서 변별력이 없다는 뜻이거나, 이벤트 로그가 없어 판정 자체가 안 된다는
뜻이다. 임계값은 조정하지 않는다.

    python scripts/pattern_prevalence.py            # 원본이 있는 데이터셋 전부
    python scripts/pattern_prevalence.py replace_bg # 하나만

Replace-BG는 원본이 837MB라 한 번에 DataFrame으로 올리지 않는다. 원저자의 경로대로
``data/interim/`` Parquet → ``preprocess_parquet`` → ``data/processed/`` Parquet을
거치고, 이미 있으면 다시 만들지 않는다. 원본이 없으면 그 데이터셋만 건너뛴다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.loaders import ohio_t1dm, replace_bg, t1d_uom
from src.metrics import recent_metrics, sliding_metrics
from src.patterns import PATTERN_DEFINITIONS, PATTERN_ORDER, detect_patterns
from src.preprocess import preprocess_cgm, preprocess_parquet
from src.storage import read_parquet, write_parquet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "patterns"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

#: 조사 보고서 1부 매트릭스의 "출처 수"와 티어. 규칙 키 → (보고서 enum, 출처 수, 티어).
LITERATURE: dict[str, tuple[str, str, str]] = {
    "low_data_coverage": ("LOW_DATA_COVERAGE", "6", "MVP"),
    "excess_hypoglycemia": ("HYPO_OVERALL_L1", "7", "MVP"),
    "excess_hypoglycemia_level2": ("HYPO_OVERALL_L2", "7", "MVP"),
    "nocturnal_hypoglycemia": ("NOCTURNAL_HYPO", "6", "MVP"),
    "excess_hyperglycemia": ("HYPER_SUSTAINED", "7", "MVP"),
    "excess_hyperglycemia_level2": ("HYPER_SUSTAINED", "7", "MVP"),
    "high_variability": ("HIGH_VARIABILITY", "6", "MVP"),
    "postprandial_spike": ("POSTPRANDIAL_HYPER", "6", "MVP (식사 로그)"),
    "fasting_hyperglycemia": ("FASTING_MORNING_HYPER", "6", "MVP"),
    "low_time_in_range": ("(TIR 목표)", "전 출처", "지표"),
    "dawn_rise": ("DAWN_PHENOMENON", "3", "Phase 2"),
    "rapid_drop": ("RAPID_DROP", "3", "Phase 2"),
    "exercise_delayed_hypoglycemia": ("EXERCISE_DELAYED_HYPO", "4", "Phase 2 (운동 로그)"),
    "sickday_ketone_risk": ("SICKDAY_KETONE_RISK", "4", "Phase 2"),
    "prolonged_hypoglycemia": ("PROLONGED_HYPO", "2", "Phase 2"),
    "hypo_concentration": ("(시간대 검토)", "AGP-lit", "기존"),
    "hyper_concentration": ("(시간대 검토)", "AGP-lit", "기존"),
}


def _uom():
    uom = t1d_uom.load(modalities=t1d_uom.MANUAL_MODALITIES)
    mdi = uom.cgm["patient_id"].isin([t1d_uom.patient_id(p) for p in t1d_uom.MDI_PATIENTS])
    return {
        "T1D-UOM (수동관리 12)": (preprocess_cgm(uom.cgm), uom.events),
        "T1D-UOM (MDI 8)": (preprocess_cgm(uom.cgm[mdi]), uom.events),
    }


def _ohio():
    ohio = ohio_t1dm.load()
    return {"OhioT1DM (12)": (preprocess_cgm(ohio.cgm), ohio.events)}


def _replace_bg():
    """원본 → interim → processed. 이미 있는 단계는 건너뛴다."""
    cgm_interim = INTERIM / "replace_bg_cgm.parquet"
    events_interim = INTERIM / "replace_bg_events.parquet"
    cgm_processed = PROCESSED / "replace_bg_cgm.parquet"

    if not cgm_processed.exists():
        if not cgm_interim.exists():
            INTERIM.mkdir(parents=True, exist_ok=True)
            print(f"  Replace-BG 원본 → {cgm_interim} ...")
            stats = replace_bg.export_cgm_parquet(cgm_interim)
            print("  " + stats.summary())
        PROCESSED.mkdir(parents=True, exist_ok=True)
        print(f"  전처리 → {cgm_processed} ...")
        preprocess_parquet(cgm_interim, cgm_processed)
    if not events_interim.exists():
        INTERIM.mkdir(parents=True, exist_ok=True)
        write_parquet(replace_bg.load_events(), events_interim)

    return {"Replace-BG (226)": (read_parquet(cgm_processed), read_parquet(events_interim))}


#: 이름 → (로더 모듈, 원본 존재 여부 검사, 로더). 원본이 없으면 그 데이터셋만 건너뛴다.
SOURCES = {
    "t1d_uom": (t1d_uom, lambda: t1d_uom.DEFAULT_RAW_DIR.is_dir(), _uom),
    "ohio_t1dm": (ohio_t1dm, lambda: ohio_t1dm.DEFAULT_RAW_DIR.is_dir(), _ohio),
    "replace_bg": (
        replace_bg,
        lambda: replace_bg.DEFAULT_RAW_DIR.is_dir()
        or (PROCESSED / "replace_bg_cgm.parquet").exists(),
        _replace_bg,
    ),
}


def _datasets(only: str | None) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    if only and only not in SOURCES:
        raise SystemExit(f"모르는 데이터셋 {only!r}. 가능한 값: {sorted(SOURCES)}")
    out = {}
    for key, (module, available, loader) in SOURCES.items():
        if only and key != only:
            continue
        if not available():
            print(f"[건너뜀] {key}: 원본이 없다. 기대 경로: {module.DEFAULT_RAW_DIR}")
            continue
        print(f"[로드] {key}")
        out.update(loader())
    return out


def main(only: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    all_findings = []

    datasets = _datasets(only)
    if not datasets:
        print("돌릴 데이터셋이 없다.")
        return

    for name, (proc, events) in datasets.items():
        n_patients = proc["patient_id"].nunique()

        recent = recent_metrics(proc, drop_invalid=False)
        f_recent = detect_patterns(proc, events=events, metrics=recent)
        f_recent["dataset"] = name
        f_recent["mode"] = "recent"
        all_findings.append(f_recent)

        sliding = sliding_metrics(proc, stride_days=7, drop_invalid=False)
        f_slide = detect_patterns(proc, events=events, metrics=sliding)
        f_slide["dataset"] = name
        f_slide["mode"] = "sliding"
        all_findings.append(f_slide)

        n_valid_recent = int(recent["is_valid"].sum())
        n_windows = len(sliding)
        n_valid_windows = int(sliding["is_valid"].sum())

        for key in PATTERN_ORDER:
            pat_r = f_recent[f_recent["pattern"] == key]
            # 시간대 집중 패턴은 창 하나에 시간대별로 여러 행이 나온다. 창 단위로 센다.
            pat_s = (f_slide[f_slide["pattern"] == key]
                     .drop_duplicates(subset=["patient_id", "window_start"]))
            # low_data_coverage의 분모는 전체 창, 나머지는 유효 창이다.
            denom_p = n_patients if key == "low_data_coverage" else n_valid_recent
            denom_w = n_windows if key == "low_data_coverage" else n_valid_windows
            enum, n_src, tier = LITERATURE.get(key, ("", "", ""))
            rows.append({
                "dataset": name,
                "pattern": key,
                "name": PATTERN_DEFINITIONS[key].name,
                "report_enum": enum,
                "n_sources": n_src,
                "tier": tier,
                "patients_recent": int(pat_r["patient_id"].nunique()),
                "patients_denom": denom_p,
                "pct_patients_recent": round(100 * pat_r["patient_id"].nunique() / denom_p, 1) if denom_p else float("nan"),
                "windows_sliding": len(pat_s),
                "windows_denom": denom_w,
                "pct_windows_sliding": round(100 * len(pat_s) / denom_w, 1) if denom_w else float("nan"),
            })

        print(f"\n=== {name}: 환자 {n_patients}, 최근 창 유효 {n_valid_recent}, "
              f"슬라이딩 창 {n_windows} (유효 {n_valid_windows}) ===")

    df = pd.DataFrame(rows)
    findings = pd.concat(all_findings, ignore_index=True)
    if only:
        # 한 데이터셋만 돌렸으면 기존 결과에서 그 데이터셋 행만 바꿔 끼운다.
        for fname, new in (("pattern_prevalence.csv", df), ("pattern_findings.csv", findings)):
            path = OUT / fname
            if path.exists():
                old = pd.read_csv(path)
                old = old[~old["dataset"].isin(new["dataset"].unique())]
                new = pd.concat([old, new], ignore_index=True)
            new.to_csv(path, index=False)
        df = pd.read_csv(OUT / "pattern_prevalence.csv")
    else:
        df.to_csv(OUT / "pattern_prevalence.csv", index=False)
        findings.to_csv(OUT / "pattern_findings.csv", index=False)

    wide = df.pivot_table(
        index=["pattern", "name", "report_enum", "n_sources", "tier"],
        columns="dataset",
        values=["pct_patients_recent", "pct_windows_sliding"],
        aggfunc="first",
    )
    wide = wide.reindex([k for k in PATTERN_ORDER if k in wide.index.get_level_values(0)], level=0)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print()
        print(wide.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
