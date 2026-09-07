# t1d-agp-pipeline — T1D-UOM 브랜치

CGM 데이터 전처리 · AGP 지표 · 규칙 기반 패턴 탐지 파이프라인에
**T1D-UOM(University of Manchester) 데이터셋**을 붙인 브랜치다.

12/2 시연을 리플레이 방식으로 하려면 공개 데이터셋 환자의 14일 창을 가상환자로
적재해야 하는데, 그동안 이벤트 검증을 걸어 두었던 OhioT1DM은 승인이 나지 않았다.
**T1D-UOM이 Ohio를 대체할 수 있는지 판정한 결과와 그 판정을 만든 코드**가 여기 있다.

**판정: GO with caveats.** 근거는 [판정 보고서](docs/t1d_uom/판정보고서.md)에 있다.

## 데이터셋

| 항목 | 값 |
|---|---|
| 출처 | T1D-UOM – A Longitudinal Multimodal Dataset of Type 1 Diabetes (University of Manchester) |
| 라이선스 | CC BY 4.0 |
| 기간 | 2023-09 ~ 2024-09 |
| 환자 | 17명 (수동관리 12명 · 폐루프 2명 · 분류 불가 3명) |
| 구성 | 혈당 · 기저 · 볼러스 · 식사 · 활동 · 수면, 환자별 CSV |
| 혈당 단위 | mmol/L (×18.018로 변환) |
| 센서 주기 | **5분과 15분이 섞여 있다** |

인용 문구와 라이선스 표기는 [스키마 노트 §5](docs/schema_notes.md)에 있다.

## 이 브랜치에서 조심할 것 세 가지

원본 README를 그대로 믿으면 틀리는 지점이다. 셋 다 실측으로 뒤집었다.

### 1. 날짜는 `DD/MM/YYYY`다 — README는 `MM/DD/YYYY`라고 적어 놨다

`22/10/2023` 같은 값이 있고, 두 번째 필드가 12를 넘는 행은 전 파일에 0건이다.
문제는 **두 필드가 모두 12 이하인 날짜는 뒤집어 읽어도 예외가 나지 않는다**는
점이다(`05/10` → 5월 10일 vs 10월 5일). 조용히 어긋나므로 로더가
`STUDY_START`~`STUDY_END` 범위로 파싱 결과를 검증한다.

### 2. 치료 방식 컬럼이 없다 — 기저 기록의 *모양*으로 추론했다

README가 광고하는 `Demographics/` 폴더가 배포본에 없다. `src/loaders/t1d_uom.py`의
`MODALITY`가 그 추론 결과이고, 근거는 다음과 같다.

| 구분 | 기저 기록 모양 | 환자 |
|---|---|---|
| MDI | `insulin_kind=L`, 하루 1~2건, 서로 다른 용량 ≤9개 | 2302, 2305, 2306, 2313, 2314, 2401, 2403, 2405 (8명) |
| 개방루프 펌프 | `insulin_kind=R`, 하루 4~8건, 서로 다른 용량 3~9개 | 2304, 2308, 2309, 2310 (4명) |
| **폐루프(AID)** | `insulin_kind=R`, 하루 120~157건, 서로 다른 용량 1,034~1,974개 | 2301, 2307 (2명) — **제외** |
| 분류 불가 | 기저 파일 없음 | 2303, 2320, 2404 (3명) — **제외** |

기저가 5분마다 2,000개 가까운 서로 다른 값으로 바뀌는 것은 사람이 손으로 넣는
값이 아니다. 수동관리 분석에 섞이면 안 되므로 `load()`는 기본값에서 뺀다.

### 3. CGM 확보율이 이벤트 확보율을 보장하지 않는다

2304와 2310은 CGM 창 227개를 통과하지만 **리플레이 가능한 창이 0개**다. 인슐린·식사
기록이 CGM 기간을 덮지 않기 때문이다(2310은 식사 파일 자체가 없다). 창을 고를 때
두 조건을 모두 걸어야 한다.

## 쓰는 법

```bash
pip install -r requirements.txt
pytest
```

원본은 저장소에 커밋하지 않는다. `src/loaders/t1d_uom.py`의 `DEFAULT_RAW_DIR`
(`data/raw/ManchesterCSCoordinatedDiabetesStudy-V1.0.1/sharpic-…`)에 풀어 두면
실데이터 통합 테스트까지 돈다. 없으면 그 테스트만 건너뛴다.

```python
from src.loaders.t1d_uom import load
from src.preprocess import preprocess_cgm
from src.metrics import recent_metrics
from src.patterns import detect_patterns

data = load()                      # 기본값 = 수동관리 12명
proc = preprocess_cgm(data.cgm)
metrics = recent_metrics(proc)
findings = detect_patterns(proc, events=data.events, metrics=metrics)
```

`data.patients`에 환자별 치료 방식·센서 주기·관측 구간이 들어 있다. 치료 방식이
추론값이라 무엇을 근거로 걸렀는지 남겨 두기 위한 것이다.

폐루프 환자를 일부러 보고 싶으면 명시적으로 요청해야 한다.

```python
load(modalities={"CLOSED_LOOP"})
```

## 분석 재현

```bash
python scripts/explore_t1d_uom.py all
```

판정 보고서의 숫자를 만든 스크립트다. `docs/t1d_uom/`에 CSV 세 개를 쓴다.

| 파일 | 내용 |
|---|---|
| `window_candidates.csv` | 슬라이딩 14일 창별 확보율·유효일수·통과 여부 |
| `event_density.csv` | 통과 창별 식사·볼러스 밀도와 식사–볼러스 쌍 |
| `pattern_candidates.csv` | 통과 창별 CV·TIR·TBR·야간 저혈당 일수와 시연 유형 판정 |

## 공용 파이프라인

`src/schema.py` · `src/preprocess.py` · `src/metrics.py` · `src/patterns.py` ·
`src/storage.py`는 데이터셋과 무관한 공용 계약이다. Replace-BG와 ShanghaiT1DM
로더는 `Replace-BG` 브랜치에 있다.

`src/preprocess.py`의 `SENSOR_LIMITS`에 `t1d_uom: (39.6, 500.9)`을 추가했다.
**이 데이터셋은 센서가 한 종류가 아니라서** 상한이 22.2 mmol/L(=400.0)인 환자도
있다(2309는 22.2가 207건). 등록된 값은 넓은 쪽이므로 22.2에서 검열된 환자의
캡핑은 잡히지 않는다.
