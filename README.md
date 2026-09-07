# t1d-agp-pipeline

CGM 데이터 전처리 · AGP 지표 · 규칙 기반 패턴 탐지 파이프라인.

주력 데이터셋은 **T1D-UOM**(University of Manchester, CC BY 4.0)이고, `main`의
로더는 그중 **MDI 환자 8명만** 싣는다.

## 왜 MDI만인가

12/2 시연은 리플레이 방식이다. 공개 데이터셋 환자의 14일 창을 가상환자로 적재하고,
식사·인슐린 이벤트를 "보호자가 앱에 입력한 것처럼" 흘려 넣는다. 그 서사가 성립하려면
**사람이 용량을 결정한 기록**이어야 한다.

| 구분 | 환자 | main 기본값 |
|---|---|---|
| MDI (지속형 주사 1회 + 매 끼 볼러스) | 2302, 2305, 2306, 2313, 2314, 2401, 2403, 2405 (8명) | **○** |
| 개방루프 펌프 (기저는 기기가 쥔다) | 2304, 2308, 2309, 2310 (4명) | × |
| 폐루프 AID (볼러스까지 알고리즘) | 2301, 2307 (2명) | × |
| 분류 불가 (기저 파일 없음) | 2303, 2320, 2404 (3명) | × |

**치료 방식 컬럼은 원본에 없다.** 원본 README가 광고하는 `Demographics/` 폴더가
배포본에 빠져 있어서, 기저 기록의 *모양*으로 추론했다. 폐루프 2명은 기저가 5분마다
2,000개 가까운 서로 다른 값으로 바뀐다 — 사람이 손으로 넣는 값이 아니다.
근거는 [스키마 노트](docs/schema_notes.md#t1d-uom-주력-main은-mdi-8명만-싣는다),
판정 과정은 `T1D-UOM` 브랜치의 `docs/t1d_uom/판정보고서.md`에 있다.

기본값에서 빠질 뿐 못 읽는 것은 아니다.

```python
load(modalities={"PUMP_OPEN"})   # 개방루프 펌프 4명
```

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

data = load()                      # MDI 8명
proc = preprocess_cgm(data.cgm)
metrics = recent_metrics(proc)
findings = detect_patterns(proc, events=data.events, metrics=metrics)
```

`data.patients`에 환자별 치료 방식·센서 주기·관측 구간이 들어 있다. 치료 방식이
추론값이라 무엇을 근거로 걸렀는지 남겨 두기 위한 것이다.

## 구조

| 경로 | 역할 |
|---|---|
| `src/schema.py` | 공통 스키마 계약. 변경 시 팀 합의가 필요하다 |
| `src/preprocess.py` | 단위 변환 · 5분 격자 리샘플 · 결측 보간 · 캡핑 플래그 |
| `src/metrics.py` | AGP 지표 (TIR/TAR/TBR/CV/GMI), 최근 14일 창 |
| `src/patterns.py` | 규칙 기반 패턴 탐지. ML을 쓰지 않는다 |
| `src/storage.py` | Parquet 입출력 (pyarrow → fastparquet 폴백) |
| `src/loaders/t1d_uom.py` | T1D-UOM 로더 |
| `docs/schema_notes.md` | 원본 구조 실측 기록 |

## 로더가 원본 README를 믿지 않는 지점

셋 다 실측으로 뒤집었다. 자세한 근거는 스키마 노트에 있다.

1. **날짜는 `DD/MM/YYYY`다.** README는 `MM/DD/YYYY`라고 적어 놨다. 두 필드가 모두
   12 이하인 날짜는 뒤집어 읽어도 예외가 나지 않고 **조용히** 어긋나므로, 로더가
   연구 기간(`STUDY_START`~`STUDY_END`) 범위로 파싱 결과를 검증한다.
2. **지속형 기저는 주입률이 아니라 주사다.** `insulin_kind=L`은 하루 1~2회 주사라
   `insulin_basal_rate`(IU/h)가 아니라 `insulin_sc`(IU)로 보낸다.
3. **센서가 한 종류가 아니다.** 측정 주기가 5분인 환자와 15분인 환자가 섞여 있고
   (MDI 8명 중 2313만 5분), 상한도 27.8 mmol/L과 22.2 mmol/L로 갈린다.
   `SENSOR_LIMITS["t1d_uom"]`은 넓은 쪽 `(39.6, 500.9)`이라 MDI 8명에는 맞지만,
   펌프 환자 2309를 불러오면 그 환자의 캡핑은 잡히지 않는다.

**운동 이벤트는 만들지 않는다.** `UoMActivity*.csv`는 15분 간격 웨어러블 에폭
연속 스트림이고 대부분의 행이 `SEDENTARY`다. 자기보고 운동 로그가 아니므로 여기서
이벤트를 만들면 지어낸 값이 된다. 그 결과 `EXERCISE_DELAYED_HYPO`는 이 데이터셋으로
정답을 놓고 검증할 수 없다.

## 다른 브랜치

| 브랜치 | 내용 |
|---|---|
| `T1D-UOM` | 리플레이 적합성 판정 보고서와 분석 스크립트, 수동관리 12명(MDI + 개방루프)을 다루는 로더 |
| `Replace-BG` | Replace-BG · ShanghaiT1DM 로더와 원본 구조 기록 |

`src/patterns.py`의 임계값과 `src/preprocess.py`의 `SENSOR_LIMITS`에는 Replace-BG
실측 수치가 근거로 남아 있다. 로더는 옮겼지만 값이 코드에 남아 있는 한 근거도
같이 둔다.
