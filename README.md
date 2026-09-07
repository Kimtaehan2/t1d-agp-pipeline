# t1d-agp-pipeline — Replace-BG · ShanghaiT1DM 브랜치

CGM 데이터 전처리 · AGP 지표 · 규칙 기반 패턴 탐지 파이프라인 중
**Replace-BG와 ShanghaiT1DM 데이터셋 분석**을 보존한 브랜치다.

주력 데이터셋이 T1D-UOM으로 바뀌면서 `main`에서 이 두 로더를 걷어냈다.
지워버리면 실측 근거까지 같이 사라지므로 여기에 그대로 남긴다.

## 이 브랜치에만 있는 것

| 경로 | 내용 |
|---|---|
| `src/loaders/replace_bg.py` | Replace-BG 로더. 226명 · 26주 · 1,480만 행. 837MB 단일 파일을 청크로 읽고 Parquet로 흘려 쓴다 |
| `src/loaders/shanghai_t1dm.py` | ShanghaiT1DM 로더. 환자별 Excel 1개 = 내원 1회 |
| `tests/test_replace_bg.py` | Replace-BG 로더 테스트 |
| `tests/test_shanghai_t1dm.py` | ShanghaiT1DM 로더 테스트 |
| `docs/schema_notes.md` | 두 데이터셋 원본 구조 실측 기록 |

## 공용 파이프라인

`src/schema.py` · `src/preprocess.py` · `src/metrics.py` · `src/patterns.py` ·
`src/storage.py`는 `main`과 공유한다. 이 브랜치의 사본은 두 로더가 있던 시점의
스냅샷이므로, 파이프라인 자체를 고칠 일이 있으면 `main`에서 고쳐라.

## 남겨 두는 실측 근거

`src/patterns.py`의 임계값 주석과 `src/preprocess.py`의 `SENSOR_LIMITS`에는
Replace-BG 실측 수치가 근거로 박혀 있다. `main`에서도 이 값들은 살아 있으며
출처가 Replace-BG임을 주석에 명시해 두었다.

특히 기억할 것:

- **Dexcom G4 센티널은 스펙의 40/400이 아니라 39/401이다.** 39가 28,885건,
  401이 48,935건으로 주변 값의 6~40배다.
- **Replace-BG는 날짜가 익명화돼 있다.** 절대 날짜와 요일은 의미가 없고, 환자
  내부의 시간 간격과 하루 중 시각만 의미가 있다.
- **식후 스파이크 임계 30%는 Replace-BG 219명 분포에서 나왔다.** 50%로 두면
  1명만 걸려 규칙이 죽고, 20%로 두면 중앙값 환자가 걸린다.

## 실행

```bash
pip install -r requirements.txt
pytest
```

원본 데이터는 저장소에 커밋하지 않는다. 각 로더의 `DEFAULT_RAW_DIR`이 기대하는
경로에 원본을 두고 쓴다.
