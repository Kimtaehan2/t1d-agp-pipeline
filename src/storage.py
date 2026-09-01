"""Parquet 입출력 헬퍼.

대용량 데이터는 CSV가 아니라 Parquet으로 저장한다(팀 규칙). 이 PC에서는
pyarrow의 DLL이 애플리케이션 제어 정책에 막혀 import되지 않으므로 fastparquet로
넘어간다. pyarrow가 다시 쓸 수 있게 되면 자동으로 그쪽을 쓴다.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd


class ParquetEngineError(RuntimeError):
    """사용 가능한 Parquet 엔진이 없을 때 발생."""


@functools.lru_cache(maxsize=1)
def parquet_engine() -> str:
    """쓸 수 있는 Parquet 엔진 이름을 돌려준다. pyarrow 우선."""
    problems = []
    for name in ("pyarrow", "fastparquet"):
        try:
            __import__(name)
        except Exception as exc:  # ImportError 외에 DLL 차단도 잡는다
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
        else:
            return name
    raise ParquetEngineError("Parquet 엔진을 못 찾았다.\n  " + "\n  ".join(problems))


def write_parquet(df: pd.DataFrame, path: str | Path, append: bool = False) -> Path:
    """DataFrame을 Parquet으로 쓴다. ``append=True``면 기존 파일에 이어 붙인다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = parquet_engine()

    kwargs = {"engine": engine, "index": False}
    if append:
        if not path.exists():
            append = False
        elif engine == "fastparquet":
            kwargs["append"] = True
        else:
            raise ParquetEngineError(
                "pyarrow 엔진에서는 append를 지원하지 않는다. 파일을 나눠 써라."
            )

    df.to_parquet(path, **kwargs)
    return path


def read_parquet(path: str | Path, **kwargs) -> pd.DataFrame:
    """Parquet을 읽는다. dtype 복원은 호출 측에서 ``coerce_*_frame``으로 한다."""
    return pd.read_parquet(path, engine=parquet_engine(), **kwargs)
