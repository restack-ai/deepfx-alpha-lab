from __future__ import annotations

import os
from dataclasses import dataclass

import clickhouse_connect
import pandas as pd
from dotenv import load_dotenv


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str
    port: int
    username: str
    password: str
    database: str
    secure: bool = False


def config_from_env() -> ClickHouseConfig:
    """Build ClickHouse configuration from environment variables."""
    load_dotenv()

    host = os.getenv("CLICKHOUSE_HOST")
    if not host:
        raise RuntimeError("CLICKHOUSE_HOST is required")

    return ClickHouseConfig(
        host=host,
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        database=os.getenv("CLICKHOUSE_DATABASE", "deepfx"),
        secure=os.getenv("CLICKHOUSE_SECURE", "0").lower() in {"1", "true", "yes"},
    )


def get_client(config: ClickHouseConfig | None = None):
    """Create a clickhouse-connect client."""
    config = config or config_from_env()
    return clickhouse_connect.get_client(
        host=config.host,
        port=config.port,
        username=config.username,
        password=config.password,
        database=config.database,
        secure=config.secure,
    )


def load_ohlcv(
    symbol: str,
    timeframe: str,
    start: str | None = None,
    end: str | None = None,
    *,
    table: str = "ohlcv",
    client=None,
) -> pd.DataFrame:
    """Load OHLCV bars from ClickHouse as a time-indexed DataFrame."""
    client = client or get_client()
    conditions = ["symbol = %(symbol)s", "timeframe = %(timeframe)s"]
    params: dict[str, object] = {"symbol": symbol, "timeframe": timeframe}

    if start is not None:
        conditions.append("time >= %(start)s")
        params["start"] = start
    if end is not None:
        conditions.append("time < %(end)s")
        params["end"] = end

    query = f"""
        SELECT
            symbol,
            timeframe,
            time,
            open,
            high,
            low,
            close,
            volume,
            tick_volume
        FROM {table}
        WHERE {" AND ".join(conditions)}
        ORDER BY time
    """
    frame = client.query_df(query, parameters=params)
    if frame.empty:
        return frame

    frame["time"] = pd.to_datetime(frame["time"], utc=False)
    frame = frame.sort_values("time").drop_duplicates("time", keep="last")
    frame = frame.set_index("time")
    numeric_cols = ["open", "high", "low", "close", "volume", "tick_volume"]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return frame
