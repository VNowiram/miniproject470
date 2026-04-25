# measurement_services.py

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Optional, Dict, Any
import time
import math
import numpy as np

from measurement.meas_manager import ParallelModbus

# Our Package
from .modbusconv import ModbusMeter


class PerUnitOperator:
    """
    Convert measurement values and standard deviations between
    physical units and per-unit.

    Supported keys:
        V, I, P, Q
    Optional uncertainty dict:
        measurement["sd"] = {"V": ..., "P": ..., ...}
    """

    def __init__(self, V_base: float = 380.0, S_base: float = 1000.0):
        self.V_base = V_base
        self.S_base = S_base
        self.I_base = S_base / (np.sqrt(3) * V_base)   # 3-phase
        self.Z_base = (V_base ** 2) / S_base

    def _get_base(self, key: str) -> Optional[float]:
        if key == "V":
            return self.V_base
        if key == "I":
            return self.I_base
        if key in ("P", "Q"):
            return self.S_base
        return None

    def _to_pu_scalar(self, key: str, value: float) -> float:
        base = self._get_base(key)
        if base is None:
            return value
        return value / base

    def _from_pu_scalar(self, key: str, value_pu: float) -> float:
        base = self._get_base(key)
        if base is None:
            return value_pu
        return value_pu * base

    def to_pu(self, measurement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert one measurement dict from physical units to per-unit.

        Example:
        {
            "id": 1,
            "status": "ok",
            "V": 380.0,
            "P": 420.0,
            "Q": 100.0,
            "sd": {"V": 2.0, "P": 20.0, "Q": 15.0}
        }
        """
        result = deepcopy(measurement)

        for key in ("V", "I", "P", "Q"):
            if key in result and result[key] is not None:
                result[key] = self._to_pu_scalar(key, result[key])

        if "sd" in result and isinstance(result["sd"], dict):
            for key in ("V", "I", "P", "Q"):
                if key in result["sd"] and result["sd"][key] is not None:
                    result["sd"][key] = self._to_pu_scalar(key, result["sd"][key])

        return result

    def from_pu(self, measurement_pu: Dict[str, Any]) -> Dict[str, Any]:
        result = deepcopy(measurement_pu)

        for key in ("V", "I", "P", "Q"):
            if key in result and result[key] is not None:
                result[key] = self._from_pu_scalar(key, result[key])

        if "sd" in result and isinstance(result["sd"], dict):
            for key in ("V", "I", "P", "Q"):
                if key in result["sd"] and result["sd"][key] is not None:
                    result["sd"][key] = self._from_pu_scalar(key, result["sd"][key])

        return result

    def to_pu_batch(self, batch: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        return {
            meter_id: self.to_pu(measurement)
            for meter_id, measurement in batch.items()
        }

    def from_pu_batch(self, batch_pu: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        return {
            meter_id: self.from_pu(measurement_pu)
            for meter_id, measurement_pu in batch_pu.items()
        }

    def sd_to_pu(self, key: str, sd_value: float) -> float:
        return self._to_pu_scalar(key, sd_value)

    def var_to_pu(self, key: str, var_value: float) -> float:
        base = self._get_base(key)
        if base is None:
            return var_value
        return var_value / (base ** 2)


class MeasurementBuffer:
    """
    Store RAW measurement snapshots only.

    Each snapshot format:
    {
        "timestamp": 1712345678.12,
        "data": {
            1: {"V": ..., "P": ..., "Q": ..., "status": "ok", ...},
            2: {...},
            ...
        }
    }
    """

    def __init__(self, maxlen: int = 30):
        self._buffer = deque(maxlen=maxlen)

    def push(self, raw_batch: Dict[int, Dict[str, Any]], timestamp: Optional[float] = None):
        snapshot = {
            "timestamp": time.time() if timestamp is None else float(timestamp),
            "data": deepcopy(raw_batch),
        }
        self._buffer.append(snapshot)

    def latest_snapshot(self) -> Optional[Dict[str, Any]]:
        if not self._buffer:
            return None
        return deepcopy(self._buffer[-1])

    def latest_data(self) -> Optional[Dict[int, Dict[str, Any]]]:
        latest = self.latest_snapshot()
        if latest is None:
            return None
        return latest["data"]

    def all_snapshots(self) -> list[Dict[str, Any]]:
        return deepcopy(list(self._buffer))

    def clear(self):
        self._buffer.clear()

    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def __len__(self) -> int:
        return len(self._buffer)

    def get_series(self, meter_id: int, key: str, only_ok: bool = True) -> list[float]:
        series = []
        for snapshot in self._buffer:
            batch = snapshot["data"]
            m = batch.get(meter_id)
            if m is None:
                continue
            if only_ok and m.get("status") != "ok":
                continue
            value = m.get(key)
            if value is not None:
                series.append(value)
        return series

    def get_timestamped_series(self, meter_id: int, key: str, only_ok: bool = True) -> list[tuple[float, float]]:
        series = []
        for snapshot in self._buffer:
            ts = snapshot["timestamp"]
            batch = snapshot["data"]
            m = batch.get(meter_id)
            if m is None:
                continue
            if only_ok and m.get("status") != "ok":
                continue
            value = m.get(key)
            if value is not None:
                series.append((ts, value))
        return series

    def get_statistics(self, meter_id: int, key: str, only_ok: bool = True) -> Optional[Dict[str, float]]:
        series = self.get_series(meter_id, key, only_ok=only_ok)
        if not series:
            return None

        arr = np.asarray(series, dtype=float)
        mean = float(np.mean(arr))
        min_ = float(np.min(arr))
        max_ = float(np.max(arr))
        count = int(arr.size)
        std = float(np.std(arr, ddof=0))
        var = float(np.var(arr, ddof=0))

        return {
            "mean": mean,
            "min": min_,
            "max": max_,
            "count": count,
            "std": std,
            "var": var,
        }


class MeasurementService:
    """
    Read raw measurements from MeterGroup, store RAW snapshots in buffer,
    and provide RAW / PU views when requested.
    """

    def __init__(
        self,
        meter_group: ParallelModbus,
        converter: Optional[PerUnitOperator] = None,
        buffer: Optional[MeasurementBuffer] = None,
    ):
        self.meter_group = meter_group
        self.converter = converter
        self.buffer = buffer

    def connect(self):
        return self.meter_group.connect_all()

    def close(self):
        self.meter_group.close_all()

    # -------------------------
    # Read / Push RAW only
    # -------------------------
    def read_raw(self, timestamp: Optional[float] = None) -> Dict[int, Dict[str, Any]]:
        raw = self.meter_group.get_measurements()
        if self.buffer is not None:
            self.buffer.push(raw, timestamp=timestamp)
        return raw

    def push_raw(self, raw_batch: Dict[int, Dict[str, Any]], timestamp: Optional[float] = None):
        if self.buffer is not None:
            self.buffer.push(raw_batch, timestamp=timestamp)

    # -------------------------
    # Views from RAW
    # -------------------------
    def read_pu(self, timestamp: Optional[float] = None) -> Dict[int, Dict[str, Any]]:
        raw = self.read_raw(timestamp=timestamp)
        if self.converter is None:
            return raw
        return self.converter.to_pu_batch(raw)

    def latest_raw(self) -> Optional[Dict[int, Dict[str, Any]]]:
        if self.buffer is None:
            return None
        return self.buffer.latest_data()

    def latest_pu(self) -> Optional[Dict[int, Dict[str, Any]]]:
        raw = self.latest_raw()
        if raw is None or self.converter is None:
            return raw
        return self.converter.to_pu_batch(raw)

    def latest_snapshot(self) -> Optional[Dict[str, Any]]:
        if self.buffer is None:
            return None
        return self.buffer.latest_snapshot()

    # -------------------------
    # Series / Statistics
    # -------------------------
    def get_series_raw(self, meter_id: int, key: str, only_ok: bool = True) -> list[float]:
        if self.buffer is None:
            return []
        return self.buffer.get_series(meter_id, key, only_ok=only_ok)

    def get_series_pu(self, meter_id: int, key: str, only_ok: bool = True) -> list[float]:
        raw_series = self.get_series_raw(meter_id, key, only_ok=only_ok)
        if self.converter is None:
            return raw_series
        return [self.converter.sd_to_pu(key, x) for x in raw_series]

    def get_statistics_raw(self, meter_id: int, key: str, only_ok: bool = True) -> Optional[Dict[str, float]]:


        '''
        เก็บ
        '''

        if self.buffer is None:
            return None
        return self.buffer.get_statistics(meter_id, key, only_ok=only_ok)

    def get_statistics_pu(self, meter_id: int, key: str, only_ok: bool = True) -> Optional[Dict[str, float]]:
        raw_stats = self.get_statistics_raw(meter_id, key, only_ok=only_ok)
        if raw_stats is None:
            return None
        if self.converter is None:
            return raw_stats

        return {
            "mean": self.converter.sd_to_pu(key, raw_stats["mean"]),
            "min": self.converter.sd_to_pu(key, raw_stats["min"]),
            "max": self.converter.sd_to_pu(key, raw_stats["max"]),
            "count": raw_stats["count"],
            "std": self.converter.sd_to_pu(key, raw_stats["std"]),
            "var": self.converter.var_to_pu(key, raw_stats["var"]),
        }

    # -------------------------
    # Standard deviation helper
    # -------------------------
    def get_latest_sd_raw(self, meter_id: int, key: str) -> Optional[float]:
        raw = self.latest_raw()
        if raw is None:
            return None

        meter = raw.get(meter_id)
        if meter is None:
            return None

        sd_dict = meter.get("sd")
        if not isinstance(sd_dict, dict):
            return None

        return sd_dict.get(key)

    def get_latest_sd_pu(self, meter_id: int, key: str) -> Optional[float]:
        sd_raw = self.get_latest_sd_raw(meter_id, key)
        if sd_raw is None or self.converter is None:
            return sd_raw
        return self.converter.sd_to_pu(key, sd_raw)

    # -------------------------
    # Build SE helpers
    # -------------------------
    def build_latest_measurement_snapshot(
        self,
        as_pu: bool = True,
        only_ok: bool = True,
    ) -> Optional[Dict[int, Dict[str, Any]]]:
        raw = self.latest_raw()
        if raw is None:
            return None

        filtered = {}
        for meter_id, m in raw.items():
            if only_ok and m.get("status") != "ok":
                continue
            filtered[meter_id] = deepcopy(m)

        if as_pu and self.converter is not None:
            return self.converter.to_pu_batch(filtered)
        return filtered

    def clear(self):
        if self.buffer is not None:
            self.buffer.clear()


if __name__ == "__main__":
    print("=== SELF TEST START ===")

    meters = [
        ModbusMeter("192.168.1.20", slave_id=1),
        ModbusMeter("192.168.1.21", slave_id=2),
        ModbusMeter("192.168.1.22", slave_id=3),
    ]

    group = ParallelModbus(meters)
    converter = PerUnitOperator(V_base=380.0, S_base=1000.0)
    buffer = MeasurementBuffer(maxlen=30)

    ms = MeasurementService(
        meter_group=group,
        converter=converter,
        buffer=buffer,
    )

    ms.connect()

    try:
        while True:
            raw = ms.read_raw()          # push RAW only
            pu = ms.latest_pu()          # derive PU from latest RAW

            print("\nLatest snapshot:", ms.latest_snapshot())
            print("\nLatest raw:", raw)
            print("\nLatest pu:", pu)

            print("\nRAW statistics P meter 1:", ms.get_statistics_raw(1, "P"))
            print("PU  statistics P meter 1:", ms.get_statistics_pu(1, "P"))

            time.sleep(0.5)

            if len(buffer) > 30:
                break

    except KeyboardInterrupt:
        print("\n=== Interrupted by user ===")
        print("ALL snapshot:", buffer.all_snapshots())

    finally:
        ms.clear()
        ms.close()
        print("=== SELF TEST END ===")