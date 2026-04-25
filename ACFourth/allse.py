// measurement/device_profile.json
{
  "device_profiles": {
    "siemens_pac4200": {
      "byte_order": "big",
      "word_order": "big",
      "registers": {
        "V": {"reg": 7, "count": 2, "type": "float32"},
        "I": {"reg": 13, "count": 2, "type": "float32"},
        "P": {"reg": 65, "count": 2, "type": "float32"},
        "Q": {"reg": 67, "count": 2, "type": "float32"}
      }
    },
    "lemoco_lmc168": {
      "byte_order": "big",
      "word_order": "big",
      "registers": {
        "V": {"reg": 1020, "count": 2, "type": "float32"},
        "I": {"reg": 1000, "count": 2, "type": "float32"},
        "P": {"reg": 1034, "prefix": "k", "count": 2, "type": "float32"},
        "Q": {"reg": 1042, "prefix": "k", "count": 2, "type": "float32"}
      }
    }
  },
  "ip_ranges": {
  "siemens": {
    "start": "192.168.1.1",
    "end": "192.168.1.200",
    "profile": "siemens_pac4200"
    },
  "lemoco": {
    "start": "192.168.1.201",
    "end": "192.168.1.255",
    "profile": "lemoco_lmc168"
    }
  },
  "connection_settings": {
    "timeout": 5,
    "retries": 3,
    "unit_id": 1
  }
}

# measurement/meas_manager.py
"""
name: meas_manager.py
Author: Wasan (Refactored)
Date: 2024-06-20
Version: 2.0 (SOLID Architecture)
Description: Parallel Modbus TCP acquisition manager.
Handles concurrent polling of multiple meters and returns raw hardware data.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# นำเข้าคลาส ModbusMeter ที่เราเพิ่งแก้ไขไป
from measurement.modbusconv import ModbusMeter

class ParallelModbus:
    """
    คลาสสำหรับจัดการการสื่อสารกับ ModbusMeter หลายๆ ตัวพร้อมกันแบบขนาน (Parallel/Threading)
    """
    def __init__(self, meters: List[ModbusMeter]):
        self.meters = meters

    def connect_all(self) -> List[Dict[str, Any]]:
        """
        พยายามเชื่อมต่อมิเตอร์ทุกตัวในรายการ
        คืนค่าผลลัพธ์สถานะการเชื่อมต่อ
        """
        results = []
        for meter in self.meters:
            is_connected = False
            try:
                is_connected = meter.connect()
            except Exception as e:
                # กันเหนียว ป้องกันโค้ดหลักแครช
                print(f"⚠️ Exception while connecting to {meter.ip}: {e}")
            
            results.append({
                "ip": meter.ip,
                "id": meter.slave_id,
                "connected": is_connected
            })
        return results

    def close_all(self):
        """
        ตัดการเชื่อมต่อมิเตอร์ทุกตัวเมื่อใช้งานเสร็จ
        """
        for meter in self.meters:
            try:
                meter.close()
            except Exception:
                pass  # เพิกเฉยต่อ error ตอนสั่งปิด

    def get_measurements(self) -> Dict[str, Dict[str, Any]]:
        """
        ดึงข้อมูลจากมิเตอร์ทุกตัวแบบขนานด้วย ThreadPoolExecutor
        คืนค่าเป็น Dictionary โดยใช้ IP เป็น Key หลัก
        """
        if not self.meters:
            return {}

        batch_time = time.time()
        results_dict = {}

        def poll_single_meter(meter: ModbusMeter) -> Dict[str, Any]:
            """ฟังก์ชันย่อยสำหรับให้ Thread ทำงาน"""
            try:
                # ดึงข้อมูลจากคลาส ModbusMeter (ค่าที่ได้คูณ Prefix เป็นหน่วยฐานแล้ว)
                data = meter.get_measurement()
                data["batch_time"] = batch_time
                return data
            except Exception as e:
                return {
                    "ip": meter.ip,
                    "id": meter.slave_id,
                    "status": "error",
                    "error": str(e),
                    "batch_time": batch_time
                }

        # รัน ThreadPool สูงสุด 50 threads พร้อมกันเพื่อประหยัดทรัพยากร
        max_workers = min(len(self.meters), 50)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # ส่งคำสั่งให้ไปดึงข้อมูล
            future_to_meter = {executor.submit(poll_single_meter, meter): meter for meter in self.meters}
            
            # รอรับผลลัพธ์ที่ดึงสำเร็จ (ตัวไหนดึงเสร็จก่อน ก็เอาข้อมูลออกมาก่อน)
            for future in as_completed(future_to_meter):
                data = future.result()
                ip = data.get("ip")
                if ip:
                    results_dict[ip] = data

        # เรียงลำดับ Dictionary ตามหมายเลข IP เพื่อให้ดูง่ายเวลา Print หรือทำ Log
        sorted_results = dict(sorted(results_dict.items()))
        return sorted_results


class Management:
    """
    Facade/Wrapper Class เพื่อครอบการทำงานของ ParallelModbus อีกชั้น
    ช่วยให้เรียกใช้งานได้ง่ายขึ้นแบบคำสั่งเดียว
    """
    def __init__(self, meters: List[ModbusMeter]):
        self.group = ParallelModbus(meters)

    def run(self, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        เชื่อมต่อ -> ดึงข้อมูล -> ปิดการเชื่อมต่อ แบบจบในฟังก์ชันเดียว
        """
        if verbose:
            connect_results = self.group.connect_all()
            print("📡 Connect results:", connect_results)
        else:
            self.group.connect_all()

        measurements = self.group.get_measurements()

        if verbose:
            print(f"\n⚡ Measurements retrieved from {len(measurements)} meters.")

        self.group.close_all()
        return measurements


# ==============================================================================
# ตัวอย่างการเรียกใช้งานเพื่อทดสอบเฉพาะโมดูลนี้ (Standalone Testing)
# ==============================================================================
if __name__ == "__main__":
    import pymodbus
    print(f"🔧 PyModbus version: {getattr(pymodbus, '__version__', 'unknown')}")

    # 1. จำลอง Profile ฐานข้อมูล
    DEVICE_PROFILES = {
        "siemens_pac3200": {
            "V": {"addr": 7},
            "I": {"addr": 13},
            "P": {"addr": 65},
            "Q": {"addr": 67}
        },
        "lemoco_1000": {
            "V": {"addr": 1020},
            "I": {"addr": 1000},
            "P": {"addr": 1034, "prefix": "k"},
            "Q": {"addr": 1042, "prefix": "k"}
        }
    }

    # 2. สร้างมิเตอร์ (จำลองการดึง IP จากคอนฟิก)
    meters_to_test = [
        # ข้อควรระวัง: หากไม่มีมิเตอร์จริงรันอยู่ที่ IP นี้ โค้ดจะขึ้น Status: error
        ModbusMeter(ip="192.168.1.201", slave_id=1, device_profile=DEVICE_PROFILES["siemens_pac3200"], timeout=1.0),
        ModbusMeter(ip="192.168.1.202", slave_id=2, device_profile=DEVICE_PROFILES["lemoco_1000"], timeout=1.0),
    ]

    # 3. สั่งรันการดึงข้อมูล
    manager = Management(meters_to_test)
    raw_data = manager.run(verbose=True)

    # 4. แสดงผลลัพธ์ดิบ
    print("\n--- Raw Data from Modbus (Base Units) ---")
    for ip, data in raw_data.items():
        print(f"{ip} -> {data}")

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

import json
import ipaddress
import time
import struct
from threading import Lock
import pymodbus
from pymodbus.client import ModbusTcpClient


def parse_version(version_string: str):
    parts = version_string.split(".")
    nums = []

    for p in parts:
        digits = "".join(ch for ch in p if ch.isdigit())
        nums.append(int(digits) if digits else 0)

    while len(nums) < 3:
        nums.append(0)

    return tuple(nums[:3])


class ModbusCompat:
    PYMODBUS_VERSION = parse_version(
        getattr(pymodbus, "__version__", "0.0.0")
    )

    @classmethod
    def read_holding_registers(cls, client, address, count, slave_id):
        if cls.PYMODBUS_VERSION >= (3, 10, 0):
            return client.read_holding_registers(
                address=address,
                count=count,
                device_id=slave_id
            )
        elif cls.PYMODBUS_VERSION >= (3, 8, 0):
            return client.read_holding_registers(
                address=address,
                count=count,
                slave=slave_id
            )
        else:
            return client.read_holding_registers(
                address,
                count,
                slave=slave_id
            )


class ModbusMeter:
    PREFIX_MAP = {
        'k': 1e3,
        'M': 1e6,
        'G': 1e9,
        'm': 1e-3,
        'u': 1e-6
    }

    TYPE_FORMAT = {
        "float32": "f",
        "float64": "d",
        "int16": "h",
        "uint16": "H",
        "int32": "i",
        "uint32": "I",
        "int64": "q",
        "uint64": "Q"
    }

    from pathlib import Path


    _CONFIG = None
    try:
        config_path = Path(__file__).resolve().parent / "device_profile.json"

        with open(config_path, "r", encoding="utf-8") as f:
            _CONFIG = json.load(f)
            # print(f"Loaded config from {_CONFIG}")

    except Exception as e:
        print(f"Warning loading config: {e}")

    def __init__(self, ip, port=502):
        self.ip = ip
        self.port = port
        self._lock = Lock()

        conn = self._CONFIG.get("connection_settings", {})
        self.timeout = conn.get("timeout", 3)
        self.max_retry = conn.get("retries", 3)
        self.slave_id = conn.get("unit_id", 1)
        self.retry_delay = 0.5

        self.client = ModbusTcpClient(
            ip,
            port=port,
            timeout=self.timeout
        )

        self.profile = self._auto_select_profile(ip)

        if not self.profile:
            raise ValueError(f"No profile for {ip}")

        self.byte_order = self.profile.get("byte_order", "big")
        self.word_order = self.profile.get("word_order", "big")
        self.register_map = self.profile["registers"]

        regs = []
        for info in self.register_map.values():
            regs.append(info["reg"])
            regs.append(info["reg"] + info["count"] - 1)

        self.start_register = min(regs)
        self.block_count = max(regs) - self.start_register + 1

    def _auto_select_profile(self, target_ip):
        addr = ipaddress.ip_address(target_ip)

        profiles = self._CONFIG["device_profiles"]

        for _, r in self._CONFIG["ip_ranges"].items():
            start = ipaddress.ip_address(r["start"])
            end = ipaddress.ip_address(r["end"])

            if start <= addr <= end:
                if "profile" in r:
                    return profiles[r["profile"]]

        return None

    def _build_raw_bytes(self, regs_slice):
        words = list(regs_slice)

        if self.word_order == "little":
            words.reverse()

        raw = b""

        for word in words:
            high = (word >> 8) & 0xFF
            low = word & 0xFF

            if self.byte_order == "big":
                raw += bytes([high, low])
            else:
                raw += bytes([low, high])

        return raw

    def _decode_value(self, regs, address, count, dtype):
        offset = address - self.start_register

        if offset < 0 or offset + count > len(regs):
            return None

        regs_slice = regs[offset: offset + count]

        raw = self._build_raw_bytes(regs_slice)

        fmt_char = self.TYPE_FORMAT.get(dtype)

        if fmt_char is None:
            raise ValueError(f"Unsupported datatype: {dtype}")

        fmt = ">" + fmt_char

        expected_size = struct.calcsize(fmt)

        if len(raw) != expected_size:
            return None

        return struct.unpack(fmt, raw)[0]

    def _ensure_connection(self):
        try:
            if self.client.is_socket_open():
                return True
            return self.client.connect()
        except Exception:
            return False

    def _read_block(self):
        for _ in range(self.max_retry):
            if not self._ensure_connection():
                time.sleep(self.retry_delay)
                continue

            try:
                response = ModbusCompat.read_holding_registers(
                    self.client,
                    self.start_register,
                    self.block_count,
                    self.slave_id
                )

                if response and not response.isError():
                    return response.registers, None

            except Exception as e:
                err = str(e)

            self.client.close()
            time.sleep(self.retry_delay)

        return None, "read failed"

    def get_measurement(self):
        with self._lock:
            regs, error = self._read_block()

            result = {
                "ip": self.ip,
                "id": self.slave_id,
                "read_time": time.time()
            }

            if error:
                result["status"] = "error"
                result["error"] = error
                return result

            result["status"] = "ok"

            for key, info in self.register_map.items():
                value = self._decode_value(
                    regs,
                    info["reg"],
                    info["count"],
                    info["type"]
                )

                if value is not None:
                    prefix = info.get("prefix", "")
                    multiplier = self.PREFIX_MAP.get(prefix, 1.0)
                    result[key] = value * multiplier

            return result
        

if __name__ == "__main__":
    meter = ModbusMeter("192.168.1.201")
    data = meter.get_measurement()
    print(data)

# measurement/unit_converter.py
import dataclasses
import numpy as np
from typing import Optional, Any, List

# สมมติว่าดึงมาจากไฟล์ structural_data.py ของคุณ
from structural_data import Load, Generator, Measurement

PREFIX_MULTIPLIERS = {
    'T': 1e12, 'tera': 1e12,
    'G': 1e9,  'giga': 1e9,
    'M': 1e6,  'mega': 1e6,
    'k': 1e3,  'kilo': 1e3,
    'h': 1e2,  'hecto': 1e2,
    'da': 1e1, 'deca': 1e1,
    '': 1.0,   'base': 1.0, 'none': 1.0,
    'd': 1e-1, 'deci': 1e-1,
    'c': 1e-2, 'centi': 1e-2,
    'm': 1e-3, 'milli': 1e-3,
    'u': 1e-6, 'µ': 1e-6, 'micro': 1e-6,
    'n': 1e-9, 'nano': 1e-9,
    'p': 1e-12, 'pico': 1e-12,
}

class PrefixConverter:
    """แปลงค่าทางไฟฟ้าที่มีหน่วย Prefix ให้เป็นหน่วยฐาน"""
    @staticmethod
    def _get_multiplier(unit: str) -> float:
        if not unit: return 1.0
        unit_stripped, unit_lower = unit.strip(), unit.strip().lower()
        prefixes = sorted(PREFIX_MULTIPLIERS.keys(), key=len, reverse=True)
        
        for prefix in prefixes:
            if not prefix: continue
            if len(prefix) > 1:
                if unit_lower.startswith(prefix.lower()): return PREFIX_MULTIPLIERS[prefix]
            else:
                if unit_stripped.startswith(prefix): return PREFIX_MULTIPLIERS[prefix]
        return 1.0

    @staticmethod
    def _get_keys(obj: Any) -> List[str]:
        if isinstance(obj, dict): return list(obj.keys())
        elif dataclasses.is_dataclass(obj): return [f.name for f in dataclasses.fields(obj)]
        elif hasattr(obj, '__dict__'): return list(obj.__dict__.keys())
        return []

    @staticmethod
    def to_base(obj: Any, target_fields: Optional[List[str]] = None) -> Any:
        is_dict = isinstance(obj, dict)
        unit_str = obj.get('unit') if is_dict else getattr(obj, 'unit', None)
        if not unit_str: return obj
            
        multiplier = PrefixConverter._get_multiplier(unit_str)
        if multiplier == 1.0: return obj

        keys = target_fields if target_fields else PrefixConverter._get_keys(obj)
        for key in keys:
            if key in ('id', 'pos_id', 'bus_id', 'branch_id', 'unit') or key.endswith('_pu'):
                continue
            val = obj.get(key) if is_dict else getattr(obj, key, None)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                is_var = 'var' in key.lower() and key.lower() != 'var_pu'
                actual_multiplier = multiplier ** 2 if is_var else multiplier
                new_val = float(val) * actual_multiplier
                if is_dict: obj[key] = new_val
                else: setattr(obj, key, new_val)
        return obj

    @staticmethod
    def from_base(obj: Any, target_unit: str, target_fields: Optional[List[str]] = None) -> Any:
        """
        แปลงค่าจากหน่วยฐานเป็นหน่วย Prefix ตามที่ระบุ (เช่น target_unit="kilo") (In-place)
        """
        is_dict = isinstance(obj, dict)
        multiplier = PrefixConverter._get_multiplier(target_unit)
        
        if multiplier == 1.0:
            return obj

        keys = target_fields if target_fields else PrefixConverter._get_keys(obj)

        for key in keys:
            if key in ('id', 'pos_id', 'bus_id', 'branch_id', 'unit') or key.endswith('_pu'):
                continue
                
            val = obj.get(key) if is_dict else getattr(obj, key, None)
            
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                is_var = 'var' in key.lower() and key.lower() != 'var_pu'
                
                actual_multiplier = multiplier ** 2 if is_var else multiplier
                new_val = float(val) / actual_multiplier # หารเพื่อแปลงเป็นหน่วยที่ใหญ่ขึ้น
                
                if is_dict:
                    obj[key] = new_val
                else:
                    setattr(obj, key, new_val)
                    
        # อัปเดตชื่อหน่วยใหม่ให้ Object
        if is_dict:
            obj['unit'] = target_unit
        else:
            if hasattr(obj, 'unit'):
                setattr(obj, 'unit', target_unit)

        return obj

    @staticmethod
    def to_base_batch(objs: List[Any], target_fields: Optional[List[str]] = None) -> List[Any]:
        for obj in objs:
            PrefixConverter.to_base(obj, target_fields)
        return objs

    @staticmethod
    def from_base_batch(objs: List[Any], target_unit: str, target_fields: Optional[List[str]] = None) -> List[Any]:
        for obj in objs:
            PrefixConverter.from_base(obj, target_unit, target_fields)
        return objs


class PerUnitConvert:
    """
    Utility Class สำหรับแปลงค่าทางไฟฟ้าให้อยู่ในระบบ Per Unit (pu) 
    และแปลงกลับเป็นค่าจริง (Physical Value) (ใช้งานแบบ Static Methods)
    """

    @staticmethod
    def _get_base(mtype: str, V_base: float, S_base: float) -> Optional[float]:
        """เลือก Base ที่เหมาะสมโดยดูจากอักษรตัวแรกของชนิดตัวแปร"""
        if not mtype:
            return None
        mtype = mtype.lower()
        if mtype.startswith("v") and (not mtype.endswith("deg") and not mtype.endswith("angle")): 
            return V_base
        if mtype.startswith("i"): 
            return S_base / (np.sqrt(3) * V_base) # คำนวณ I_base สดๆ
        if mtype.startswith(("p", "q", "s")): 
            return S_base
        return None

    @staticmethod
    def _to_pu_scalar(mtype: str, value: float, V_base: float, S_base: float, is_variance: bool = False) -> float:
        """คำนวณค่าจริง -> Per Unit"""
        base = PerUnitConvert._get_base(mtype, V_base, S_base)
        if base is None or base == 0:
            return value
        
        if is_variance:
            return value / (base ** 2)
        return value / base

    @staticmethod
    def _from_pu_scalar(mtype: str, value_pu: float, V_base: float, S_base: float, is_variance: bool = False) -> float:
        """คำนวณ Per Unit -> ค่าจริง"""
        base = PerUnitConvert._get_base(mtype, V_base, S_base)
        if base is None:
            return value_pu
            
        if is_variance:
            return value_pu * (base ** 2)
        return value_pu * base

    @staticmethod
    def to_pu(obj: Any, V_base: float = 380.0, S_base: float = 1000.0) -> Any:
        """แปลงค่าใน Data Class เป็น Per Unit (In-place)"""
        if not dataclasses.is_dataclass(obj):
            return obj

        main_mtype = getattr(obj, 'mtype', None)

        for field in dataclasses.fields(obj):
            attr_name = field.name
            
            if attr_name.endswith('_pu'):
                base_attr = attr_name[:-3] 
                
                if hasattr(obj, base_attr):
                    val_pu = getattr(obj, attr_name)
                    val = getattr(obj, base_attr)

                    if val_pu is None and val is not None:
                        mtype_to_use = main_mtype if main_mtype else base_attr 
                        is_var = 'var' in base_attr.lower()
                        
                        # ส่ง V_base และ S_base ลงไปใน helper
                        new_pu_val = PerUnitConvert._to_pu_scalar(
                            mtype_to_use, val, V_base, S_base, is_variance=is_var
                        )
                        setattr(obj, attr_name, new_pu_val)
        return obj
        
    @staticmethod
    def from_pu(obj: Any, V_base: float = 380.0, S_base: float = 1000.0) -> Any:
        """แปลงค่าใน Data Class จาก Per Unit กลับเป็นค่าจริง (In-place)"""
        if not dataclasses.is_dataclass(obj):
            return obj

        main_mtype = getattr(obj, 'mtype', None)

        for field in dataclasses.fields(obj):
            attr_name = field.name
            
            if attr_name.endswith('_pu'):
                base_attr = attr_name[:-3] 
                
                if hasattr(obj, base_attr):
                    val_pu = getattr(obj, attr_name)
                    val = getattr(obj, base_attr)

                    if val is None and val_pu is not None:
                        mtype_to_use = main_mtype if main_mtype else base_attr 
                        is_var = 'var' in base_attr.lower()
                        
                        new_val = PerUnitConvert._from_pu_scalar(
                            mtype_to_use, val_pu, V_base, S_base, is_variance=is_var
                        )
                        setattr(obj, base_attr, new_val)
        return obj

    @staticmethod
    def to_pu_batch(objs: List[Any], V_base: float = 380.0, S_base: float = 1000.0) -> List[Any]:
        """รันแปลงค่าเป็น Per Unit ให้กับ List"""
        for obj in objs:
            PerUnitConvert.to_pu(obj, V_base, S_base)
        return objs

    @staticmethod
    def from_pu_batch(objs: List[Any], V_base: float = 380.0, S_base: float = 1000.0) -> List[Any]:
        """รันแปลงค่าจาก Per Unit กลับเป็นค่าจริง ให้กับ List"""
        for obj in objs:
            PerUnitConvert.from_pu(obj, V_base, S_base)
        return objs
    

class PowerDirection:
    @staticmethod
    def adjust_load_values(measurements: List['Measurement'], loads: List['Load']):
        load_buses = {load.lbus for load in loads}
        for m in measurements:
            if m.pos_id in load_buses and m.mtype and m.mtype.lower() in ['pinject', 'qinject']:
                m.mvalue = -abs(m.mvalue)
                if m.mvalue_pu is not None:
                    m.mvalue_pu = -abs(m.mvalue_pu)

    @staticmethod
    def adjust_gen_values(measurements: List['Measurement'], generators: List['Generator']):
        gen_buses = {gen.gbus for gen in generators}
        for m in measurements:
            if m.pos_id in gen_buses and m.mtype and m.mtype.lower() in ['pinject', 'qinject']:
                m.mvalue = abs(m.mvalue)
                if m.mvalue_pu is not None:
                    m.mvalue_pu = abs(m.mvalue_pu)
# structural_data.py

from dataclasses import dataclass
from typing import List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(kw_only=True)
class Component:
    id: Optional[int] = None
    name: Optional[str] = None
    status : Optional[bool] = True
    unit: Optional[str] = None
    

@dataclass(kw_only=True)
class Bus:
    """
    bus_id : unique integer bus id  (required)
    bus_type  : 'PQ' | 'PV' | 'slack'
    slack  : True marks the reference / slack bus
    """
    id: int
    name: Optional[str] = None
    type: Optional[str] = None
    slack: Optional[bool] = None
    nominal: Optional[float] = None
    unit : Optional[str] = None

    @property
    def is_slack(self) -> bool:
        return bool(self.slack) or (
            isinstance(self.type, str) and self.type.lower() == "slack"
        )


@dataclass(kw_only=True)
class Branch:
    """
    id : unique branch id
    fbus      : from-bus id
    tbus      : to-bus id
    rs        : series resistance   [pu]
    xs        : series reactance    [pu]
    xsh       : total shunt susceptance (charging) [pu] – split equally at ends
    """
    id: int
    name: str = None
    fbus: int
    tbus: int
    rs: float
    xs: float
    xsh: float = 0.0


@dataclass(kw_only=True)
class Generator:
    """
    gen_id  : unique generator id
    gbus    : bus id where generator is connected
    pg      : active power generation [pu]
    qg      : reactive power generation [pu]
    unit    : optional label
    """
    id: int
    name: Optional[str] = None
    gbus: int
    pg: Optional[float] = None
    qg: Optional[float] = None
    vgen: Optional[float] = None
    pg_pu: Optional[float] = None
    qg_pu: Optional[float] = None
    vgen_pu: Optional[float] = None
    unit: Optional[str] = None


@dataclass(kw_only=True)
class Load:
    """
    load_id : unique load id
    lbus    : bus id where load is connected
    pl      : active power load [pu]
    ql      : reactive power load [pu]
    unit    : optional label
    """
    id: int
    name: Optional[str] = None
    lbus: int
    pl: Optional[float] = None
    ql: Optional[float] = None
    pl_pu: Optional[float] = None
    ql_pu: Optional[float] = None
    unit: Optional[str] = None



@dataclass(kw_only=True)
class Measurement:
    """
    mpostyp: 'bus' | 'branch' (or 'line') | 'Gen' | 'Load' – position of measurement
    mpos: 1 | 2 | 3 … (bus_id or branch_id or gen_id or load_id)
    mtype   : 'P' | 'Q' | 'V' | 'pflow' | 'qflow' | 'pinject' | 'qinject' | etc.
    mid     : unique measurement id
    mbus    : bus id         (pinject / qinject / vmag)
    mbranch : branch_id      (pflow / qflow)
    mside   : 'from' | 'to'  (pflow / qflow)
    mvalue  : measured value [pu]
    msd     : standard deviation σ [pu]
    """
    id: int
    name: str = None #P V Q I
    position: str = None  # 'bus', 'branch', 'gen', 'load'
    pos_id: int = None
    mtype: Optional[str] = None
    mvalue: float = 0.0
    msd: float = 1.0
    mvalue_pu: Optional[float] = None
    msd_pu: Optional[float] = None
    var: Optional[float] = 1.0
    var_pu: Optional[float] = None
    unit: Optional[str] = None

    bus_id: Optional[int] = None
    branch_id: Optional[int] = None
    # mside: Optional[str] = None #for branch flow measurements

    def __post_init__(self):
        # Smart Sign Convention for P, Q, I
        if self.position and self.position.lower() in ['bus', 'branch', 'line'] and self.pos_id is not None:
            if self.mtype is None:
                position = self.position.lower()  # take the first word and lowercase it
                name = self.name.lower() if self.name else ""

                if position.startswith('bus') or position.startswith('b'):
                    # buses_ids = [bus.id for bus in buses]
                    # self.bus_id = self.pos_id if self.pos_id in buses_ids else None
                    if name.startswith('p'):
                        self.mtype = 'pinject'
                    elif name.startswith('q'):
                        self.mtype = 'qinject'
                    elif name.startswith('v'):
                        self.mtype = 'vmag'
                elif position.startswith('branch') or position.startswith('line') or position.startswith('l'):
                    # branches_ids = [branch.id for branch in branches]
                    # self.branch_id = self.pos_id if self.pos_id in branches_ids else None
                    if name.startswith('p'):
                        self.mtype = 'pflow'
                    elif name.startswith('q'):
                        self.mtype = 'qflow'
                    elif name.startswith('i'):
                        self.mtype = 'iflow'


if __name__ == "__main__":
    buses = [
        Bus(id=1, name="Bus1", type="slack", slack=True),
        Bus(id=2, name="Bus2",  type="PQ"),
        Bus(id=3, name="Bus3",  type="PQ"),
    ]
    branches = [
        Branch(id=1, name="Branch1", fbus=1, tbus=2, rs=0.035, xs=0.25, xsh=0.00),
        Branch(id=2, name="Branch2", fbus=1, tbus=3, rs=0.035, xs=0.25, xsh=0.00),
        Branch(id=3, name="Branch3", fbus=2, tbus=3, rs=0.035, xs=0.25, xsh=0.00),
    ]
    generators = [
        Generator(id=1, name="Gen1", gbus=1, pg=0.2148315, qg=0.000171),
    ]
    loads = [
        Load(id=1, name="Load1", lbus=2),
        Load(id=2, name="Load2", lbus=3),
    ]

    measurements = [
        Measurement(id=1, name="p1", position="bus", pos_id=1, mvalue=0.2148315, msd=0.010),
        Measurement(id=2, name="q1", position="bus", pos_id=1, mvalue=0.000171, msd=0.010),
        Measurement(id=3, name="p2", position="bus", pos_id=2, mvalue=0.141229, msd=0.010),
        Measurement(id=4, name="q2", position="bus", pos_id=2, mvalue=0.000171, msd=0.010),
        Measurement(id=5, name="p3", position="bus", pos_id=3, mvalue=0.07211, msd=0.010),
        Measurement(id=6, name="q3", position="bus", pos_id=3, mvalue=0.0000018, msd=0.010),]
    
    print("\nMeasurements:")
    for m in measurements:
        print(f"  {m.mvalue:.5f}")


    '''
    # bus
    mbus: Optional[int] = None

    # branch
    mbranch: Optional[int] = None
    mside: Optional[str] = None

    # generator
    mgen: Optional[int] = None

    # load
    mload: Optional[int] = None

    ── Dynamic Properties ──────────────────────────────────────────
    @property
    def mbus(self) -> Optional[int]:
        if self.position and self.position.lower() in ['bus', 'b']:
            return self.pos_id
        return None

    @property
    def mbranch(self) -> Optional[int]:
        if self.position and self.position.lower() in ['branch', 'line', 'l']:
            return self.id
        return None

    @property
    def mgen(self) -> Optional[int]:
        if self.position and self.position.lower() in ['gen', 'g']:
            return self.id
        return None

    @property
    def mload(self) -> Optional[int]:
        if self.position and self.position.lower() in ['load', 'ld']:
            return self.id
        return None
    '''
    '''
    def __post_init__(self):
        # if self.comp_type is not None and self.comp_id is not None:
        #     if self.comp_type.lower() in ['bus', 'b']:
        #         self.mbus = self.comp_id
        #     elif self.comp_type.lower() in ['branch', 'line', 'l']:
        #         self.mbranch = self.comp_id
        #     elif self.comp_type.lower() in ['gen', 'g']:
        #         self.mgen = self.comp_id
        #     elif self.comp_type.lower() in ['load', 'ld']:
        #         self.mload = self.comp_id

        # 2. Smart Sign Convention for P, Q, I
        if self.mtype and self.mvalue is not None:
            mtype_lower = self.mtype.lower()
            
            # เช็คว่าเป็นการวัด Active Power (P), Reactive Power (Q) หรือ Current (I)
            is_power_or_current = mtype_lower.startswith('p') or \
                                  mtype_lower.startswith('q') or \
                                  mtype_lower.startswith('i')
            
            if is_power_or_current:
                if self.mload is not None:
                    # ถ้าเป็น Load บังคับให้เป็นค่าติดลบ (-) เสมอ
                    self.mvalue = -abs(self.mvalue)
                    if self.mvalue_pu is not None:
                        self.mvalue_pu = -abs(self.mvalue_pu)
                
                elif self.mgen is not None:
                    # ถ้าเป็น Gen บังคับให้เป็นค่าบวก (+) เสมอ
                    self.mvalue = abs(self.mvalue)
                    if self.mvalue_pu is not None:
                        self.mvalue_pu = abs(self.mvalue_pu)
    '''


# @dataclass
# class MeasurementMapping:
#     mid: int
#     mip: str
#     mpos: str
#     mside: Optional[str] = None

#     def __init__(self, MeterGroup: MeterGroup):
#         self.mapping = {}
#         for m in MeterGroup.get_measurements().values():
#             mid = m.get("mid")
#             mip = m.get("mip")-
#             mpos = m.get("mpos")
#             mside = m.get("mside")


#     def __post_init__(self):
#         if MeterGroup.get_measurements:

# from dataclasses import dataclass
# from typing import Optional, List, Dict

# @dataclass
# class MeasurementMapping:
#     mip: str           # IP ของมิเตอร์ที่ใช้อ่านค่า
#     meter_key: str     # คีย์ของข้อมูลที่มิเตอร์อ่านมาได้ (เช่น 'p', 'q', 'v', 'kw')
#     mtype: str         # ประเภทข้อมูลในระบบ ('pinject', 'pflow', 'vmag', ฯลฯ)
#     mid: int           # ID สำหรับ Measurement
#     mpos: str          # 'bus' หรือ 'branch'
#     bid: int           # ID ของ Bus หรือ Branch (ใช้ตัวแปรเดียวเพื่อให้สั้นลง)
#     mside: Optional[str] = None # 'from' หรือ 'to' (เฉพาะกรณีเป็น branch)
#     msd: float = 1.0   # ค่า Standard Deviation

# def process_meter_data(meter_results: dict, mappings: List[MeasurementMapping]) -> List['Measurement']:
#     """
#     แปลงข้อมูลดิบจาก MeterGroup ให้กลายเป็น Measurement Objects
#     """
#     measurements = []
    
#     for mapping in mappings:
#         # 1. เช็คว่า IP ที่ระบุใน Mapping มีข้อมูลส่งกลับมาไหม
#         if mapping.mip not in meter_results:
#             print(f"Warning: ขาดการเชื่อมต่อหรือไม่มีข้อมูลจาก IP {mapping.mip}")
#             continue
            
#         device_data = meter_results[mapping.mip]
        
#         # 2. เช็คว่า มิเตอร์ตัวนี้ มีคีย์ข้อมูล (meter_key) ที่เราต้องการไหม
#         if mapping.meter_key not in device_data:
#             print(f"Warning: มิเตอร์ IP {mapping.mip} ไม่มีข้อมูลคีย์ '{mapping.meter_key}'")
#             continue
            
#         # ดึงค่าที่วัดได้
#         val = device_data[mapping.meter_key]
        
#         # 3. สร้าง Object Measurement ตามตำแหน่ง (Bus หรือ Branch)
#         if mapping.mpos.lower() == 'bus':
#             m = Measurement(
#                 mtype=mapping.mtype, 
#                 mid=mapping.mid, 
#                 mbus=mapping.bid, 
#                 mvalue=val, 
#                 msd=mapping.msd
#             )
#         elif mapping.mpos.lower() in ['branch', 'line']:
#             m = Measurement(
#                 mtype=mapping.mtype, 
#                 mid=mapping.mid, 
#                 mbranch=mapping.bid, 
#                 mside=mapping.mside, 
#                 mvalue=val, 
#                 msd=mapping.msd
#             )
#         else:
#             continue
            
#         measurements.append(m)
        
#     return measurements



# # สมมติคลาส Measurement (ย่อมาจากของคุณ)
# @dataclass
# class Measurement:
#     mtype: str
#     mid: int
#     mbus: Optional[int] = None
#     mbranch: Optional[int] = None
#     mside: Optional[str] = None
#     mvalue: float = 0.0
#     msd: float = 1.0

# ---------------------------------------------------------
# สถานการณ์จำลอง
# ---------------------------------------------------------

# # 1. จำลองผลลัพธ์ที่ได้จาก meter_group.get_measurements() (เรียงตาม IP แล้ว)
# # ปกติค่านี้จะได้มาจาก Hardware จริง
# mock_meter_results = {
#     "192.168.1.20": {"ip": "192.168.1.20", "id": 1, "p_kw": 1.5, "q_kvar": 0.5, "v_ln": 1.02},
#     "192.168.1.21": {"ip": "192.168.1.21", "id": 2, "p_flow": 2.1, "q_flow": 0.8}
# }

# # 2. สร้างกฎการแปลงข้อมูล (Mapping Configuration)
# # ปกติส่วนนี้อาจจะดึงมาจาก Database หรือไฟล์ JSON/Excel
# mappings_config = [
#     # มิเตอร์ตัวแรกติดที่ Bus 1 (อ่าน 3 ค่า P, Q, V)
#     MeasurementMapping(mip="192.168.1.20", meter_key="p_kw",   mtype="pinject", mid=101, mpos="bus", bid=1),
#     MeasurementMapping(mip="192.168.1.20", meter_key="q_kvar", mtype="qinject", mid=102, mpos="bus", bid=1),
#     MeasurementMapping(mip="192.168.1.20", meter_key="v_ln",   mtype="vmag",    mid=103, mpos="bus", bid=1, msd=0.01),
    
#     # มิเตอร์ตัวที่สองติดที่ต้นทางของ Branch 5 (อ่าน Flow P, Q)
#     MeasurementMapping(mip="192.168.1.21", meter_key="p_flow", mtype="pflow",   mid=201, mpos="branch", bid=5, mside="from"),
#     MeasurementMapping(mip="192.168.1.21", meter_key="q_flow", mtype="qflow",   mid=202, mpos="branch", bid=5, mside="from"),
# ]

# # 3. ประมวลผล
# final_measurements = process_meter_data(mock_meter_results, mappings_config)

# # 4. ดูผลลัพธ์
# for m in final_measurements:
#     print(m)

# solver.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from structural_data import Bus, Branch, Measurement
from math_model.static_parameter import YBusMatrix, MeasurementMatrix
from math_model.dynamic_parameter import StateVector, MeasurementFunc, Jacobian

from measurement.unit_converter import PerUnitConvert, PowerDirection


# @dataclass
# class SEResult:
#     """
#     Full solution of the WLS state estimation.
#     """
#     x_est       : np.ndarray
#     converged   : bool
#     iterations  : int
#     theta_deg   : np.ndarray
#     V_pu        : np.ndarray
#     residuals   : np.ndarray
#     norm_res    : np.ndarray
#     J           : float
#     method_used : str = "analytical"
#     gain_matrix : np.ndarray = field(default_factory=lambda: np.array([]))
@dataclass
class BusResult:
    bus_id      : int
    mtype       : float
    value_pu    : float
    value       : Optional[float] = None
    residual    : Optional[float] = None
    residual_pu : Optional[float] = None
    norm_res    : Optional[float] = None


@dataclass
class BranchResult:
    branch_id   : int
    mtype       : str
    value_pu       : float
    value    : Optional[float] = None
    residual    : Optional[float] = None
    residual_pu : Optional[float] = None
    norm_res    : Optional[float] = None


@dataclass
class SEResult:
    """
    Full solution of the WLS state estimation.
    """
    x_est       : np.ndarray
    converged   : bool
    iterations  : int
    theta_deg   : np.ndarray
    
    # Paring V and V_pu
    V_pu        : np.ndarray
    
    # ถ้ามี P, Q ที่ได้จาก State Estimation ด้วย ก็ตั้งชื่อให้จับคู่กันแบบนี้ได้
    # P_est       : Optional[np.ndarray] = None
    # P_est_pu    : Optional[np.ndarray] = None
    
    residuals   : np.ndarray
    norm_res    : np.ndarray
    J           : float
    # bus_result: Optional[dict] = field(default_factory=dict)
    # branch_result: Optional[dict] = field(default_factory=dict)
    method_used : str = "analytical"
    V           : Optional[np.ndarray] = None
    gain_matrix : np.ndarray = field(default_factory=lambda: np.array([]))
    # system_state_dict: Dict = field(default_factory=dict)
    


# ══════════════════════════════════════════════════════════════════════════════
# SOLVER
# ══════════════════════════════════════════════════════════════════════════════

class WLSSolver:
    """
    Newton-Raphson WLS State Estimator.
    """

    def __init__(
        self,
        jac      : Jacobian,
        sv       : StateVector,
        mm       : MeasurementMatrix,
        method   : str   = "analytical",
        max_iter : int   = 50,
        tol      : float = 1e-6,
    ) -> None:
        self.jac      = jac
        self.sv       = sv
        self.mm       = mm
        self.method   = method.lower()
        self.max_iter = max_iter
        self.tol      = tol
        self.theta     = None
        self.V         = None

        if self.method not in ("analytical", "numerical"):
            raise ValueError("method must be 'analytical' or 'numerical'")


    def solve(
        self,
        x0      : Optional[np.ndarray] = None,
        verbose : bool = True,
    ) -> SEResult:
        
        x = self.sv.flat_start() if x0 is None else x0.copy()
        z, W = self.mm.z, self.mm.W

        H_func = (
            self.jac.H_analytical
            if self.method == "analytical"
            else self.jac.H_numerical
        )

        if verbose:
            self._print_header()

        converged   = False
        G_wls_final = np.empty((0, 0))

        for it in range(1, self.max_iter + 1):
            hx    = self.jac.h(x)
            r     = z - hx
            H     = H_func(x)
            G_wls = H.T @ W @ H
            rhs   = H.T @ W @ r
            dx    = _solve_linear(G_wls, rhs)

            x += dx
            max_dx      = float(np.max(np.abs(dx)))
            J_iter      = float(r @ W @ r)
            G_wls_final = G_wls

            if verbose:
                print(f"  {it:>4}  {max_dx:>14.6e}  {J_iter:>14.4f}")

            if max_dx < self.tol:
                converged = True
                break

        if verbose:
            _sep()
            status = (
                f"✓ Converged in {it} iteration(s)."
                if converged
                else f"✗ Did NOT converge after {self.max_iter} iterations."
            )
            print(f"  {status}")
            _sep()
        
        hx_final  = self.jac.h(x)
        residuals = z - hx_final
        sigma     = np.sqrt(np.diag(self.mm.R))
        norm_res  = residuals / sigma
        J_final   = float(residuals @ W @ residuals)
        theta, V  = self.sv.unpack(x)
        self.theta = theta
        self.V     = V
        

        return SEResult(
            x_est       = hx_final,
            # x_est_final = hx_final,
            # x_est_pu    = self.jac.h_pu(x),
            converged   = converged,
            iterations  = it,
            theta_deg   = np.degrees(theta),
            V_pu        = V,
            residuals   = residuals,
            norm_res    = norm_res,
            J           = J_final,
            method_used = self.method,
            gain_matrix = G_wls_final,
            # system_state_dict = dashboard_data, # ส่ง Dict กลับไปที่นี่
            
        )
    
    @staticmethod
    def _print_header() -> None:
        _banner("WLS ITERATION LOG")
        print(f"  {'Iter':>4}  {'max|Δx|':>14}  {'J(x)':>14}")
        _sep()

def _to_dict(bus_result: List[BusResult], branch_result: List[BranchResult]) -> Dict:

    dashboard_dict = {"buses": {}, "branches": {}}
    
    # 1. จัดกลุ่ม Bus
    for b in bus_result:
        if b.bus_id not in dashboard_dict["buses"]:
            dashboard_dict["buses"][b.bus_id] = {}
            
        # ใช้ mtype (เช่น vmag, pinject) เป็น Key ย่อย
        dashboard_dict["buses"][b.bus_id][b.mtype] = {
            "value_pu": b.value_pu,
            "value": b.value  # ค่านี้จะโผล่มาหลังจากผ่าน PerUnitConvert
        }

    # 2. จัดกลุ่ม Branch
    for br in branch_result:
        if br.branch_id not in dashboard_dict["branches"]:
            dashboard_dict["branches"][br.branch_id] = {}
            
        dashboard_dict["branches"][br.branch_id][br.mtype] = {
            "value_pu": br.value_pu,
            "value": br.value
        }

    return dashboard_dict



class Estimation(WLSSolver):
    
    def run(self, verbose: bool = True) -> SEResult:
        result = self.solve(verbose=verbose)
        return result

    def get_estimate_var(self, v_base: float, s_base: float, jac: Jacobian) -> dict:
        v = self.V
        # print("\nVoltage magnitudes (pu):", type(v))
        th = self.theta
        
        # สรุปผล State Variables ทั้งหมดให้อยู่ในรูปแบบ Dictionary สำหรับ Dashboard
        # ─────────────────────────────────────────────────────────────────
        dashboard_data = {
            "buses": {},
            "branches": {}
        }

        bus_detail = []
        branch_detail = []

        bus_keys = ["vmag", "vang_deg", "pinject", "qinject"]
        branch_keys = ["pflow_from", "qflow_from", "pflow_to", "qflow_to"]
        # 1. คำนวณ P, Q, V, Theta สำหรับทุกบัส
        for bus in self.sv.buses:
            i = self.sv.ybus.bus_index(bus.id)
            bus_vals = {
                "vmag": float(v[i]),
                "vang_deg": float(np.degrees(th[i])),
                "pinject": float(self.jac._Pi(i, th, v)),
                "qinject": float(self.jac._Qi(i, th, v))
            }
            dashboard_data["buses"][bus.id] = bus_vals
            measured_keys = []
            for m in self.mm.measurements:
                if m.mtype in bus_keys and m.pos_id == bus.id:
                    mtype = m.mtype
                    measured_keys.append(mtype)
                    # bus_detail.append(BusResult(bus_id=bus.id, mtype=mtype, value_pu=dashboard_data["buses"][bus.id][mtype]))
                # bus_detail.append(BusResult(bus_id=bus.id, mtype="vmag", value_pu=dashboard_data["buses"][bus.id]["vmag"]))
                # bus_detail.append(BusResult(bus_id=bus.id, mtype="vang_deg", value_pu=dashboard_data["buses"][bus.id]["vang_deg"]))
                # bus_detail.append(BusResult(bus_id=bus.id, mtype="pinject", value_pu=dashboard_data["buses"][bus.id]["pinject"]))
                # bus_detail.append(BusResult(bus_id=bus.id, mtype="qinject", value_pu=dashboard_data["buses"][bus.id]["qinject"]))
                # else: 
                #     mtype = "vang_deg"
                #     bus_detail.append(BusResult(bus_id=bus.id, mtype=mtype, value_pu=dashboard_data["buses"][bus.id][mtype]))
            measured_keys = [m.mtype for m in self.mm.measurements if m.pos_id == bus.id]
            added_keys = set()
            for mkey in measured_keys:
                    if mkey not in added_keys:
                
                        bus_detail.append(BusResult(bus_id=bus.id, mtype=f"{mkey[0:1]}{bus.id}", value_pu=bus_vals[mkey]))
                        added_keys.add(mkey)

            # 4. ลำดับท้าย: ใส่ค่าที่ "ไม่มี" ใน Measurement ลงไปด้านล่าง
            for key in bus_keys:
                if key not in added_keys:
                    bus_detail.append(BusResult(bus_id=bus.id, mtype=key, value_pu=bus_vals[key]))
        
        bus_result = PerUnitConvert.from_pu_batch(bus_detail, v_base, s_base)
        # bus = PerUnitConvert.from_pu_batch(bus_detail)
        # bus = bus_detail
        print("\nBus details in per unit:")
        for b in bus_result:
            print(b) 
        # bus = PerUnitConvert.to_pu_batch(bus_detail)
        # 2. คำนวณ P_flow, Q_flow สำหรับทุกสายส่ง (Branch) ทั้งฝั่ง from และ to
        for br in self.sv.ybus.branches:
            bid = br.id
            bf = self.sv.ybus.bus_index(br.fbus)+1
            bt = self.sv.ybus.bus_index(br.tbus)+1
            dashboard_data["branches"][bid] = {
                f"p{bf}{bt}_from": float(self.jac._Pij(bid, "from", th, v)),
                f"q{bf}{bt}_from": float(self.jac._Qij(bid, "from", th, v)),
                f"p{bf}{bt}_to": float(self.jac._Pij(bid, "to", th, v)),
                f"q{bf}{bt}_to": float(self.jac._Qij(bid, "to", th, v))
            }
            print(f"\np{bf}{bt}_from:", dashboard_data["branches"][bid][f"p{bf}{bt}_from"])
            branch_detail.append(BranchResult(branch_id=bid, mtype=f"p{bf}{bt}_from", value_pu=dashboard_data["branches"][bid][f"p{bf}{bt}_from"]))
            branch_detail.append(BranchResult(branch_id=bid, mtype=f"q{bf}{bt}_from", value_pu=dashboard_data["branches"][bid][f"q{bf}{bt}_from"]))
            branch_detail.append(BranchResult(branch_id=bid, mtype=f"p{bf}{bt}_to", value_pu=dashboard_data["branches"][bid][f"p{bf}{bt}_to"]))
            branch_detail.append(BranchResult(branch_id=bid, mtype=f"q{bf}{bt}_to", value_pu=dashboard_data["branches"][bid][f"q{bf}{bt}_to"]))
        branch_result = PerUnitConvert.from_pu_batch(branch_detail, v_base, s_base)
        data = _to_dict(bus_result, branch_result)  # แปลง BusResult เป็น Dict สำหรับ Dashboard
        # print("\nDashboard Data (per unit):", data)
        bus_result = data["buses"]
        
        branch_result = data["branches"]
        
        # print("\nBus Results (per unit):", bus_result)
        # print("\nBranch Results (per unit):", branch_result)
        return {"buses": bus_result, "branches": branch_result}
    


# class Result(WLSSolver):
#     @staticmethod
#     def result( se_result: SEResult, v_base: float, s_base: float, jac: Jacobian) -> dict:
#         v = se_result.V_pu
#         print("\nVoltage magnitudes (pu):", type(v))
#         th = np.radians(se_result.theta_deg)
#         # สรุปผล State Variables ทั้งหมดให้อยู่ในรูปแบบ Dictionary สำหรับ Dashboard
#         # ─────────────────────────────────────────────────────────────────
#         dashboard_data = {
#             "buses": {},
#             "branches": {}
#         }

#         bus_detail = []
#         branch_detail = []
#         # 1. คำนวณ P, Q, V, Theta สำหรับทุกบัส
#         for bus in jac.sv.buses:
            
#             i = jac.sv.ybus.bus_index(bus.id)
#             dashboard_data["buses"][bus.id] = {
#                 "vmag": float(v[i]),
#                 "vang_deg": float(th[i]),
#                 "pinject": float(jac._Pi(i, th, v)),
#                 "qinject": float(jac._Qi(i, th, v))
#             }

#             bus_detail.append(BusResult(bus_id=bus.id, mtype="vmag", value_pu=dashboard_data["buses"][bus.id]["vmag"]))
#             bus_detail.append(BusResult(bus_id=bus.id, mtype="vang_deg", value_pu=dashboard_data["buses"][bus.id]["vang_deg"]))
#             bus_detail.append(BusResult(bus_id=bus.id, mtype="pinject", value_pu=dashboard_data["buses"][bus.id]["pinject"]))
#             bus_detail.append(BusResult(bus_id=bus.id, mtype="qinject", value_pu=dashboard_data["buses"][bus.id]["qinject"]))
#         bus_result = PerUnitConvert.from_pu_batch(bus_detail, v_base, s_base)
#         # bus = PerUnitConvert.from_pu_batch(bus_detail)
#         # bus = bus_detail
#         # print("\nBus details in per unit:")
#         # for b in bus:
#         #     print(b) 
#         # bus = PerUnitConvert.to_pu_batch(bus_detail)
#         # 2. คำนวณ P_flow, Q_flow สำหรับทุกสายส่ง (Branch) ทั้งฝั่ง from และ to
#         for br in jac.ybus.branches:
#             bid = br.id
#             dashboard_data["branches"][bid] = {
#                 "pflow_from": float(jac._Pij(bid, "from", th, v)),
#                 "qflow_from": float(jac._Qij(bid, "from", th, v)),
#                 "pflow_to": float(jac._Pij(bid, "to", th, v)),
#                 "qflow_to": float(jac._Qij(bid, "to", th, v))
#             }
#             branch_detail.append(BranchResult(branch_id=bid, mtype="pflow_from", value_pu=dashboard_data["branches"][bid]["pflow_from"]))
#             branch_detail.append(BranchResult(branch_id=bid, mtype="qflow_from", value_pu=dashboard_data["branches"][bid]["qflow_from"]))
#             branch_detail.append(BranchResult(branch_id=bid, mtype="pflow_to", value_pu=dashboard_data["branches"][bid]["pflow_to"]))
#             branch_detail.append(BranchResult(branch_id=bid, mtype="qflow_to", value_pu=dashboard_data["branches"][bid]["qflow_to"]))
#         branch_result = PerUnitConvert.from_pu_batch(branch_detail, v_base, s_base)
#         result = _to_dict(bus_result, branch_result)  # แปลง BusResult เป็น Dict สำหรับ Dashboard
#         # print("\nDashboard Data (per unit):", result)
#         bus_result = result["buses"]
        
#         branch_result = result["branches"]
        
#         # print("\nBus Results (per unit):", bus_result)
#         # print("\nBranch Results (per unit):", branch_result)
#         return {"buses": bus_result, "branches": branch_result}
    
# class ResultExtractor:
#     """ตัวช่วยสกัดข้อมูล State Estimation เพื่อส่งให้ Dashboard"""
    
#     @staticmethod
#     def get_dashboard_data(se_result: SEResult, jac: Jacobian, v_base: float, s_base: float) -> dict:
#         # ดึง V และ theta ออกมาจากผลลัพธ์ของ Solver
#         V = se_result.V_pu
#         theta = np.radians(se_result.theta_deg) # แปลงองศากลับเป็นเรเดียนเพื่อเข้าสมการ Jacobian

#         dashboard_data = {"buses": {}, "branches": {}}

#         # 1. คำนวณ P, Q, V, Theta สำหรับทุกบัส
#         for bus in jac.sv.buses:
#             i = jac.ybus.bus_index(bus.id)
#             vmag_pu = float(V[i])
#             vang_deg = float(np.degrees(theta[i]))
#             pinject_pu = float(jac._Pi(i, theta, V))
#             qinject_pu = float(jac._Qi(i, theta, V))

#             # เก็บค่า PU และแอบคูณ Base กลับเป็นหน่วยจริง (Physical Value) ให้เลย จบในที่เดียว!
#             dashboard_data["buses"][bus.id] = {
#                 "vmag_pu": vmag_pu,
#                 "vmag": vmag_pu * v_base,
#                 "vang_deg": vang_deg,
#                 "pinject_pu": pinject_pu,
#                 "pinject": pinject_pu * s_base,
#                 "qinject_pu": qinject_pu,
#                 "qinject": qinject_pu * s_base
#             }

#         # 2. คำนวณ P_flow, Q_flow สำหรับสายส่ง
#         for br in jac.ybus.branches:
#             bid = br.id
#             dashboard_data["branches"][bid] = {
#                 "pflow_from_pu": float(jac._Pij(bid, "from", theta, V)),
#                 "qflow_from_pu": float(jac._Qij(bid, "from", theta, V)),
#                 "pflow_to_pu": float(jac._Pij(bid, "to", theta, V)),
#                 "qflow_to_pu": float(jac._Qij(bid, "to", theta, V)),
#             }
            
#             # แปลงเป็นหน่วยจริงทั้งหมด
#             for key in list(dashboard_data["branches"][bid].keys()):
#                 real_key = key.replace("_pu", "")
#                 dashboard_data["branches"][bid][real_key] = dashboard_data["branches"][bid][key] * s_base

#         return dashboard_data

#     # @staticmethod
#     # def bus_result() -> dict:
#     #     return bus_result
    
#     # @staticmethod
#     # def branch_result() -> dict:
#     #     return branch_result

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_results(
    result       : SEResult,
    ybus         : YBusMatrix,
    bus_2idx: dict[Bus],
    branch_2idx: dict[Branch],
    measurements : Optional[List[Measurement]] = None,
) -> None:
    """
    Print a formatted table of bus voltages and measurement residuals.
    """
    _banner("WLS STATE ESTIMATION RESULTS")
    print(f"  Status     : {'CONVERGED ✓' if result.converged else 'NOT CONVERGED ✗'}")
    print(f"  Iterations : {result.iterations}")
    print(f"  Jacobian   : {result.method_used}")
    print(f"  Objective J: {result.J:.6f}")

    _sep()
    print(f"  {'Bus':>4}  {'Name':>12}  {'θ (deg)':>10}  {'V (pu)':>10}  {'Type'}")
    _sep()
    for b in ybus.buses:
        i    = ybus.bus_index(b.id)
        name = b.name or "—"
        btyp = b.type or ("slack" if b.slack else "—")
        print(f"  {b.id:>4}  {name:>12}  "
              f"{result.theta_deg[i]:>10.4f}  {result.V_pu[i]:>10.6f}  {btyp}")

    _sep()
    print(f"\n  {'#':>3}  {'mid':>4}  {'mtype':>10}  "
          f"{'z (meas)':>10}  {'h(x̂)':>10}  {'hval':>10}  {'residual':>10}  {'|r/σ|':>7}")
    _sep()

    for k, (r, nr, hx) in enumerate(zip(result.residuals, result.norm_res, result.x_est)):
    #     if measurements is not None:
    #         m   = measurements[k]
    #         mid = m.id
    #         mt  = m.name
    #         mv  = m.mvalue  # แปลงกลับจาก pu เป็นหน่วยจริงสำหรับการแสดงผลเฉพาะ V เท่านั้น
    #         mv_pu = m.mvalue_pu
    #         # r = PerUnitConvert.from_pu(r)  # แปลงกลับจาก pu เป็นหน่วยจริงสำหรับการแสดงผล
    #     else:
    #         mid, mt, mv = k, "—", float("nan")

        if measurements is not None and k < len(measurements):
            m   = measurements[k]
            mid = m.id
            mt  = m.name or m.mtype or "unknown" # ป้องกัน NoneType
            
            mv      = m.mvalue
            mv_pu   = m.mvalue_pu
            
            # 2. ป้องกัน ZeroDivisionError 
            # ถ้าค่าไม่ใช่ 0 ให้หา Base จาก (ค่าจริง / ค่า PU) ตามไอเดียของคุณ
            if mv_pu is not None and mv_pu != 0:
                base_multiplier = mv / mv_pu
            else:
                # ถ้าค่าเป็น 0 ต้องยอมใช้ Default ป้องกัน Error
                base_multiplier = 380.0 if str(mt).lower().startswith('v') else 1000.0
        else:
            # 3. ป้องกัน NameError ให้ตัวแปรครบถ้วนในกรณีเข้า else
            mid, mt, mv = k, "—", float("nan")
            base_multiplier = 1.0

        r = r*base_multiplier  # แปลงกลับจาก pu เป็นหน่วยจริงสำหรับการแสดงผลเฉพาะ V เท่านั้น
        h = hx*base_multiplier  # แปลงกลับจาก pu เป็นหน่วยจริงสำหรับการแสดงผลเฉพาะ V เท่านั้น
        hval = mv - r  # h(x̂) = z - r
        print(f"  {k:>3}  {mid:>4}  {mt:>8}    "
              f"{mv:>10.5f}  {h:>10.5f} {hval:>10.5f}  {r:>10.5f}  {abs(nr):>7.4f}")
    print(f"\n hx is \n {result.x_est}\n")
    _sep()
    for b, v in bus_2idx.items():
        print(f"Bus {b} value {v}")
        
    for b, v in branch_2idx.items():
        print(f"Branch {b} value {v}")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _solve_linear(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b; fall back to least-squares if A is singular."""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x


def _sep(n: int = 64) -> None:
    print("  " + "─" * n)


def _banner(text: str, n: int = 64) -> None:
    print("\n  " + "═" * n)
    print(f"  {text}")
    print("  " + "═" * n)


class BadDataDetector:
    pass

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# (สมมติว่า Import ของคุณถูกต้องทั้งหมด)
from structural_data import Bus, Branch, Measurement, Generator, Load
from math_model.static_parameter import MeasurementMatrix, YBusMatrix
from math_model.dynamic_parameter import StateVector, Jacobian
from measurement.unit_converter import PerUnitConvert, PowerDirection, PrefixConverter
from estimator import Estimation, WLSSolver, SEResult, print_results # , Result

class System:
    def __init__(self, s_base: float = 1000.0, v_base: float = 380.0) -> None:
        self.s_base = s_base
        self.v_base = v_base
        
        # Use dictionaries to store components by their IDs for easy access and updates
        self.buses: Dict[int, Bus] = {}       
        self.branches: Dict[int, Branch] = {}
        self.loads: Dict[int, Load] = {}
        self.generators: Dict[int, Generator] = {} # เปลี่ยนชื่อเป็นพหูพจน์ (generators)
        self.measurements: Dict[int, Measurement] = {}

        self.ybus: Optional[YBusMatrix] = None
        self.sv: Optional[StateVector] = None
        self.mm: Optional[MeasurementMatrix] = None
        self.jac: Optional[Jacobian] = None
        self.se_result: Optional[SEResult] = None
        self.solver: Optional[Estimation] = None


    # ✅ แก้ไข: เปลี่ยน -> None เป็น -> 'System' เพื่อรองรับ Method Chaining
    def add_bus(self, id: int, name: Optional[str] = None, type: Optional[str] = None, 
                slack: Optional[bool] = None, nominal: Optional[float] = None, unit: Optional[str] = None) -> 'System':
        bus = Bus(id=id, name=name, type=type, slack=slack, nominal=nominal, unit=unit)
        self.buses[bus.id] = bus
        return self

    def add_branch(self, id: int, name: Optional[str] = None, fbus: int = None, tbus: int = None, 
                   rs: float = 0.0, xs: float = 0.0, xsh: float = 0.0) -> 'System':
        branch = Branch(id=id, name=name, fbus=fbus, tbus=tbus, rs=rs, xs=xs, xsh=xsh)
        self.branches[branch.id] = branch
        return self
    
    def add_load(self, id: int, name: Optional[str] = None, lbus: int = None, pl: Optional[float] = None, 
                 ql: Optional[float] = None, pl_pu: Optional[float] = None, ql_pu: Optional[float] = None, 
                 unit: Optional[str] = None) -> 'System':
        load = Load(id=id, name=name, lbus=lbus, pl=pl, ql=ql, pl_pu=pl_pu, ql_pu=ql_pu, unit=unit) 
        self.loads[load.id] = load
        return self
    
    def add_generator(self, id: int, name: Optional[str] = None, gbus: int = None, pg: Optional[float] = None, 
                      qg: Optional[float] = None, vgen: Optional[float] = None, pg_pu: Optional[float] = None, 
                      qg_pu: Optional[float] = None, vgen_pu: Optional[float] = None, unit: Optional[str] = None) -> 'System':
        generator = Generator(id=id, name=name, gbus=gbus, pg=pg, qg=qg, vgen=vgen, pg_pu=pg_pu, qg_pu=qg_pu, vgen_pu=vgen_pu, unit=unit)
        self.generators[generator.id] = generator
        return self

    def add_measurement(self, id: int, name: str = None, position: str = None, pos_id: int = None,
                        mtype: Optional[str] = None, mvalue: float = 0.0, msd: float = 1.0,
                        mvalue_pu: Optional[float] = None, msd_pu: Optional[float] = None) -> 'System':
        meas = Measurement(id=id, name=name, position=position, pos_id=pos_id, mtype=mtype, 
                           mvalue=mvalue, msd=msd, mvalue_pu=mvalue_pu, msd_pu=msd_pu)
        self.measurements[meas.id] = meas
        return self

    def get_ready_measurements(self) -> List[Measurement]:
        """เตรียมข้อมูล Measurement ทั้งหมดให้พร้อมก่อนเข้าสมการ"""
        # ดึงออกมาเป็น List เพื่อส่งเข้าฟังก์ชันอัปเดต (ถ้าฟังก์ชันพวกนี้อัปเดตแบบ In-place)
        meas_list = list(self.measurements.values())
        load_list = list(self.loads.values())
        gen_list = list(self.generators.values())

        # ✅ ทำ Data Pipeline ได้สวยมากครับ!
        PowerDirection.adjust_load_values(meas_list, load_list)
        PowerDirection.adjust_gen_values(meas_list, gen_list)
        PrefixConverter.to_base_batch(meas_list)
        PerUnitConvert.to_pu_batch(meas_list, self.v_base, self.s_base)
        
        return meas_list
    
    def build_system(self) -> None:
        # ส่งค่าเป็น List เข้าไปใน Math Model เพราะมันน่าจะต้องการ List
        ready_meas = self.get_ready_measurements()
        self.ybus = YBusMatrix(list(self.buses.values()), list(self.branches.values()))
        self.mm = MeasurementMatrix(ready_meas)
        self.sv = StateVector(self.ybus)
        self.jac = Jacobian(ready_meas, self.ybus, self.sv)
    
    def estimate(self) -> None:
        """รัน State Estimation"""
        # ใช้ list ของ measurement ที่เตรียมพร้อมแล้ว

        # ✅ แก้ไขที่ 1: ส่ง jac, sv, mm เข้าไปตอนสร้าง Solver ตามโครงสร้าง __init__
        self.solver = Estimation(self.jac, self.sv, self.mm, method="analytical", tol=1e-8)

        # ✅ แก้ไขที่ 2: ฟังก์ชัน solve() ปกติรับแค่ x0 (ค่าเริ่มต้น) และ verbose (เปิด/ปิด log)
        # ไม่ต้องส่ง mm หรือ H เข้าไปซ้ำ เพราะ Solver มันดึงจาก self.jac และ self.mm ได้เอง
        self.se_result: SEResult = self.solver.run()
        print_results(self.se_result, self.ybus, self.buses,self.branches, self.get_ready_measurements())

    def get_bus_results(self) -> Dict[int, Any]:
        """ดึงผลลัพธ์ของ Bus ทั้งหมดในรูปแบบ Dictionary"""
        if getattr(self, 'se_result', None) is None:
            raise ValueError("ต้องรัน grid.estimate() ก่อนดึงผลลัพธ์ครับ!")
        bus_results = {}
        bus_results = self.solver.get_estimate_var(self.v_base, self.s_base, self.jac)["buses"]  # เรียกใช้ Method result() ของ Result class เพื่อดึงผลลัพธ์ของ Bus
        return bus_results
    
    def get_branch_results(self) -> Dict[int, Any]:
        """ดึงผลลัพธ์ของ Branch ทั้งหมดในรูปแบบ Dictionary"""
        if getattr(self, 'se_result', None) is None:
            raise ValueError("ต้องรัน grid.estimate() ก่อนดึงผลลัพธ์ครับ!")
        branch_results = {}
        branch_results = self.solver.get_estimate_var(self.v_base, self.s_base, self.jac)["branches"]  # เรียกใช้ Method result() ของ Result class เพื่อดึงผลลัพธ์ของ Branch
        return branch_results

    def get_measurement_var(self) -> Dict[int, Any]:
        meas = self.measurements 
        print("\nMeasurements in System:", meas)
        meas_var = {}
        for mid, m in meas.items():
            pos_id = meas[mid].pos_id  
            meas_var[pos_id] = {}
        
        for mid, m in meas.items():
            pos_id = meas[mid].pos_id  
            if meas[mid].position == 'bus':
                # meas_var[mid]['position'] = meas[mid].position
                mtype = meas[mid].mtype
                print(f"\n{mtype}")
                if mtype.startswith('p'):
                    
                    meas_var[pos_id][f"{mtype[0:1]}{pos_id}"] = {'value': m.mvalue, 'value_pu': m.mvalue_pu}
                elif mtype.startswith('q'):
                    meas_var[pos_id][f"{mtype[0:1]}{pos_id}"] = {'value': m.mvalue, 'value_pu': m.mvalue_pu}
                elif mtype.startswith('v'):
                    
                    meas_var[pos_id][f"{mtype[0:1]}{pos_id}"] = {'value': m.mvalue, 'value_pu': m.mvalue_pu}

        #         mtype = meas[mid].mtype
        #         if mtype.startswith('p'):
        #             meas_var[mid][f"{mtype[0:1]}{pos_id}"]['value'] = m.mvalue 
        #             meas_var[mid][f"{mtype[0:1]}{pos_id}"]['value_pu'] = m.mvalue_pu
        #         elif mtype.startswith('q'):
        #             meas_var[mid][f"{mtype[0:1]}{pos_id}"]['value'] = m.mvalue 
        #             meas_var[mid][f"{mtype[0:1]}{pos_id}"]['value_pu'] = m.mvalue_pu
        #         elif mtype.startswith('v'):
        #             meas_var[mid][f"{mtype[0:1]}{pos_id}"]['value'] = m.mvalue 
        #             meas_var[mid][f"{mtype[0:1]}{pos_id}"]['value_pu'] = m.mvalue_pu
            # if meas[mid].position == 'branch':
            #     pos_id = meas[mid].pos_id
            #     for br_id, br in self.branches.items():
            #         if pos_id == br_id:
            #             meas_var[mid]['position'] = meas[mid].position
            #             mtype = meas[mid].mtype
            #             meas_var[mid][f"{mtype[0:1]}{pos_id}"] = {'value': m.mvalue, 'value_pu': m.mvalue_pu}
            #     mtype = meas[mid].mtype
            #     meas_var[mid][f"{mtype[0:1]}{pos_id}"] = {'value': m.mvalue, 'value_pu': m.mvalue_pu}
        return meas_var
    

    # def get_bus_results(self) -> dict:
    #     """ดึงผลลัพธ์ของ Bus สำหรับแสดงผล Dashboard"""
    #     # อิมพอร์ต ResultExtractor ที่เราเพิ่งสร้าง
    #     from estimator import ResultExtractor
        
    #     if getattr(self, 'se_result', None) is None:
    #         raise ValueError("ต้องรัน grid.estimate() ก่อนดึงผลลัพธ์ครับ!")
            
    #     # โยน se_result ตัวจริงที่ได้จาก Solver เข้าไป
    #     dashboard_data = ResultExtractor.get_dashboard_data(self.se_result, self.jac, self.v_base, self.s_base)
        
    #     return dashboard_data["buses"]
    
    # def get_branch_results(self) -> dict:
    #     """ดึงผลลัพธ์ของ Branch สำหรับแสดงผล Dashboard"""
    #     # อิมพอร์ต ResultExtractor ที่เราเพิ่งสร้าง
    #     from estimator import ResultExtractor
        
    #     if getattr(self, 'se_result', None) is None:
    #         raise ValueError("ต้องรัน grid.estimate() ก่อนดึงผลลัพธ์ครับ!")
            
    #     # โยน se_result ตัวจริงที่ได้จาก Solver เข้าไป
    #     dashboard_data = ResultExtractor.get_dashboard_data(self.se_result, self.jac, self.v_base, self.s_base)
        
    #     return dashboard_data["branches"]


# ... (ส่วน if __name__ == "__main__": ของคุณสามารถใช้งานได้เลย เพราะ Method Chaining จะทำงานได้สมบูรณ์แล้ว)
    

if __name__ == "__main__":
    from user_config import create_sample_3bus_config
    grid = System()

    grid.add_bus(id=1, name="Slack", type="slack", slack=True)
    grid.add_bus(id=2, name="Bus2",  type="PQ")
    grid.add_bus(id=3, name="Bus3",  type="PQ")
    grid.add_branch(id=1, fbus=1, tbus=2, rs=0.035, xs=0.25, xsh=0.00)
    grid.add_branch(id=2, fbus=1, tbus=3, rs=0.035, xs=0.25, xsh=0.00)
    grid.add_branch(id=3, fbus=2, tbus=3, rs=0.035, xs=0.25, xsh=0.00)
    grid.add_generator(id=1, name="Gen1", gbus=1, pg=0.2148315, qg=0.000171)
    grid.add_load(id=1, name="Load1", lbus=2)
    grid.add_load(id=2, name="Load2", lbus=3)


    # test
    grid.add_measurement(position = 'bus', name="p1", id=1,  pos_id=1, mvalue=217.84, msd=0.010)
    grid.add_measurement(position = 'bus', name="q1", id=2,  pos_id=1, mvalue=58.784, msd=0.010)
    grid.add_measurement(position = 'bus', name="p2", id=3,  pos_id=2, mvalue=138.18, msd=0.010)
    grid.add_measurement(position = 'bus', name="q2", id=4,  pos_id=2, mvalue=0.1359, msd=0.010)
    grid.add_measurement(position = 'bus', name="p3", id=5,  pos_id=3, mvalue=77.79, msd=0.010)
    grid.add_measurement(position = 'bus', name="q3", id=6,  pos_id=3, mvalue=51.2691, msd=0.010)
    grid.add_measurement(position = 'bus', name="v1", id=7,  pos_id=1, mvalue=368.54, msd=0.010)
    grid.add_measurement(position = 'bus', name="v2", id=8,  pos_id=2, mvalue=362.3988, msd=0.010)
    grid.add_measurement(position = 'bus', name="v3", id=9,  pos_id=3, mvalue=361.4268, msd=0.010)

    # config = create_sample_3bus_config()
    grid.build_system()    
    grid.get_ready_measurements()
    # print("\nReady measurements (after unit conversion):", grid.measurements)
    grid.estimate()
    grid.get_bus_results()
    grid.get_branch_results()
    print("\nBus Results:", grid.get_bus_results())
    print("\n\nBranch Results:", grid.get_branch_results())
