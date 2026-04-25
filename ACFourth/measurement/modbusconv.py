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