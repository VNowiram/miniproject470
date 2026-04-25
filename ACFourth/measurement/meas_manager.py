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