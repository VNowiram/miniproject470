# user_config.py

from dataclasses import dataclass, asdict, field
from typing import List

from structural_data import Bus, Branch, Load, Generator, Measurement

@dataclass
class UserConfig:
    """
    User configuration for the WLS state estimation.
    Holds the system data in a structured format.
    """
    # ใช้ default_factory เพื่อป้องกันปัญหา Mutable Default Argument ใน Python
    buses: List[Bus] = field(default_factory=list)
    branches: List[Branch] = field(default_factory=list)
    generators: List[Generator] = field(default_factory=list)
    loads: List[Load] = field(default_factory=list)
    measurements: List[Measurement] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        Convert the user configuration to a dictionary format for easier processing.
        """
        # ข้อดีของ dataclass คือใช้ asdict() คลุมตัวเองได้เลย ไม่ต้องเขียน Loop เอง
        return asdict(self)




def create_sample_3bus_config() -> UserConfig:
    """สร้างข้อมูลจำลองระบบ 3-Bus สำหรับการทดสอบ"""
    return UserConfig(
        buses = [
            Bus(id=1, name="Slack", type="slack", slack=True),
            Bus(id=2, name="Bus2",  type="PQ"),
            Bus(id=3, name="Bus3",  type="PQ"),
            ],
        branches = [
            Branch(id=1, fbus=1, tbus=2, rs=0.035, xs=0.25, xsh=0.00),
            Branch(id=2, fbus=1, tbus=3, rs=0.035, xs=0.25, xsh=0.00),
            Branch(id=3, fbus=2, tbus=3, rs=0.035, xs=0.25, xsh=0.00),
        ],
        generators = [
            Generator(id=1, name="Gen1", gbus=1, pg=0.2148315, qg=0.000171),
        ],
        loads = [
            Load(id=1, name="Load1", lbus=2),
            Load(id=2, name="Load2", lbus=3),
        ],

        # test
        measurements = [
            Measurement(position = 'bus',  name="p1", id=1,  pos_id=1, mvalue=214.8315, msd=0.010),
            Measurement(position = 'bus', name="q1", id=2,  pos_id=1, mvalue=0.171, msd=0.010),
            Measurement(position = 'bus', name="p2", id=3,  pos_id=2, mvalue=141.229, msd=0.010),
            Measurement(position = 'bus', name="q2", id=4,  pos_id=2, mvalue=0.171, msd=0.010),
            Measurement(position = 'bus', name="p3", id=5,  pos_id=3, mvalue=72.11, msd=0.010),
            Measurement(position = 'bus', name="q3", id=6,  pos_id=3, mvalue=0.0018, msd=0.010),
            Measurement(position = 'bus', name="v1",    id=7,  pos_id=1, mvalue=372, msd=0.004),
            Measurement(position = 'bus', name="v2",    id=8, pos_id=2, mvalue=368, msd=0.004),
            Measurement(position = 'bus', name="v3",    id=9, pos_id=3, mvalue=366, msd=0.004),
        ]
    )


if __name__ == "__main__":
    from user_config import UserConfig
    from structural_data import Bus, Branch

    # 1. สร้าง Object ข้อมูลตามปกติ
    my_config = UserConfig(
        buses=[Bus(id=1, name="Slack", type="slack")],
        branches=[Branch(id=1, fbus=1, tbus=2, rs=0.035, xs=0.25, xsh=0.0)],
        # ... ข้อมูลอื่นๆ
    )

    # 2. แปลงเป็น Dictionary เตรียมพร้อมใช้งาน
    config_dict = my_config.to_dict()

    # ลอง Print ดูหน้าตาของ Dictionary
    print(config_dict) 
    # ผลลัพธ์: {'buses': [{'id': 1, 'name': 'Slack', 'type': 'slack', ...}], 'branches': [...], ...}