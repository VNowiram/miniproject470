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
    name: str = None  # 'P', 'Q', 'V', 'I', etc.
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
    mside: Optional[str] = None  # 'from' | 'to' — required for pflow/qflow measurements

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