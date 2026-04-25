from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# (สมมติว่า Import ของคุณถูกต้องทั้งหมด)
from structural_data import Bus, Branch, Measurement, Generator, Load
from math_model.static_parameter import MeasurementMatrix, YBusMatrix
from math_model.dynamic_parameter import StateVector, Jacobian
from measurement.unit_converter import PerUnitConvert, PowerDirection, PrefixConverter
from estimator import Estimation, WLSSolver, SEResult, print_results, BadDataDetector, Observability

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

        # ── Internal state flags ─────────────────────────────────────────────
        # Prevents get_ready_measurements() from applying unit conversions twice
        # when measurements haven't changed (e.g. duplicate calls from build_system
        # and the main loop).  Reset whenever add_measurement() is called.
        self._meas_converted: bool = False


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
        self._meas_converted = False   # new measurement data → must re-convert
        return self

    def get_ready_measurements(self) -> List[Measurement]:
        """
        Prepare all measurements for the estimator (sign convention → prefix
        conversion → per-unit).  Safe to call multiple times — conversions are
        applied only once per set of measurements; subsequent calls return the
        already-converted objects immediately.
        """
        if self._meas_converted:
            return list(self.measurements.values())

        meas_list = list(self.measurements.values())
        load_list = list(self.loads.values())
        gen_list  = list(self.generators.values())

        PowerDirection.adjust_load_values(meas_list, load_list)
        PowerDirection.adjust_gen_values(meas_list, gen_list)
        PrefixConverter.to_base_batch(meas_list)
        PerUnitConvert.to_pu_batch(meas_list, self.v_base, self.s_base)

        self._meas_converted = True
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

    def check_bad_data(self, alpha: float = 0.05) -> None:
        """Run chi-squared + LNR bad-data tests and print a formatted report."""
        if self.se_result is None:
            raise ValueError("Run grid.estimate() before checking for bad data.")
        BadDataDetector.report(self.se_result, self.sv.n_state, alpha=alpha)

    def check_observability(
        self,
        rank_tol    : float = 1e-8,
        crit_thresh : float = 0.99,
    ) -> dict:
        """
        Run a full observability analysis on the current Jacobian and print a
        formatted report.  Safe to call before or after estimate().

        Parameters
        ----------
        rank_tol    : singular-value ratio below which a value is treated as zero
        crit_thresh : s_kk threshold for declaring a measurement critical (0–1)

        Returns
        -------
        dict from Observability.analyse() — keyed by 'observable', 'rank',
        'critical_indices', 'sensitivity_diag', 'unobservable_modes', etc.

        Raises
        ------
        ValueError if build_system() has not been called yet.
        """
        if self.jac is None or self.mm is None or self.sv is None:
            raise ValueError("Call grid.build_system() before check_observability().")

        # Evaluate H at flat start (angles = 0, |V| = 1 p.u.)
        x0 = self.sv.flat_start()
        H  = self.jac.H_analytical(x0)

        ready_meas = self.get_ready_measurements()
        Observability.report(H, self.mm, self.sv, ready_meas, rank_tol, crit_thresh)

        return Observability.analyse(H, self.mm, self.sv, rank_tol, crit_thresh)

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

    grid.add_measurement(position='bus', name="p1", id=1, pos_id=1, mvalue=217.84,   msd=0.010)
    grid.add_measurement(position='bus', name="q1", id=2, pos_id=1, mvalue=58.784,   msd=0.010)
    grid.add_measurement(position='bus', name="p2", id=3, pos_id=2, mvalue=138.18,   msd=0.010)
    grid.add_measurement(position='bus', name="q2", id=4, pos_id=2, mvalue=0.1359,   msd=0.010)
    grid.add_measurement(position='bus', name="p3", id=5, pos_id=3, mvalue=77.79,    msd=0.010)
    grid.add_measurement(position='bus', name="q3", id=6, pos_id=3, mvalue=51.2691,  msd=0.010)
    grid.add_measurement(position='bus', name="v1", id=7, pos_id=1, mvalue=368.54,   msd=0.010)
    grid.add_measurement(position='bus', name="v2", id=8, pos_id=2, mvalue=362.3988, msd=0.010)
    grid.add_measurement(position='bus', name="v3", id=9, pos_id=3, mvalue=361.4268, msd=0.010)

    grid.build_system()   # calls get_ready_measurements() internally
    grid.check_observability()  # run BEFORE estimate — uses flat-start H
    grid.estimate()
    grid.check_bad_data()

    print("\nBus Results:",    grid.get_bus_results())
    print("\nBranch Results:", grid.get_branch_results())