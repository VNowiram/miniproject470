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
    mtype       : str          # was incorrectly typed as float
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
    x_est    : h(x̂) — estimated measurement vector (shape m)
    x_state  : x̂   — estimated state vector [θ_non-slack | V_all] (shape 2n-1)
    """
    x_est       : np.ndarray   # h(x̂) — measurement function at solution
    x_state     : np.ndarray   # x̂   — actual state vector (angles + voltages)
    converged   : bool
    iterations  : int
    theta_deg   : np.ndarray
    V_pu        : np.ndarray
    residuals   : np.ndarray
    norm_res    : np.ndarray
    J           : float
    method_used : str = "analytical"
    V           : Optional[np.ndarray] = None
    gain_matrix : np.ndarray = field(default_factory=lambda: np.array([]))
    


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
        import warnings

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
        obs_checked = False

        for it in range(1, self.max_iter + 1):
            hx    = self.jac.h(x)
            r     = z - hx
            H     = H_func(x)

            # ── Observability check (first iteration only) ──────────────────
            if not obs_checked:
                obs_checked = True
                rank = int(np.linalg.matrix_rank(H))
                if rank < self.sv.n_state:
                    warnings.warn(
                        f"Observability warning: rank(H)={rank} < n_state={self.sv.n_state}. "
                        "System may not be fully observable — results may be unreliable.",
                        RuntimeWarning, stacklevel=2,
                    )

            G_wls = H.T @ W @ H
            rhs   = H.T @ W @ r
            dx    = _solve_linear(G_wls, rhs)

            # ── Armijo backtracking line search ─────────────────────────────
            J_curr = float(r @ W @ r)
            alpha  = 1.0
            for _ in range(10):                        # max 10 halvings → α_min ≈ 0.001
                x_try = x + alpha * dx
                r_try = z - self.jac.h(x_try)
                if float(r_try @ W @ r_try) < J_curr:
                    break
                alpha *= 0.5

            x += alpha * dx
            max_dx      = float(np.max(np.abs(alpha * dx)))
            J_iter      = float(r @ W @ r)
            G_wls_final = G_wls

            if verbose:
                print(f"  {it:>4}  {max_dx:>14.6e}  {J_iter:>14.4f}  α={alpha:.4f}")

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
            x_est       = hx_final,     # h(x̂) — estimated measurements
            x_state     = x.copy(),     # x̂    — actual state vector
            converged   = converged,
            iterations  = it,
            theta_deg   = np.degrees(theta),
            V_pu        = V,
            residuals   = residuals,
            norm_res    = norm_res,
            J           = J_final,
            method_used = self.method,
            gain_matrix = G_wls_final,
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._var_cache: Optional[dict] = None   # cleared on each run()

    def run(self, verbose: bool = True) -> SEResult:
        self._var_cache = None          # invalidate cache before new solve
        result = self.solve(verbose=verbose)
        return result

    def get_estimate_var(self, v_base: float, s_base: float, jac: Jacobian) -> dict:
        # Return cached result if available (avoids double-compute when
        # get_bus_results() and get_branch_results() are called separately)
        if self._var_cache is not None:
            return self._var_cache
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
        self._var_cache = {"buses": bus_result, "branches": branch_result}
        return self._var_cache
    


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
    """
    Two complementary bad-data tests.

    chi2_test  — global test: reject H0 (no bad data) when J(x̂) > χ²_{m-n, 1-α}
    lnr        — local test:  flag the measurement with the largest |rᵢ / σᵢ|
    """

    @staticmethod
    def chi2_test(
        result   : SEResult,
        n_state  : int,
        alpha    : float = 0.05,
    ) -> dict:
        """
        Parameters
        ----------
        result  : SEResult from WLSSolver
        n_state : number of state variables (sv.n_state)
        alpha   : significance level (default 5 %)

        Returns
        -------
        dict with keys: passed, J, threshold, dof, alpha, suspicious_indices
        """
        try:
            from scipy.stats import chi2
        except ImportError:
            return {
                "passed":  None,
                "message": "scipy not installed — install it for the chi-squared test",
            }

        m   = len(result.norm_res)
        dof = m - n_state
        if dof <= 0:
            return {
                "passed":  True,
                "message": f"No degrees of freedom for test (m={m}, n_state={n_state}). "
                           "Add more measurements.",
            }

        threshold  = float(chi2.ppf(1.0 - alpha, dof))
        passed     = result.J <= threshold
        suspicious = [int(i) for i in np.where(np.abs(result.norm_res) > 3.0)[0]]

        return {
            "passed":             passed,
            "J":                  result.J,
            "threshold":          threshold,
            "dof":                dof,
            "alpha":              alpha,
            "suspicious_indices": suspicious,   # |r/σ| > 3σ
        }

    @staticmethod
    def lnr(result: SEResult, threshold: float = 3.0) -> dict:
        """
        Largest Normalized Residual method.

        Returns the index and value of the worst measurement.
        If its |rᵢ / σᵢ| exceeds `threshold` it is suspected bad data.
        """
        abs_nr = np.abs(result.norm_res)
        idx    = int(np.argmax(abs_nr))
        value  = float(abs_nr[idx])
        return {
            "index":     idx,
            "value":     value,
            "is_bad":    value > threshold,
            "threshold": threshold,
        }

    @staticmethod
    def report(result: SEResult, n_state: int, alpha: float = 0.05) -> None:
        """Print a formatted bad-data summary."""
        _banner("BAD DATA DETECTION")
        chi = BadDataDetector.chi2_test(result, n_state, alpha)
        lnr = BadDataDetector.lnr(result)

        if chi.get("passed") is None:
            print(f"  Chi-squared test: {chi['message']}")
        else:
            status = "PASS ✓" if chi["passed"] else "FAIL ✗  ← bad data likely present"
            print(f"  Chi-squared test  : {status}")
            if "J" in chi:
                print(f"    J(x̂)      = {chi['J']:.4f}")
                print(f"    χ²threshold = {chi['threshold']:.4f}  (dof={chi['dof']}, α={chi['alpha']})")
            if chi.get("suspicious_indices"):
                print(f"    |r/σ| > 3σ at measurement indices: {chi['suspicious_indices']}")

        _sep()
        flag = "BAD ✗" if lnr["is_bad"] else "OK  ✓"
        print(f"  LNR test          : {flag}")
        print(f"    Largest |r/σ|  = {lnr['value']:.4f}  at index {lnr['index']}")
        print(f"    Threshold      = {lnr['threshold']:.1f} σ")
        _sep()

# ══════════════════════════════════════════════════════════════════════════════
# OBSERVABILITY
# ══════════════════════════════════════════════════════════════════════════════

class Observability:
    """
    Network observability analysis for WLS AC State Estimation.

    Theory
    ------
    A system is **numerically observable** iff  rank(H) == n_state.

    Sensitivity / hat matrix
        S = H G⁻¹ Hᵀ W     where G = HᵀWH
        Diagonal entry s_kk ∈ [0, 1]:
          s_kk → 1 : critical — removing this measurement breaks observability.
          s_kk → 0 : fully redundant — state well covered by other measurements.

    Critical measurements
        s_kk ≥ crit_thresh (default 0.99).

    Unobservable modes  (rank-deficient case only)
        Null-space vectors of H reveal which *linear combinations* of state
        variables the current measurement set cannot resolve.

    Usage
    -----
        x0  = sv.flat_start()
        H   = jac.H_analytical(x0)

        # Full printed report:
        Observability.report(H, mm, sv, ready_measurements)

        # Just the data dict:
        obs = Observability.analyse(H, mm, sv)
    """

    _RANK_TOL    : float = 1e-8   # σ / σ_max below which → treated as zero
    _CRIT_THRESH : float = 0.99   # s_kk above which → critical measurement

    # ── Main entry points ────────────────────────────────────────────────────

    @staticmethod
    def analyse(
        H           : np.ndarray,
        mm          : MeasurementMatrix,
        sv          : StateVector,
        rank_tol    : float = 1e-8,
        crit_thresh : float = 0.99,
    ) -> dict:
        """
        Full observability analysis.

        Returns
        -------
        dict
            observable         bool
            rank               int
            n_state            int
            m                  int          number of measurements
            redundancy         int          m − rank(H)
            redundancy_ratio   float        m / n_state
            singular_values    ndarray      shape (min(m,n),)  descending
            condition_number   float        σ_max / σ_min  (inf when unobservable)
            sensitivity_diag   ndarray      shape (m,)  diagonal of hat matrix S
            critical_indices   list[int]    measurement indices with s_kk ≥ crit_thresh
            unobservable_modes list[ndarray] null-space vectors (empty when observable)
        """
        m, n = H.shape
        W    = mm.W

        # ── Rank via SVD ─────────────────────────────────────────────────
        sv_vals = np.linalg.svd(H, compute_uv=False)          # descending
        sv_max  = sv_vals[0] if sv_vals[0] > 0 else 1.0
        rank    = int(np.sum(sv_vals > rank_tol * sv_max))
        sv_min  = sv_vals[-1]
        cond    = sv_max / sv_min if sv_min > 0 else float("inf")

        observable = rank >= sv.n_state

        # ── Sensitivity (hat-matrix) diagonal ────────────────────────────
        sens = Observability._sensitivity_diag(H, W)

        # ── Critical measurements ─────────────────────────────────────────
        critical = [int(k) for k, s in enumerate(sens) if s >= crit_thresh]

        # ── Null-space modes (rank-deficient only) ────────────────────────
        modes: List[np.ndarray] = []
        if not observable:
            modes = Observability._null_space(H, rank_tol)

        return {
            "observable":         observable,
            "rank":               rank,
            "n_state":            sv.n_state,
            "m":                  m,
            "redundancy":         m - rank,
            "redundancy_ratio":   round(m / sv.n_state, 4) if sv.n_state else 0.0,
            "singular_values":    sv_vals,
            "condition_number":   cond,
            "sensitivity_diag":   sens,
            "critical_indices":   critical,
            "unobservable_modes": modes,
        }

    @staticmethod
    def is_observable(H: np.ndarray, n_state: int, tol: float = 1e-8) -> bool:
        """Quick single-call rank check.  True iff rank(H) >= n_state."""
        sv_vals = np.linalg.svd(H, compute_uv=False)
        sv_max  = sv_vals[0] if sv_vals[0] > 0 else 1.0
        return int(np.sum(sv_vals > tol * sv_max)) >= n_state

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _sensitivity_diag(H: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Diagonal of  S = H G⁻¹ Hᵀ W,  G = HᵀWH.

        Computed in numerically stable weighted form:
            H_w  = diag(w)^(1/2) H
            s_kk = H_w[k,:] · (G_w⁻¹ · H_w[k,:]ᵀ)
        where G_w = H_w' H_w = HᵀWH.
        """
        w   = np.diag(W)                        # (m,)
        H_w = np.sqrt(w)[:, None] * H            # (m, n)
        G_w = H_w.T @ H_w                        # (n, n)

        try:
            P = np.linalg.solve(G_w, H_w.T)     # (n, m)
        except np.linalg.LinAlgError:
            P, _, _, _ = np.linalg.lstsq(G_w, H_w.T, rcond=None)

        # s_kk = row_k of H_w  dot  col_k of P
        return np.clip(np.sum(H_w * P.T, axis=1), 0.0, 1.0)   # (m,)

    @staticmethod
    def _null_space(H: np.ndarray, tol: float = 1e-8) -> List[np.ndarray]:
        """
        Right singular vectors of H whose σ ≤ tol·σ_max.
        These span the unobservable state subspace.
        """
        _, sv_vals, Vt = np.linalg.svd(H, full_matrices=False)
        threshold      = tol * (sv_vals[0] if sv_vals[0] > 0 else 1.0)
        return [Vt[i] for i, s in enumerate(sv_vals) if s <= threshold]

    @staticmethod
    def _state_labels(sv: StateVector) -> List[str]:
        """
        Human-readable label for every column of H.
        Order: [θ_non-slack … | |V|_all …]  — matches StateVector layout.
        """
        labels: List[str] = []
        for s_idx in sorted(sv._th_s2b.keys()):
            bus_idx = sv._th_s2b[s_idx]
            labels.append(f"θ_bus{sv.buses[bus_idx].id}")
        for s_idx in sorted(sv._v_s2b.keys()):
            bus_idx = sv._v_s2b[s_idx]
            labels.append(f"|V|_bus{sv.buses[bus_idx].id}")
        return labels

    # ── Formatted report ──────────────────────────────────────────────────────

    @staticmethod
    def report(
        H            : np.ndarray,
        mm           : MeasurementMatrix,
        sv           : StateVector,
        measurements : Optional[List[Measurement]] = None,
        rank_tol     : float = 1e-8,
        crit_thresh  : float = 0.99,
    ) -> None:
        """
        Print a complete formatted observability report.

        Parameters
        ----------
        H            : Jacobian matrix (m × n_state) at any x (flat-start fine)
        mm           : MeasurementMatrix — provides weight matrix W
        sv           : StateVector      — provides n_state and state ordering
        measurements : optional list of Measurement objects for name labels
        rank_tol     : singular-value threshold for numerical rank
        crit_thresh  : s_kk threshold to declare a measurement critical
        """
        res    = Observability.analyse(H, mm, sv, rank_tol, crit_thresh)
        labels = Observability._state_labels(sv)

        _banner("OBSERVABILITY ANALYSIS")

        obs_str = "OBSERVABLE ✓" if res["observable"] else "NOT OBSERVABLE ✗"
        print(f"  Status              : {obs_str}")
        print(f"  Measurements  (m)   : {res['m']}")
        print(f"  State variables (n) : {res['n_state']}")
        print(f"  rank(H)             : {res['rank']}")
        print(f"  Redundancy (m − n)  : {res['redundancy']}"
              f"  (ratio = {res['redundancy_ratio']:.2f})")

        # ── Numerical conditioning ────────────────────────────────────────
        _sep()
        sv_vals = res["singular_values"]
        cond    = res["condition_number"]
        health  = ("good"     if cond < 1e6  else
                   "marginal" if cond < 1e10 else
                   "poor — verify branch parameters and measurement units")
        print(f"  Conditioning")
        print(f"    σ_max   = {sv_vals[0]:>14.6e}")
        print(f"    σ_min   = {sv_vals[-1]:>14.6e}")
        print(f"    cond(H) = {cond:>14.4e}  [{health}]")

        # ── Sensitivity table ─────────────────────────────────────────────
        _sep()
        print(f"  Measurement sensitivity  (s_kk  diagonal of hat matrix S)")
        _sep()
        print(f"  {'k':>3}  {'name':>10}  {'type':>10}  {'s_kk':>8}  coverage")
        _sep()
        for k, s_kk in enumerate(res["sensitivity_diag"]):
            mname = (measurements[k].name  or f"m{k}") if measurements and k < len(measurements) else f"m{k}"
            mtype = (measurements[k].mtype or "—")      if measurements and k < len(measurements) else "—"

            if s_kk >= crit_thresh:
                coverage = "⚠  CRITICAL — no backup"
            elif s_kk >= 0.70:
                coverage = "low redundancy"
            elif s_kk >= 0.30:
                coverage = "moderate"
            else:
                coverage = "✓  well redundant"

            print(f"  {k:>3}  {mname:>10}  {mtype:>10}  {s_kk:>8.4f}  {coverage}")

        # ── Critical summary ──────────────────────────────────────────────
        _sep()
        if res["critical_indices"]:
            crit_names = [
                (measurements[i].name or f"m{i}")
                if measurements and i < len(measurements) else f"m{i}"
                for i in res["critical_indices"]
            ]
            print(f"  ⚠  {len(crit_names)} critical measurement(s):  {crit_names}")
            print(f"     Losing any one makes the system unobservable.")
        else:
            print(f"  ✓  No critical measurements — all state directions are multiply covered.")

        # ── Unobservable modes ────────────────────────────────────────────
        if not res["observable"]:
            _sep()
            print(f"  Unobservable modes ({len(res['unobservable_modes'])} found)")
            print(f"  State directions that cannot be resolved by current measurements:")
            for i, mode in enumerate(res["unobservable_modes"]):
                top3   = np.argsort(np.abs(mode))[::-1][:3]
                detail = ",  ".join(
                    f"{labels[j] if j < len(labels) else f'state[{j}]'}: {mode[j]:+.3f}"
                    for j in top3
                )
                dominant = labels[top3[0]] if top3[0] < len(labels) else f"state[{top3[0]}]"
                print(f"    Mode {i}:  dominant → {dominant}")
                print(f"            components: [{detail}]")
            print()
            print(f"  → Add measurements that inject information into the dominant")
            print(f"    state of each unobservable mode.")
        _sep()