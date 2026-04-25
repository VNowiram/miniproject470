# math_model/dynamic_parameter.py

# new module for static parameters
from structural_data import  Bus, Branch, Measurement
from .static_parameter import YBusMatrix, MeasurementMatrix

# Build-in imports for dynamic parameters
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

#open-source imports for dynamic parameters
import numpy as np


class StateVector:
    """
    AC state vector x ∈ R^(2n-1):

        x = [ θ_{non-slack buses}  (n-1 entries, ascending bus_id)
              V_{all buses}        (n   entries, ascending bus_id) ]

    The slack-bus angle is fixed at 0 rad and excluded from x.
    """

    def __init__(self, ybus: YBusMatrix) -> None:
        self.ybus = ybus
        self.buses = ybus.buses

        slacks = [b for b in self.buses if b.is_slack]
        if not slacks:
            raise ValueError(
                "No slack bus found. Set slack=True or btype='slack' on one bus."
            )

        self.slack_bus: Bus = slacks[0]
        self.slack_bus_idx: int = ybus.bus_index(self.slack_bus.id)

        self._non_slack: List[Bus] = [b for b in self.buses if not b.is_slack]

        self.n_theta: int = len(self._non_slack)
        self.n_v: int = ybus.n
        self.n_state: int = self.n_theta + self.n_v

        self._th_s2b: Dict[int, int] = {
            s: ybus.bus_index(b.id)
            for s, b in enumerate(self._non_slack)
        }

        self._v_s2b: Dict[int, int] = {
            (self.n_theta + s): ybus.bus_index(b.id)
            for s, b in enumerate(self.buses)
        }

        self.bus_idx_to_th: Dict[int, Optional[int]] = {
            bus_idx: s for s, bus_idx in self._th_s2b.items()
        }
        self.bus_idx_to_th[self.slack_bus_idx] = None

        self.bus_idx_to_v: Dict[int, int] = {
            bus_idx: s for s, bus_idx in self._v_s2b.items()
        }

    def flat_start(self) -> np.ndarray:
        """Initial state vector: all θ = 0 rad, all V = 1.0 pu."""
        x = np.zeros(self.n_state)
        for s in self._v_s2b:
            x[s] = 1.0
        return x

    def unpack(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        theta : (n,) – angles in radians (slack bus entry = 0)
        V     : (n,) – voltage magnitudes in pu
        """
        theta = np.zeros(self.ybus.n)
        V = np.zeros(self.ybus.n)

        for s, bus_idx in self._th_s2b.items():
            theta[bus_idx] = x[s]
        for s, bus_idx in self._v_s2b.items():
            V[bus_idx] = x[s]

        return theta, V

    def pack(self, theta: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Assemble x from full bus arrays (slack angle is ignored)."""
        x = np.zeros(self.n_state)
        for s, bus_idx in self._th_s2b.items():
            x[s] = theta[bus_idx]
        for s, bus_idx in self._v_s2b.items():
            x[s] = V[bus_idx]
        return x

    def __repr__(self) -> str:
        th = ", ".join(f"θ_bus{b.id}" for b in self._non_slack)
        vv = ", ".join(f"V_bus{b.id}" for b in self.buses)
        return (
            f"StateVector(n_state={self.n_state})\n"
            f"  angles   ({self.n_theta}): {th}\n"
            f"  voltages ({self.n_v}):   {vv}\n"
            f"  slack: bus {self.slack_bus.id} (θ fixed at 0)"
        )



class MeasurementFunc:
    """
    Nonlinear measurement function h(x).

    Supported measurement types
    ---------------------------
    'pinject'  – active power injection      P_i
    'qinject'  – reactive power injection    Q_i
    'pflow'    – active power flow           P_ij
    'qflow'    – reactive power flow         Q_ij
    'vmag'     – voltage magnitude           |V_i|
    """

    def __init__(
        self,
        measurements: List[Measurement],
        ybus: YBusMatrix,
        sv: StateVector,
    ) -> None:
        self.measurements = measurements
        self.ybus = ybus
        self.sv = sv
        self.m = len(measurements)

    # ─────────────────────────────────────────────────────────────────────────
    # h(x)
    # ─────────────────────────────────────────────────────────────────────────

    def h(self, x: np.ndarray) -> np.ndarray:
        """Compute estimated measurement vector h(x) ∈ ℝ^m."""
        theta, V = self.sv.unpack(x)
        hx = np.empty(self.m)

        for k, meas in enumerate(self.measurements):
            mt = meas.mtype.lower()

            if mt == "pinject":
                hx[k] = self._Pi(self.ybus.bus_index(meas.pos_id), theta, V)

            elif mt == "qinject":
                hx[k] = self._Qi(self.ybus.bus_index(meas.pos_id), theta, V)

            elif mt == "pflow":
                hx[k] = self._Pij(meas.pos_id, meas.mside, theta, V)

            elif mt == "qflow":
                hx[k] = self._Qij(meas.pos_id, meas.mside, theta, V)

            elif mt == "vmag":
                hx[k] = V[self.ybus.bus_index(meas.pos_id)]

            else:
                raise ValueError(
                    f"Unknown mtype='{meas.mtype}' (measurement mid={meas.mid})"
                )

        return hx

    # ─────────────────────────────────────────────────────────────────────────
    # Power injection / flow primitives
    # ─────────────────────────────────────────────────────────────────────────

    # def _Pi(self, i: int, theta: np.ndarray, V: np.ndarray) -> float:
    #     """P_i = V_i Σ_j V_j (G_ij cos θ_ij + B_ij sin θ_ij)"""
    #     G, B = self.ybus.G, self.ybus.B
    #     return V[i] * float(sum(
    #         V[j] * (G[i, j] * np.cos(theta[i] - theta[j])
    #               + B[i, j] * np.sin(theta[i] - theta[j]))
    #         for j in range(self.ybus.n)
    #     ))

    # def _Qi(self, i: int, theta: np.ndarray, V: np.ndarray) -> float:
    #     """Q_i = V_i Σ_j V_j (G_ij sin θ_ij − B_ij cos θ_ij)"""
    #     G, B = self.ybus.G, self.ybus.B
    #     return V[i] * float(sum(
    #         V[j] * (G[i, j] * np.sin(theta[i] - theta[j])
    #               - B[i, j] * np.cos(theta[i] - theta[j]))
    #         for j in range(self.ybus.n)
    #     ))

    def _Pi(self, i: int, theta: np.ndarray, V: np.ndarray) -> float:
        """P_i = V_i Σ_j V_j (G_ij cos θ_ij + B_ij sin θ_ij)  — vectorised"""
        G, B = self.ybus.G, self.ybus.B
        dth = theta[i] - theta                                    # shape (n,)
        return float(V[i] * np.dot(V, G[i] * np.cos(dth) + B[i] * np.sin(dth)))

    def _Qi(self, i: int, theta: np.ndarray, V: np.ndarray) -> float:
        """Q_i = V_i Σ_j V_j (G_ij sin θ_ij − B_ij cos θ_ij)  — vectorised"""
        G, B = self.ybus.G, self.ybus.B
        dth = theta[i] - theta
        return float(V[i] * np.dot(V, G[i] * np.sin(dth) - B[i] * np.cos(dth)))


    def _Pij(
        self, branch_id: int, side: str, theta: np.ndarray, V: np.ndarray
    ) -> float:
        """P_ij = V_i² g_s − V_i V_j (g_s cos θ_ij + b_s sin θ_ij)"""
        g_s, b_s, _ = self.ybus.branch_params(branch_id)
        i, j = self.ybus.branch_buses(branch_id)

        if side == "to":
            i, j = j, i

        dth = theta[i] - theta[j]
        return V[i] ** 2 * g_s - V[i] * V[j] * (g_s * np.cos(dth) + b_s * np.sin(dth))

    def _Qij(
        self, branch_id: int, side: str, theta: np.ndarray, V: np.ndarray
    ) -> float:
        """Q_ij = −V_i²(b_s + b_sh2) − V_i V_j (g_s sin θ_ij − b_s cos θ_ij)"""
        g_s, b_s, b_sh2 = self.ybus.branch_params(branch_id)
        i, j = self.ybus.branch_buses(branch_id)

        if side == "to":
            i, j = j, i

        dth = theta[i] - theta[j]
        return (
            -V[i] ** 2 * (b_s + b_sh2)
            - V[i] * V[j] * (g_s * np.sin(dth) - b_s * np.cos(dth))
        )

    def __repr__(self) -> str:
        types = [m.mtype for m in self.measurements]
        return (
            f"MeasurementFunc(m={self.m})\n"
            f"  types : {types}"
        )


class Jacobian(MeasurementFunc):
    """
    Jacobian builder for measurement function h(x).

    Inherits MeasurementFunc so it can reuse:
      - h(x)
      - _Pi, _Qi, _Pij, _Qij
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Numerical Jacobian
    # ─────────────────────────────────────────────────────────────────────────

    def H_numerical(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        H[k, s] = ( h_k(x + ε·e_s) − h_k(x − ε·e_s) ) / (2ε)
        Shape : (m, n_state)
        """
        H = np.zeros((self.m, self.sv.n_state))

        for s in range(self.sv.n_state):
            xp, xm = x.copy(), x.copy()
            xp[s] += eps
            xm[s] -= eps
            H[:, s] = (self.h(xp) - self.h(xm)) / (2.0 * eps)

        return H

    # ─────────────────────────────────────────────────────────────────────────
    # Analytical Jacobian
    # ─────────────────────────────────────────────────────────────────────────

    def H_analytical(self, x: np.ndarray) -> np.ndarray:
        """
        Closed-form Jacobian ∂h/∂x.
        Shape : (m, n_state)
        """
        theta, V = self.sv.unpack(x)
        H = np.zeros((self.m, self.sv.n_state))

        for k, meas in enumerate(self.measurements):
            mt = meas.mtype.lower()

            if mt == "pinject":
                i = self.ybus.bus_index(meas.pos_id)
                self._dPi_dT(k, i, theta, V, H)
                self._dPi_dV(k, i, theta, V, H)

            elif mt == "qinject":
                i = self.ybus.bus_index(meas.pos_id)
                self._dQi_dT(k, i, theta, V, H)
                self._dQi_dV(k, i, theta, V, H)

            elif mt == "pflow":
                self._dPij_dT(k, meas.pos_id, meas.mside, theta, V, H)
                self._dPij_dV(k, meas.pos_id, meas.mside, theta, V, H)

            elif mt == "qflow":
                self._dQij_dT(k, meas.pos_id, meas.mside, theta, V, H)
                self._dQij_dV(k, meas.pos_id, meas.mside, theta, V, H)

            elif mt == "vmag":
                i = self.ybus.bus_index(meas.pos_id)
                H[k, self.sv.bus_idx_to_v[i]] = 1.0

            else:
                raise ValueError(
                    f"Unknown mtype='{meas.mtype}' (measurement mid={meas.mid})"
                )

        return H

    # ─────────────────────────────────────────────────────────────────────────
    # Analytical partial derivatives – θ (angle) derivatives
    # ─────────────────────────────────────────────────────────────────────────

    def _dPi_dT(
        self, k: int, i: int, theta: np.ndarray, V: np.ndarray, H: np.ndarray
    ) -> None:
        """
        ∂Pᵢ/∂θᵢ  = -Qᵢ - Vᵢ² Bᵢᵢ
        ∂Pᵢ/∂θⱼ  =  VᵢVⱼ (Gᵢⱼ sin θᵢⱼ - Bᵢⱼ cos θᵢⱼ)
        """
        G, B, Vi = self.ybus.G, self.ybus.B, V[i]
        Qi = self._Qi(i, theta, V)

        s_th = self.sv.bus_idx_to_th.get(i)
        if s_th is not None:
            H[k, s_th] += -Qi - Vi ** 2 * B[i, i]

        for j in range(self.ybus.n):
            if j == i:
                continue
            s_th = self.sv.bus_idx_to_th.get(j)
            if s_th is not None:
                dth = theta[i] - theta[j]
                H[k, s_th] += Vi * V[j] * (G[i, j] * np.sin(dth) - B[i, j] * np.cos(dth))

    def _dQi_dT(
        self, k: int, i: int, theta: np.ndarray, V: np.ndarray, H: np.ndarray
    ) -> None:
        """
        ∂Qᵢ/∂θᵢ  =  Pᵢ − Vᵢ² Gᵢᵢ
        ∂Qᵢ/∂θⱼ  = −VᵢVⱼ (Gᵢⱼ cos θᵢⱼ + Bᵢⱼ sin θᵢⱼ)
        """
        G, B, Vi = self.ybus.G, self.ybus.B, V[i]
        Pi = self._Pi(i, theta, V)

        s_th = self.sv.bus_idx_to_th.get(i)
        if s_th is not None:
            H[k, s_th] += Pi - Vi ** 2 * G[i, i]

        for j in range(self.ybus.n):
            if j == i:
                continue
            s_th = self.sv.bus_idx_to_th.get(j)
            if s_th is not None:
                dth = theta[i] - theta[j]
                H[k, s_th] += -Vi * V[j] * (G[i, j] * np.cos(dth) + B[i, j] * np.sin(dth))

    def _dPij_dT(
        self, k: int, branch_id: int, side: str,
        theta: np.ndarray, V: np.ndarray, H: np.ndarray
    ) -> None:
        """
        ∂Pᵢⱼ/∂θᵢ  =  VᵢVⱼ (gₛ sin θᵢⱼ − bₛ cos θᵢⱼ)
        ∂Pᵢⱼ/∂θⱼ  = −∂Pᵢⱼ/∂θᵢ
        """
        g_s, b_s, _ = self.ybus.branch_params(branch_id)
        i, j = self.ybus.branch_buses(branch_id)

        if side == "to":
            i, j = j, i

        dth = theta[i] - theta[j]
        d_thi = V[i] * V[j] * (g_s * np.sin(dth) - b_s * np.cos(dth))

        s_th_i = self.sv.bus_idx_to_th.get(i)
        if s_th_i is not None:
            H[k, s_th_i] += d_thi
        s_th_j = self.sv.bus_idx_to_th.get(j)
        if s_th_j is not None:
            H[k, s_th_j] += -d_thi

    def _dQij_dT(
        self, k: int, branch_id: int, side: str,
        theta: np.ndarray, V: np.ndarray, H: np.ndarray
    ) -> None:
        """
        ∂Qᵢⱼ/∂θᵢ  = −VᵢVⱼ (gₛ cos θᵢⱼ + bₛ sin θᵢⱼ)
        ∂Qᵢⱼ/∂θⱼ  = −∂Qᵢⱼ/∂θᵢ
        """
        g_s, b_s, b_sh2 = self.ybus.branch_params(branch_id)
        i, j = self.ybus.branch_buses(branch_id)

        if side == "to":
            i, j = j, i

        dth = theta[i] - theta[j]
        d_thi = -V[i] * V[j] * (g_s * np.cos(dth) + b_s * np.sin(dth))

        s_th_i = self.sv.bus_idx_to_th.get(i)
        if s_th_i is not None:
            H[k, s_th_i] += d_thi
        s_th_j = self.sv.bus_idx_to_th.get(j)
        if s_th_j is not None:
            H[k, s_th_j] += -d_thi

    # ─────────────────────────────────────────────────────────────────────────
    # Analytical partial derivatives – |V| (voltage magnitude) derivatives
    # ─────────────────────────────────────────────────────────────────────────

    def _dPi_dV(
        self, k: int, i: int, theta: np.ndarray, V: np.ndarray, H: np.ndarray
    ) -> None:
        """
        ∂Pᵢ/∂|Vᵢ| = (Pᵢ + Vᵢ² Gᵢᵢ) / Vᵢ
        ∂Pᵢ/∂|Vⱼ| =  Vᵢ (Gᵢⱼ cos θᵢⱼ + Bᵢⱼ sin θᵢⱼ)
        """
        G, B, Vi = self.ybus.G, self.ybus.B, V[i]
        Pi = self._Pi(i, theta, V)

        H[k, self.sv.bus_idx_to_v[i]] += (Pi + Vi ** 2 * G[i, i]) / Vi

        for j in range(self.ybus.n):
            if j == i:
                continue
            dth = theta[i] - theta[j]
            H[k, self.sv.bus_idx_to_v[j]] += Vi * (G[i, j] * np.cos(dth) + B[i, j] * np.sin(dth))

    def _dQi_dV(
        self, k: int, i: int, theta: np.ndarray, V: np.ndarray, H: np.ndarray
    ) -> None:
        """
        ∂Qᵢ/∂|Vᵢ| = (Qᵢ − Vᵢ² Bᵢᵢ) / Vᵢ
        ∂Qᵢ/∂|Vⱼ| =  Vᵢ (Gᵢⱼ sin θᵢⱼ − Bᵢⱼ cos θᵢⱼ)
        """
        G, B, Vi = self.ybus.G, self.ybus.B, V[i]
        Qi = self._Qi(i, theta, V)

        H[k, self.sv.bus_idx_to_v[i]] += (Qi - Vi ** 2 * B[i, i]) / Vi

        for j in range(self.ybus.n):
            if j == i:
                continue
            dth = theta[i] - theta[j]
            H[k, self.sv.bus_idx_to_v[j]] += Vi * (G[i, j] * np.sin(dth) - B[i, j] * np.cos(dth))

    def _dPij_dV(
        self, k: int, branch_id: int, side: str,
        theta: np.ndarray, V: np.ndarray, H: np.ndarray
    ) -> None:
        """
        ∂Pᵢⱼ/∂|Vᵢ| =  2Vᵢ gₛ − Vⱼ (gₛ cos θᵢⱼ + bₛ sin θᵢⱼ)
        ∂Pᵢⱼ/∂|Vⱼ| = −Vᵢ (gₛ cos θᵢⱼ + bₛ sin θᵢⱼ)
        """
        g_s, b_s, _ = self.ybus.branch_params(branch_id)
        i, j = self.ybus.branch_buses(branch_id)

        if side == "to":
            i, j = j, i

        Vi, Vj = V[i], V[j]
        dth = theta[i] - theta[j]
        c, s_ = np.cos(dth), np.sin(dth)

        H[k, self.sv.bus_idx_to_v[i]] += 2 * Vi * g_s - Vj * (g_s * c + b_s * s_)
        H[k, self.sv.bus_idx_to_v[j]] += -Vi * (g_s * c + b_s * s_)

    def _dQij_dV(
        self, k: int, branch_id: int, side: str,
        theta: np.ndarray, V: np.ndarray, H: np.ndarray
    ) -> None:
        """
        ∂Qᵢⱼ/∂|Vᵢ| = −2Vᵢ (bₛ + b_sh2) − Vⱼ (gₛ sin θᵢⱼ − bₛ cos θᵢⱼ)
        ∂Qᵢⱼ/∂|Vⱼ| = −Vᵢ (gₛ sin θᵢⱼ − bₛ cos θᵢⱼ)
        """
        g_s, b_s, b_sh2 = self.ybus.branch_params(branch_id)
        i, j = self.ybus.branch_buses(branch_id)

        if side == "to":
            i, j = j, i

        Vi, Vj = V[i], V[j]
        dth = theta[i] - theta[j]
        c, s_ = np.cos(dth), np.sin(dth)

        H[k, self.sv.bus_idx_to_v[i]] += -2 * Vi * (b_s + b_sh2) - Vj * (g_s * s_ - b_s * c)
        H[k, self.sv.bus_idx_to_v[j]] += -Vi * (g_s * s_ - b_s * c)

    def __repr__(self) -> str:
        types = [m.mtype for m in self.measurements]
        return (
            f"Jacobian(m={self.m}, n_state={self.sv.n_state})\n"
            f"  types : {types}"
        )