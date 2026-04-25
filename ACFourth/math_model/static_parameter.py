# math_model/static_parameter.py

# new module for static parameters
from structural_data import  Bus, Branch, Measurement
from measurement.measurement_services import MeasurementService,PerUnitOperator

# Build-in imports for dynamic parameters
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

#open-source imports for dynamic parameters
import numpy as np


#class IndexMapping:



class YBusMatrix:
    """
    Nodal admittance matrix built from π-model branch data.

    Attributes
    ----------
    Y      : complex admittance matrix  (n × n)
    G      : conductance matrix  = Y.real
    B      : susceptance matrix  = Y.imag
    n      : number of buses
    buses  : list of Bus objects sorted by ascending bus_id
    """

    def __init__(self, buses: List[Bus], branches: List[Branch]) -> None:
        self.buses: List[Bus] = sorted(buses, key=lambda b: b.id)
        self.branches: List[Branch] = branches
        self.n: int = len(self.buses)

        self._bus_id_to_idx: Dict[int, int] = {
            b.id: i for i, b in enumerate(self.buses)
        }
        self._branch_id_to_branch: Dict[int, Branch] = {
            br.id: br for br in branches
        }

        self.Y = np.zeros((self.n, self.n), dtype=complex)

        for br in branches:
            i = self._bus_id_to_idx[br.fbus]
            j = self._bus_id_to_idx[br.tbus]

            z_s = complex(br.rs, br.xs)
            y_s = (1.0 / z_s) if abs(z_s) > 1e-15 else 0.0
            b_sh = complex(0.0, br.xsh / 2.0)

            self.Y[i, i] += y_s + b_sh
            self.Y[j, j] += y_s + b_sh
            self.Y[i, j] -= y_s
            self.Y[j, i] -= y_s

        self.G: np.ndarray = self.Y.real.copy()
        self.B: np.ndarray = self.Y.imag.copy()

    # ── helpers ───────────────────────────────────────────────────────────────

    def bus_index(self, bus_id: int) -> int:
        """Return 0-based matrix index of a bus_id."""
        return self._bus_id_to_idx[bus_id]

    def branch(self, branch_id: int) -> Branch:
        """Retrieve Branch object by branch_id."""
        return self._branch_id_to_branch[branch_id]

    def branch_buses(self, branch_id: int) -> Tuple[int, int]:
        """Return (from_bus_index, to_bus_index) of a branch."""
        br = self.branch(branch_id)
        i = self.bus_index(br.fbus)
        j = self.bus_index(br.tbus)
        return i, j

    @property
    def n_branch(self) -> int:
        """Number of branches."""
        return len(self.branches)

    def branch_params(self, branch_id: int) -> Tuple[float, float, float]:
        """
        Return (g_s, b_s, b_sh2) for branch *branch_id*.
          g_s   : series conductance
          b_s   : series susceptance
          b_sh2 : half-line charging susceptance (xsh/2)
        """
        br = self._branch_id_to_branch[branch_id]
        z_s = complex(br.rs, br.xs)
        y_s = (1.0 / z_s) if abs(z_s) > 1e-15 else 0.0
        return float(y_s.real), float(y_s.imag), br.xsh / 2.0

    def __repr__(self) -> str:
        return f"YBus(n_buses={self.n}, n_branches={self.n_branch})"



class MeasurementMatrix:
    """
    z : (m,)   – measured value vector
    R : (m×m)  – covariance matrix diag(σ₁², …, σₘ²)
    W : (m×m)  – weight matrix R⁻¹ = diag(1/σ₁², …, 1/σₘ²)
    """

    def __init__(self, measurements: List[Measurement]) -> None:
        self.measurements = measurements
        self.m: int = len(measurements)

        # sigma = np.array([m.msd for m in measurements], dtype=float)
        sigma_pu = np.array([m.msd_pu for m in measurements], dtype=float)
        pu = np.array([m.mvalue_pu for m in measurements], dtype=float)

        self.z = pu
        print("Measurement values (z):", self.z)
        self.R = np.diag(sigma_pu ** 2)
        self.W = np.diag(1.0 / sigma_pu ** 2)

    def __repr__(self) -> str:
        return (
            f"MeasurementMatrices(m={self.m})\n"
            f"  z : {self.z.shape}\n"
            f"  R : {self.R.shape}  diag(σ²)\n"
            f"  W : {self.W.shape}  diag(1/σ²)"
        )
    
'''
    class MeasurementMatrix:

        def VarienceMatrix(self):
            pass

'''