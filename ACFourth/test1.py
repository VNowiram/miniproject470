# test1.py

from estimator import WLSSolver, SEResult, print_results, BusResult, BranchResult
from math_model.static_parameter import YBusMatrix, MeasurementMatrix
from math_model.dynamic_parameter import StateVector, MeasurementFunc, Jacobian
from structural_data import Bus, Branch, Measurement, Load, Generator
from measurement.measurement_services import MeasurementService,PerUnitOperator
from measurement.unit_converter import PerUnitConvert, PowerDirection

import numpy as np

def demo():
    
    
    # ── system data ───────────────────────────────────────────────────────────
    # buses = [
    #     Bus(bus_id=1, name="Slack", btype="slack", slack=True),
    #     Bus(bus_id=2, name="Bus2",  btype="PQ"),
    #     Bus(bus_id=3, name="Bus3",  btype="PQ"),
    # ]
    # branches = [
    #     Branch(branch_id=1, fbus=1, tbus=2, rs=0.035, xs=0.25, xsh=0.00),
    #     Branch(branch_id=2, fbus=1, tbus=3, rs=0.035, xs=0.25, xsh=0.00),
    #     Branch(branch_id=3, fbus=2, tbus=3, rs=0.035, xs=0.25, xsh=0.00),
    # ]
    # measurements = [
    #     Measurement(mtype="pinject", id=1,  mbus=1, mvalue=0.2148315, msd=0.010),
    #     Measurement(mtype="qinject", id=2,  mbus=1, mvalue=0.000171, msd=0.010),
    #     Measurement(mtype="pinject", id=3,  mbus=2, mvalue=-0.141229, msd=0.010),
    #     Measurement(mtype="qinject", id=4,  mbus=2, mvalue=0.000171, msd=0.010),
    #     Measurement(mtype="pinject", id=5,  mbus=3, mvalue=-0.07211, msd=0.010),
    #     Measurement(mtype="qinject", id=6,  mbus=3, mvalue=-0.0000018, msd=0.010),
    #     # Measurement(mtype="pflow",   id=7,  mbranch=1, mside="from", mvalue= 0.251, msd=0.010),
    #     # Measurement(mtype="qflow",   id=7,  mbranch=1, mside="from", mvalue= 0.102, msd=0.010),
    #     # Measurement(mtype="pflow",   id=8,  mbranch=2, mside="from", mvalue= 0.350, msd=0.010),
    #     Measurement(mtype="vmag",    id=7,  mbus=1, mvalue=0.97895, msd=0.004),
    #     Measurement(mtype="vmag",    id=8, mbus=2, mvalue=0.96580, msd=0.004),
    #     Measurement(mtype="vmag",    id=9, mbus=3, mvalue=0.9684, msd=0.004),
    # ]

    buses = [
        Bus(id=1, name="Slack", type="slack", slack=True),
        Bus(id=2, name="Bus2",  type="PQ"),
        Bus(id=3, name="Bus3",  type="PQ"),
    ]
    branches = [
        Branch(id=1, fbus=1, tbus=2, rs=0.035, xs=0.25, xsh=0.00),
        Branch(id=2, fbus=1, tbus=3, rs=0.035, xs=0.25, xsh=0.00),
        Branch(id=3, fbus=2, tbus=3, rs=0.035, xs=0.25, xsh=0.00),
    ]
    generators = [
        Generator(id=1, name="Gen1", gbus=1, pg=0.2148315, qg=0.000171),
    ]
    loads = [
        Load(id=1, name="Load1", lbus=2),
        Load(id=2, name="Load2", lbus=3),
    ]

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
    # se = [
    #     EstimatedState(position = 'bus', name="p1", id=1,  pos_id=1, mvalue=214.15, msd=0.010),
    #     EstimatedState(position = 'bus', name="q1", id=2,  pos_id=1, mvalue=171, msd=0.010),
    #     EstimatedState(position = 'bus', name="p2", id=3,  pos_id=2, mvalue=141.229, msd=0.010),
    #     EstimatedState(position = 'bus', name="q2", id=4,  pos_id=2, mvalue=171, msd=0.010),
    #     EstimatedState(position = 'bus', name="p3", id=5,  pos_id=3, mvalue=72.11, msd=0.010),
    #     EstimatedState(position = 'bus', name="q3", id=6,  pos_id=3, mvalue=0.0018, msd=0.010),
    #     EstimatedState(position = 'bus', name="v1",    id=7,  pos_id=1, mvalue=372, msd=0.004),
    #     EstimatedState(position = 'bus', name="v2",    id=8, pos_id=2, mvalue=367, msd=0.004),
    #     EstimatedState(position = 'bus', name="v3",    id=9, pos_id=3, mvalue=368, msd=0.004),
    # ]

    for m in measurements:
        print(m)
    
    PowerDirection.adjust_load_values(measurements, loads)
    PowerDirection.adjust_gen_values(measurements, generators)
    PerUnitConvert.to_pu_batch(measurements)
    for m in measurements:
        print(m)


    print("\nMeasurements:")
    for m in measurements:
        print(f"  {m.mvalue_pu:.5f}")
    print("\nEstimated States:")
    # ── instantiate classes ───────────────────────────────────────────────────
    ybus  = YBusMatrix(buses, branches)
    sv    = StateVector(ybus)
    mm    = MeasurementMatrix(measurements)

    jac = Jacobian(measurements, ybus, sv)
    H1 = jac.H_analytical(sv.flat_start())


    print(ybus)
    print()
    print(sv)
    print()
    print(mm)
    print()
    print(H1)


    print("\nG-bus (conductance):")
    print(np.round(ybus.G, 4))
    print("\nB-bus (susceptance):")
    print(np.round(ybus.B, 4))

    # ── solve with analytical Jacobian ────────────────────────────────────────
    solver_a = WLSSolver(jac, sv, mm, method="analytical", tol=1e-8)
    
    result_a = solver_a.solve(verbose=True)
    print_results(result_a, ybus, measurements)

    print("\n" + "="*50)
    print("Results with Analytical Jacobian:")
    print(result_a)

    # ── solve with numerical Jacobian and compare ─────────────────────────────
    solver_n = WLSSolver(jac, sv, mm, method="numerical", tol=1e-8)
    result_n = solver_n.solve(verbose=False)

    np.set_printoptions(precision=15, suppress=True)
    print(f"\n  Analytical solution: {result_a.x_est}")
    print(f"  Numerical solution:  {result_n.x_est}")

    print("max|x_a - x_n| =", np.max(np.abs(result_a.x_est - result_n.x_est)))
    print("J analytical   =", result_a.J)
    print("J numerical    =", result_n.J)
    print("max residual analytical =", np.max(np.abs(result_a.residuals)))
    print("max residual numerical  =", np.max(np.abs(result_n.residuals)))

    diff = float(np.max(np.abs(result_a.x_est - result_n.x_est)))
    print(f"\n  Analytical vs Numerical  max|Δx̂| = {diff:.3e}")
    print(f"  Both converged: {result_a.converged} / {result_n.converged}")

    print("\n" + "="*50)
    print(result_a.system_state_dict)
    print(result_n.system_state_dict)
    print("\n" + "="*50)
    print("Results with Numerical Jacobian:")
    print(f"\n\n{result_a}")
    print(result_a.bus)

    # for bus in result_n.bus:
    #     print(bus.bus_id, bus.mtype, bus.value_pu)
    
    # for bus in range(3):
    #     print(bus, result_n.V_pu[bus]*380)
    #     print(bus, result_n.x_est[-1-bus]*380)

demo()