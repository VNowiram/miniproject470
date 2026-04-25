from estimator import WLSSolver, SEResult, print_results
from math_model.static_parameter import YBusMatrix, MeasurementMatrix
from math_model.dynamic_parameter import StateVector, MeasurementFunc, Jacobian
from structural_data import Bus, Branch, Measurement
from measurement.modbusconv import ModbusMeter
from measurement.meas_manager import ParallelModbus

import numpy as np

meter_map = {"bus1" : '192.168.1.201',
            #  "bus3" : '192.168.1.21',
            #  "bus2" : '192.168.1.22',
             } 

meters = [ModbusMeter(meter_map["bus1"]),
        # ModbusMeter(meter_map["bus3"]),
        # ModbusMeter(meter_map["bus2"]),
        ]

group = ParallelModbus(meters)
group.connect_all()

meas_val = group.get_measurements()
print("\nRaw measurements from Modbus:", meas_val)
# meas_val = converter.to_pu_batch(meas_val)

# ── system data ───────────────────────────────────────────────────────────
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


measurements_val = [
    Measurement(mtype="pinject",id=1,  pos_id=1, mvalue= meas_val[meter_map["bus1"]]['P'], msd=0.010),
    Measurement(mtype="qinject", id=2,  pos_id=1, mvalue= meas_val[meter_map["bus1"]]['Q'], msd=0.010),
    Measurement(mtype="vmag",  id=3,  pos_id=1, mvalue= meas_val[meter_map["bus1"]]['V'], msd=0.010),
    # Measurement(mtype="pinject", id=4,  pos_id=2, mvalue= -meas_val[meter_map["bus2"]]['P'], msd=0.010),
    # Measurement(mtype="qinject", id=5,  pos_id=2, mvalue= -meas_val[meter_map["bus2"]]['Q'], msd=0.010),
    # Measurement(mtype="vmag",  id=6,  pos_id=2, mvalue= meas_val[meter_map["bus2"]]['V'], msd=0.010),
    # Measurement(mtype="pinject", id=7,  pos_id=3, mvalue= -meas_val[meter_map["bus3"]]['P'], msd=0.010),
    # Measurement(mtype="qinject", id=8,  pos_id=3, mvalue= -meas_val[meter_map["bus3"]]['Q'], msd=0.010),
    # Measurement(mtype="vmag",  id=9,  pos_id=3, mvalue= meas_val[meter_map["bus3"]]['V'], msd=0.010),
]

# measurmemts = measurements_val
# print("\nMeasurements with values from Modbus:", measurements_val)

# ── instantiate classes ───────────────────────────────────────────────────
ybus  = YBusMatrix(buses, branches)
sv    = StateVector(ybus)
mm    = MeasurementMatrix(measurements_val)

jac = Jacobian(measurements_val, ybus, sv)
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
print_results(result_a, ybus, measurements_val)

print("\n" + "="*50)
print("Results with Analytical Jacobian:")
print(result_a.x_est)

# # ── solve with numerical Jacobian and compare ─────────────────────────────
# solver_n = WLSSolver(jac, sv, mm, method="numerical", tol=1e-8)
# result_n = solver_n.solve(verbose=True)
# print_results(result_n, ybus, measurements_val)
