from estimator import WLSSolver, SEResult, print_results
from math_model.static_parameter import YBusMatrix, MeasurementMatrix
from math_model.dynamic_parameter import StateVector, MeasurementFunc, Jacobian
from runtime import System
from structural_data import Bus, Branch, Measurement
from measurement.modbusconv import ModbusMeter
from measurement.meas_manager import ParallelModbus
import time


def test_onlabv2():
    meter_map = {"bus1" : '192.168.1.20',
             "bus3" : '192.168.1.21',
             "bus2" : '192.168.1.22',
             "branch3" : '192.168.1.201',
             } 

    meters = [ModbusMeter(meter_map["bus1"]),
        ModbusMeter(meter_map["bus3"]),
        ModbusMeter(meter_map["bus2"]),
        ModbusMeter(meter_map["branch3"])
        ]


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

    group = ParallelModbus(meters)
    group.connect_all()
    # meas_val = converter.to_pu_batch(meas_val)

    while True:
    # test
        meas_val = group.get_measurements()
        print("\nRaw measurements from Modbus:", meas_val)
        grid.add_measurement(position = 'bus', name="p1", id=1,  pos_id=1, mvalue=meas_val[meter_map["bus1"]]['P'], msd=0.010)
        grid.add_measurement(position = 'bus', name="q1", id=2,  pos_id=1, mvalue=meas_val[meter_map["bus1"]]['Q'], msd=0.010)
        grid.add_measurement(position = 'bus', name="p2", id=3,  pos_id=2, mvalue=meas_val[meter_map["bus2"]]['P'], msd=0.010)
        grid.add_measurement(position = 'bus', name="q2", id=4,  pos_id=2, mvalue=meas_val[meter_map["bus2"]]['Q'], msd=0.010)
        grid.add_measurement(position = 'bus', name="p3", id=5,  pos_id=3, mvalue=meas_val[meter_map["bus3"]]['P'], msd=0.010)
        grid.add_measurement(position = 'bus', name="q3", id=6,  pos_id=3, mvalue=meas_val[meter_map["bus3"]]['Q'], msd=0.010)
        grid.add_measurement(position = 'bus', name="v1", id=7,  pos_id=1, mvalue=meas_val[meter_map["bus1"]]['V'], msd=0.004)
        grid.add_measurement(position = 'bus', name="v2", id=8,  pos_id=2, mvalue=meas_val[meter_map["bus2"]]['V'], msd=0.004)
        grid.add_measurement(position = 'bus', name="v3", id=9,  pos_id=3, mvalue=meas_val[meter_map["bus3"]]['V'], msd=0.004)

        # config = create_sample_3bus_config()
        grid.build_system()    
        grid.get_ready_measurements()
        # print("\nReady measurements (after unit conversion):", grid.measurements)
        grid.estimate()
        bus_result = grid.get_bus_results()
        branch_result = grid.get_branch_results()
        print("\nBus Results:", bus_result)
        print("\nBranch Results:", branch_result)
        print(f"P13 = {branch_result[2]['p13_from']} pu, Q13 = {branch_result[2]['q13_from']} pu")
        time.sleep(2)
        
test_onlabv2()
