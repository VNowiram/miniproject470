from runtime import System
def test_runtime():
    grid = System()
    grid.add_bus(id=1, name="Slack", type="slack", slack=True)
    grid.add_bus(id=2, name="Bus2",  type="PQ")
    grid.add_bus(id=3, name="Bus3",  type="PQ")

    grid.add_branch(id=1, fbus=1, tbus=2, rs=0.035, xs=0.25, xsh=0.00)
    grid.add_branch(id=2, fbus=1, tbus=3, rs=0.035, xs=0.25, xsh=0.00)
    grid.add_branch(id=3, fbus=2, tbus=3, rs=0.035, xs=0.25, xsh=0.00)

    grid.add_generator(id=1, name="Gen1", gbus=1, pg=214.8315, qg=0.171)

    grid.add_load(id=1, name="Load1", lbus=2)
    grid.add_load(id=2, name="Load2", lbus=3)

    grid.add_measurement(position='bus', name="p1", id=1, pos_id=1, mvalue=217.84,   msd_pu=0.010)
    grid.add_measurement(position='bus', name="q1", id=2, pos_id=1, mvalue=72,   msd_pu=0.010)
    grid.add_measurement(position='bus', name="p2", id=3, pos_id=2, mvalue=138.18,   msd_pu=0.010)
    grid.add_measurement(position='bus', name="q2", id=4, pos_id=2, mvalue=13.5,   msd_pu=0.010)
    grid.add_measurement(position='bus', name="p3", id=5, pos_id=3, mvalue=77.79,    msd_pu=0.010)
    grid.add_measurement(position='bus', name="q3", id=6, pos_id=3, mvalue=51.2691,  msd_pu=0.010)
    grid.add_measurement(position='bus', name="v1", id=7, pos_id=1, mvalue=368.54,   msd_pu=0.010)
    grid.add_measurement(position='bus', name="v2", id=8, pos_id=2, mvalue=362.3988, msd_pu=0.010)
    grid.add_measurement(position='bus', name="v3", id=9, pos_id=3, mvalue=361.4268, msd_pu=0.010)
    # grid.add_measurement(position='bus', name="p1", id=1, pos_id=1, mvalue=217.84,   msd=0.010)
    # grid.add_measurement(position='bus', name="q1", id=2, pos_id=1, mvalue=58.784,   msd=0.010)
    # grid.add_measurement(position='bus', name="p2", id=3, pos_id=2, mvalue=138.18,   msd=0.010)
    # grid.add_measurement(position='bus', name="q2", id=4, pos_id=2, mvalue=0.1359,   msd=0.010)
    # grid.add_measurement(position='bus', name="p3", id=5, pos_id=3, mvalue=77.79,    msd=0.010)
    # grid.add_measurement(position='bus', name="q3", id=6, pos_id=3, mvalue=51.2691,  msd=0.010)
    # grid.add_measurement(position='bus', name="v1", id=7, pos_id=1, mvalue=368.54,   msd=0.010)
    # grid.add_measurement(position='bus', name="v2", id=8, pos_id=2, mvalue=362.3988, msd=0.010)
    # grid.add_measurement(position='bus', name="v3", id=9, pos_id=3, mvalue=361.4268, msd=0.010)

    grid.build_system()   # calls get_ready_measurements() internally
    grid.estimate()
    grid.check_bad_data()


if __name__ == "__main__":
    test_runtime()