from runtime import System
from gnn_se_trainer import DataGenerator
from gnn_se_model import GNNStateEstimator, GraphBuilder

grid = System()
# ... add buses, branches, measurements as normal ...

grid.add_bus(id=1, name="Slack", type="slack", slack=True)
grid.add_bus(id=2, name="Bus2",  type="PQ")
grid.add_bus(id=3, name="Bus3",  type="PQ")

grid.add_branch(id=1, fbus=1, tbus=2, rs=0.035, xs=0.25, xsh=0.00)
grid.add_branch(id=2, fbus=1, tbus=3, rs=0.035, xs=0.25, xsh=0.00)
grid.add_branch(id=3, fbus=2, tbus=3, rs=0.035, xs=0.25, xsh=0.00)

# grid.add_generator(id=1, name="Gen1", gbus=1, pg=214.8315, qg=0.171)

# grid.add_load(id=1, name="Load1", lbus=2)
# grid.add_load(id=2, name="Load2", lbus=3)

grid.add_measurement(position='bus', name="p1", id=1, pos_id=1, mvalue=217.84,   msd_pu=0.010)
grid.add_measurement(position='bus', name="q1", id=2, pos_id=1, mvalue=72,   msd_pu=0.010)
grid.add_measurement(position='bus', name="p2", id=3, pos_id=2, mvalue=138.18,   msd_pu=0.010)
grid.add_measurement(position='bus', name="q2", id=4, pos_id=2, mvalue=13.5,   msd_pu=0.010)
grid.add_measurement(position='bus', name="p3", id=5, pos_id=3, mvalue=77.79,    msd_pu=0.010)
grid.add_measurement(position='bus', name="q3", id=6, pos_id=3, mvalue=51.2691,  msd_pu=0.010)
grid.add_measurement(position='bus', name="v1", id=7, pos_id=1, mvalue=368.54,   msd_pu=0.010)
grid.add_measurement(position='bus', name="v2", id=8, pos_id=2, mvalue=362.3988, msd_pu=0.010)
grid.add_measurement(position='bus', name="v3", id=9, pos_id=3, mvalue=361.4268, msd_pu=0.010)

grid.build_system()
# Extract branch params directly from YBus
builder = GraphBuilder.from_ybus(grid.ybus, slack_idx=0)

from gnn_se_trainer import DataGenerator, Trainer

gen = DataGenerator.from_ybus(grid.ybus)   # same network parameters
samples = gen.generate(n_samples=5000, noise_sd=0.01, seed=42)

model = GNNStateEstimator(hidden_dim=64, n_mp_layers=4, slack_idx=0)
trainer = Trainer(model, samples, branch_params=gen.branch_params)
trainer.train(epochs=200, batch_size=64)
trainer.save("gnn_se_checkpoint.pt")


# After grid.estimate() has run and measurements are in p.u.:
gnn_input = builder.build(grid.get_ready_measurements())
result = model.predict(gnn_input)

print(result.V_pu)       # Voltage magnitude per bus [p.u.]
print(result.theta_deg)  # Voltage angle per bus [degrees]

model = Trainer.load("gnn_se_checkpoint.pt")
result = model.predict(gnn_input)