from classes import *

env1 = Environment()
env1.set_env_props(
    initial_individual_position=2,
    initial_size=500,
    max_size=500,
    spawn_variation=0.02
)

env1.set_barrier(barrier=Barrier(
    speed=0
))
env1.set_prospect_spawner(prospect_spawner=ProspectSpawner(
    n_outcomes=2,
    noisy=True,
    noisy_sd=1,
    expected_value=-2.0
))


env1.spawn(preferences="mixed")
env1.create_statistics()
env1.create_time_statistics()

env1.cycle(rounds=200)


# live_graphing() for the experiment specified in my document
env1.population_grapher.live_graphing()
