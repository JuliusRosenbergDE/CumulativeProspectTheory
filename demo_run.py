from classes import *


# initialize and set an environment
env1 = Environment()
env1.set_env_props(
    initial_individual_position=1,
    initial_size=500,
    max_size=500,
    spawn_variation=0.02,
    initial_preferences='mixed',
    reproduction_hierarchy='medium'
)

# set the environments barrier
env1.set_barrier(
    barrier=Barrier(
        speed=0,
        movement="random",
        move_distribution=expon(loc=1, scale=2)
    )
)

# set the environments prospect spawner
env1.set_prospect_spawner(
    prospect_spawner=ProspectSpawner(
        n_outcomes=2,
        expected_value=-0.5,
        noisy=True,
        noisy_sd=1,
        size_distribution=expon(loc=-1.5, scale=3)
    )
)

# link a grapher to the environment
grapher1 = PopulationGrapher(environment=env1)

# spawn the first population with mixed preferences
env1.spawn()

# create statistics for the beginning, this is not neccessary
env1.create_statistics()
env1.create_time_statistics()

# run the simulation for 1000 cycles
env1.cycle(rounds=100)

# print median values
print('MEDIAN ALPHA AND BETA:', env1.statistics.time_median_alpha[-1])
print('MEDIAN LAMBDA:', env1.statistics.time_median_lampda[-1])
print('MEDIAN GAMMA:', env1.statistics.time_median_gamma[-1])
print('MEDIAN DELTA:', env1.statistics.time_median_delta[-1])


# graph the final population
grapher1.graph_population_combined()
