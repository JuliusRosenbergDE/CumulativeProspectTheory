current relevant ideas:
    
general:
    - binarize populations, env, or whole setups, to save and reuse them,
        maybe cool exercise, to store them as JSON-data
    - some high level stop and go UI or CLI
    - some timeline to change randomness or induce shocks at certain points in time
    - include the possbility for constraints, framework should allow different degrees of freedom
        -> this could be done in the individual class or probably in the environment class, or in the reproduction method
    - some age property of the individuals to examine lifetime vs. presence etc.

    - include steep reproduction hierarchy,only individuals with highes position reproduce
    - statistics_every property, probably best to always increment time, as this is no computational work basically
    - only time statistics or also histograms, maybe all is useful to do the tables etc.
    - for the livegraphing: graph_every or similar property

class ideas:
    Environment class:
        - good times is not used, remove or find usecase
        - inject the spawning preferences with some kind of constants/templates that can be imported
    ProspectSpawner class:
        - add appropriate type checking for the distributions

features:
    - cemetery
        -probably best as an env prop. should store indivs and lifetime/dateofdeath/dateofbirth/params, maybe stripped objects, value function etc. is not needed anymore, space is important since many indivs die
        -export to csv method
        -binarize method
        
    - graph layout fix
    - statistics every
    - graph every
    - binarize and save/load populations/envs/setups
    