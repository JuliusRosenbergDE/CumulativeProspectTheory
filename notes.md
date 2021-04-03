current relevant ideas:
    
general:
    - binarize populations, env, or whole setups, to save and reuse them,
        maybe cool exercise, to store them as JSON-data
    - some high level stop and go UI or CLI
    - some timeline to change randomness or induce shocks at certain points in time

    - allow different reproduction behaviors, right now some randomnly chosen individual can reproduce, maybe do some property called reproduction_hierarchy or something

    - some cemetery object/table/database to store all historic individuals
    - some age property of the individuals to examine lifetime vs. presence etc.

    - statistics_every property, probably best to always increment time, as this is no computational work basically
    - only time statistics or also histograms, maybe all is useful to do the tables etc.
    - for the livegraphing: graph_every or similar property

class ideas:
    Environment class:
        - good times is not used, remove or find usecase

        - inject the spawning preferences with some kind of constants/templates that can be imported
    ProspectSpawner class:
        - sizes in spawning prospects is from mediocristan
        - probably good idea to include options for extremistan
        - also the scale of the normal distribution is the expected value, this time 0 is always the same sigma distance from the expected value for all expected values


features:
    - Prospect spawner randomness type
    - reproduction hierarchy
    - statistics every
    - graph every
    - cemetery
    