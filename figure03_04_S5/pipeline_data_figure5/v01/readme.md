# Purpose

This pipeline is to produce the exact data to show in figure 5 of the manuscript.
We provide a config.yaml file to define the parameters.

The figure will show the performance of the model to estimate the coalescent
density when ND fails.

We aim to have three rows, each row contains results from strong to weak
selection, when varying one of the parameters.
1. Vary s
2. Vary U
3. Vary N


## Pipeline
The pipeline does different things:
1. Simulate profile and click rate for the parameters
2. Calculate the expected Ne and coalescence density over time
3. Simulate coalescent density using slim.

The jupyter-notebook will do a step 2 basically and additionally put together
a figure.

### Simulations
Remember in case you want to calculate the statistics which are mutation based. I don't simulate mutations if s = 0.
