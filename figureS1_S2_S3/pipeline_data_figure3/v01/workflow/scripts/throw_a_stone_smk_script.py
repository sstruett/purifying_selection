import numpy as np
import pandas as pd

from calc_coalescence_densities import *
from travelling_wave_approach import *

# get parameters
N = int(round(float(snakemake.wildcards.N)))
s = float(snakemake.wildcards.s)
U = float(snakemake.wildcards.U)
seed = np.random.randint(0, 2**32-1)
r = int(round(float(snakemake.wildcards.r)))


# run simulation
thrown_stone = throw_a_stone(
        N,
        s, 
        U,
        options={"mode": "velocity_and_distribution",
                 "burnin": 0,
                 "burnin_fitter": snakemake.params.b,
                 "generations": snakemake.params.g,
                 "thinning": 1,
                 "return_wave": False,
                 "return_fitter": False,
                 "return_sim": False,
                 "progress_bar": False},
        seed = seed,
    )[0]  # provides a list for each repetition, but here we only do one


# summarize result
click_rate, profile = thrown_stone
df = pd.DataFrame({
    "rng_seed": seed,
    "N": N,
    "s": s,
    "U": U,
    "click_rate": click_rate,
    "profile": [profile]
}, index=[r])


# print to file
df.to_pickle(snakemake.output.data)
