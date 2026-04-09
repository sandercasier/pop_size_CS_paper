import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import genextreme
import pandas as pd
from joblib import Parallel, delayed


###----------###
#  Parameters  #
###----------###
# Demographic parameters
mut_rate = None
bottleneck_strength = None
# Pharmacodynamic parameters
µmax = 1.231
µmin = -12.31
HILL = 2.292

# Resistance and relative growth rate parameters
WT_MIC = 0.25
c_frechet=-0.6286530209634957
loc_frechet=1.571959700166432
scale_frechet=0.556903087140513

alpha = 1 # variable
loc_error = 0
sigma_error = 0.1

# Simulation parameters
length_experiment = 24 * 16
t = np.linspace(0, length_experiment, length_experiment)
days = np.arange(length_experiment) / 24
n_replicates = 1000
bottleneck_strength = 0.001

fitness_min, fitness_max, fitness_step = 0, 1.5, 0.05
fitness_bins = np.round(np.arange(fitness_min, fitness_max + fitness_step, fitness_step), 3)
n_fitness = len(fitness_bins)

MIC_min, MIC_max, MIC_step = 0.03125, 64, 2**(1/6)
n_bins = int(np.round(np.log(MIC_max / MIC_min) / np.log(MIC_step))) + 1
MIC_bins = MIC_min * MIC_step**np.arange(n_bins)

c0 = 0.0625
ab_increase = 2**(1/3)

y0_template = np.zeros((n_bins, n_fitness))

MIC_matrix = np.tile(MIC_bins[:, None], (1, n_fitness))
fitness_matrix = np.tile(fitness_bins, (n_bins, 1))

###--------------------###
# Pharmacodynamic model  #
###--------------------###
def model(y_flat, t, params, c):
    K, µmin, µmax, HILL, MIC_bins, fitness_bins = params
    y = y_flat.reshape((n_bins, n_fitness))
    dydt = np.zeros_like(y)

    total_pop = np.sum(y)
    logistic = (1 - total_pop/K) if K > 0 else 1.0

    for i, MIC in enumerate(MIC_bins):
        for j, F in enumerate(fitness_bins):
            µF = µmax * F
            denom = (c / MIC)**HILL - µmin / µF if µF != 0 else 1.0
            growth_rate = µF - (µF - µmin) * (c / MIC)**HILL / denom
            dydt[i, j] = growth_rate * y[i, j] * logistic

    return dydt.flatten()

###----------###
#  Simulation  #
###----------###
def run_simulation(K, start_pop, µ, n_replicates=1000):

    final_MICs = []
    final_fitnesses = []
    extinct_count = 0

    for _ in range(n_replicates):

        y0 = y0_template.copy()
        y0[np.where(np.isclose(MIC_bins, WT_MIC))[0][0],
           n_fitness - round((fitness_max - 1) / fitness_step) - 1] = start_pop

        c = c0

        for k in range(1, length_experiment):

            # Solve ODE's
            params = (K, µmin, µmax, HILL, MIC_bins, fitness_bins)
            ts = [t[k-1], t[k]]
            y = odeint(model, y0.flatten(), ts, args=(params, c))
            y[1][y[1] < 1] = 0
            births = np.clip(y[1].reshape((n_bins, n_fitness)).copy() - y0,0,None)
            y0 = y[1].reshape((n_bins, n_fitness)).copy()

            # Mutation process
            total_births = np.sum(births)
            n_mut = np.random.poisson(max(0, total_births * µ)) if total_births > 0 else 0
            
            if n_mut > 0 and total_births > 0:
                probs = births.flatten() / total_births
                
                for _ in range(n_mut):
                    # Determine which subpopulations mutate
                    source_flat = np.random.choice(n_bins*n_fitness, p=probs)
                    source_r, source_f = np.unravel_index(source_flat, (n_bins, n_fitness))
                    
                    # Sample resistance
                    r_increase = genextreme.rvs(c=c_frechet,loc=loc_frechet,scale=scale_frechet)
                    r_increase = max(r_increase, 1)
                    new_r = np.clip(MIC_bins[source_r] * r_increase, MIC_bins[0], MIC_bins[-1])
                    r_index = min(np.searchsorted(MIC_bins, new_r, side='left'), n_bins - 1)-1

                    # Sample fitness cost
                    fitness_sample = (1 - alpha * (np.log2(MIC_bins[r_index]) - np.log2(WT_MIC)) / (np.log2(MIC_max) - np.log2(WT_MIC)) + np.random.normal(loc=loc_error, scale=sigma_error))
                    fitness_sample = np.clip(fitness_bins[source_f] * fitness_sample, 0, None)
                    f_index = min(np.searchsorted(fitness_bins, fitness_sample, side='left'),n_fitness - 1)-1
                    
                    # Classify mutants in correct subpopulation
                    if y0[source_r, source_f] >= 1:
                        y0[source_r, source_f] -= 1
                        y0[r_index, f_index] += 1
                        
             # Bottleneck every 24h
            if k % 24 == 0:
                total_pop = np.sum(y0)
                if total_pop > 0:
                    bottlenecked_total = int(total_pop * bottleneck_strength)
                    freqs = y0.flatten() / total_pop
                    new_counts = np.random.multinomial(bottlenecked_total, freqs)
                    y0 = new_counts.reshape((n_bins, n_fitness))
                if total_pop > 0.75 * K:
                    c *= ab_increase 

        total_pop = np.sum(y0)
        if total_pop == 0:
            extinct_count += 1
        else:
            w = y0 / total_pop
            final_MICs.append(np.sum(w * MIC_matrix))
            final_fitnesses.append(np.sum(w * fitness_matrix))

    extinction_fraction = extinct_count / n_replicates

    mic_mean = np.mean(final_MICs) if final_MICs else 0
    mic_ci = np.percentile(final_MICs, [2.5, 97.5]) if final_MICs else (0, 0)

    fit_mean = np.mean(final_fitnesses) if final_fitnesses else 0
    fit_ci = np.percentile(final_fitnesses, [2.5, 97.5]) if final_fitnesses else (0, 0)

    ext_ci = np.percentile(
        np.random.binomial(1, extinction_fraction, n_replicates),
        [2.5, 97.5]
    )

    return mic_mean, mic_ci, fit_mean, fit_ci, extinction_fraction, ext_ci

###----------------------###
#  Parameter combinations  #
###----------------------###

def mutation_rate_grid():
    rates = []
    for e in [-7, -8, -9, -10]:
        base = 10**e
        for m in [1, 0.8, 0.6, 0.4, 0.2]:
            val = m * base
            if val >= 1e-10:
                rates.append(val)
    return sorted(rates, reverse=True)

mutation_rates = mutation_rate_grid()

param_combos = []
for µ in mutation_rates:
    param_combos.append({"Population": "Small", "K": 1e9,  "start_pop": 1e6, "µ": µ})
    param_combos.append({"Population": "Large", "K": 1e11, "start_pop": 1e8, "µ": µ})

###------------------------###
#  Run parallel simulations  #
###------------------------###

def run_combo(p):
    mic, mic_ci, fit, fit_ci, ext, ext_ci = run_simulation(
        K=p["K"], start_pop=p["start_pop"], µ=p["µ"]
    )
    return {
        "Population": p["Population"],
        "Mutation_rate": p["µ"],
        "MIC_mean": mic,
        "MIC_CI_low": mic_ci[0],
        "MIC_CI_high": mic_ci[1],
        "Fitness_mean": fit,
        "Fitness_CI_low": fit_ci[0],
        "Fitness_CI_high": fit_ci[1],
        "Extinction_fraction": ext,
        "Extinction_CI_low": ext_ci[0],
        "Extinction_CI_high": ext_ci[1]
    }

results = Parallel(n_jobs=15, verbose=10)(
    delayed(run_combo)(p) for p in param_combos
)

df = pd.DataFrame(results)

###-----------###
#  Save output  #
###-----------###

df.to_excel("µ-a_LVX.xlsx", index=False)