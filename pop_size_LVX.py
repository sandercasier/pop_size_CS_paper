import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import genextreme
from joblib import Parallel, delayed
import pandas as pd

###----------###
#  Parameters  #
###----------###

# Demographic parameters
mut_rate = 5e-9
bottleneck_strength = 0.001

# Pharmacodynamic parameters
µmax = 1.231
µmin = -12.31
HILL = 2.292

# Resistance and relative growth rate parameters
WT_MIC = 0.25
c_frechet=-0.6286530209634957
loc_frechet=1.571959700166432
scale_frechet=0.556903087140513

alpha = 0.25
loc_error = 0
sigma_error = 0.1

# Simulation parameters
length_experiment = 24 * 16
t = np.linspace(0, length_experiment, length_experiment)
days = np.arange(length_experiment) / 24
n_replicates = 1000

fitness_min, fitness_max, fitness_step = 0, 1.5, 0.05
fitness_bins = np.round(np.arange(fitness_min, fitness_max + fitness_step, fitness_step), 3)
n_fitness = len(fitness_bins)

MIC_min, MIC_max, MIC_step = 0.03125, 64, 2**(1/6)
n_bins = int(np.round(np.log(MIC_max / MIC_min) / np.log(MIC_step))) + 1
MIC_bins = MIC_min * MIC_step**np.arange(n_bins)

c0 = 0.0625
ab_increase = 2**(1/3)

y0_template = np.zeros((n_bins, n_fitness))

MIC_matrix = np.tile(MIC_bins[:,None], (1,len(fitness_bins)))
fitness_matrix = np.tile(fitness_bins, (n_bins,1))
solutions_over_time = []

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

def run_simulation(K, start_pop, µ, length_experiment, bottleneck_strength, ab_increase, n_replicates=10):
    final_MICs = []
    final_fitnesses = []
    extinct_count = 0

    for _ in range(n_replicates):
        y0 = y0_template.copy()
        y0[np.where(np.isclose(MIC_bins, WT_MIC))[0][0], n_fitness-round((fitness_max-1)/fitness_step)-1] = start_pop
        c = c0
        t = np.linspace(0, length_experiment, length_experiment)

        for k in range(1, length_experiment):
                    
            # Solve ODE's
            params = (K, µmin, µmax, HILL, MIC_bins, fitness_bins)
            ts = [t[k-1], t[k]]
            y = odeint(model, y0.flatten(), ts, args=(params, c))
            y[1][y[1] < 1] = 0
            births = np.clip(y[1].reshape((n_bins, n_fitness)).copy() - y0,0,None)
            y0 = y[1].reshape((n_bins, n_fitness)).copy()
            solutions_over_time.append(y0.copy())


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
         

        # Calculate final weighted MIC and fitness
        total_pop = np.sum(y0)
        if total_pop == 0:
            extinct_count += 1
        else:
            weights = y0 / total_pop
            weighted_MIC = np.sum(weights * MIC_matrix)
            weighted_fitness = np.sum(weights * fitness_matrix)
            final_MICs.append(weighted_MIC)
            final_fitnesses.append(weighted_fitness)

      # Compute MIC 
    if len(final_MICs) > 0:
        mic_mean = np.mean(final_MICs)
        mic_sd   = np.std(final_MICs, ddof=1)
        mic_se   = mic_sd / np.sqrt(len(final_MICs))
        mic_ci95_low  = mic_mean - 1.96 * mic_se
        mic_ci95_high = mic_mean + 1.96 * mic_se
    else:
        mic_mean = mic_sd = mic_ci95_low = mic_ci95_high = 0

    # Compute fitness cost
    if len(final_fitnesses) > 0:
        fit_mean = np.mean(final_fitnesses)
        fit_sd   = np.std(final_fitnesses, ddof=1)
        fit_se   = fit_sd / np.sqrt(len(final_fitnesses))
        fit_ci95_low  = fit_mean - 1.96 * fit_se
        fit_ci95_high = fit_mean + 1.96 * fit_se
    else:
        fit_mean = fit_sd = fit_ci95_low = fit_ci95_high = 0

    return (
        mic_mean, mic_sd, mic_ci95_low, mic_ci95_high,
        fit_mean, fit_sd, fit_ci95_low, fit_ci95_high,
        extinct_count
    )

###-----------------------------###
#  Parameter combinations output  #
###-----------------------------###

n_points = 28

K_values = np.logspace(12, 3, n_points)     
start_pop_values = K_values / 1e3             

param_combos = [
    {"K": K, "start_pop": start_pop, "µ": mut_rate}
    for K, start_pop in zip(K_values, start_pop_values)
]

###------------------------###
#  Run parallel simulations  #
###------------------------###

def run_param_combo(params):
    return run_simulation(
        K=params["K"], start_pop=params["start_pop"], µ=params["µ"],
        length_experiment=length_experiment, bottleneck_strength=0.001,
        ab_increase=ab_increase, n_replicates=n_replicates)

results = Parallel(n_jobs=15, verbose=10)(
    delayed(run_param_combo)(params) for params in param_combos
)

###-----------###
#  Save output  #
###-----------###

pop_sizes = [combo["K"] for combo in param_combos]
mic_mean_vals = [r[0] for r in results]
mic_sd_vals   = [r[1] for r in results]
mic_ci_low    = [r[2] for r in results]
mic_ci_high   = [r[3] for r in results]

fit_mean_vals = [r[4] for r in results]
fit_sd_vals   = [r[5] for r in results]
fit_ci_low    = [r[6] for r in results]
fit_ci_high   = [r[7] for r in results]

ext_vals      = [r[8] for r in results]


df_results = pd.DataFrame({
    "K": [combo["K"] for combo in param_combos],
    "start_pop": [combo["start_pop"] for combo in param_combos],
    "mutation_rate": [combo["µ"] for combo in param_combos],

    # MIC stats
    "MIC_mean": mic_mean_vals,
    "MIC_sd": mic_sd_vals,
    "MIC_CI95_low": mic_ci_low,
    "MIC_CI95_high": mic_ci_high,

    # Fitness stats
    "Fitness_mean": fit_mean_vals,
    "Fitness_sd": fit_sd_vals,
    "Fitness_CI95_low": fit_ci_low,
    "Fitness_CI95_high": fit_ci_high,

    # Extinctions
    "Extinct_lineages": ext_vals,
})

df_results.to_excel("pop_size_LVX.xlsx", index=False)