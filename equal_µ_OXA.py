import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import genextreme
import pandas as pd

###----------###
#  Parameters  #
###----------###
# Demographic parameters
populations = {
    "Large": {
        "K": 1e11,
        "start_pop": 1e8,
        "mut_rate": 5e-9,
        "bottleneck_strength": 0.001},
    "Small": {
        "K": 1e9,
        "start_pop": 1e6,
        "mut_rate": 5e-7,
        "bottleneck_strength": 0.001}
}

mut_rate = None
bottleneck_strength = None

# Pharmacodynamic parameters
µmax = 1.703
µmin = -1.298
HILL = 2.501

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

###---------------------###
#  Pharmacodynamic model  #
###---------------------###
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
def run_simulation(K, start_pop, µ=mut_rate,
                   bottleneck_strength=bottleneck_strength):

    # Initialize population
    y0 = y0_template.copy()
    
    wt_r_index = np.where(np.isclose(MIC_bins, WT_MIC))[0][0]
    wt_f_index = np.argmin(np.abs(fitness_bins - 1.0))
    y0[wt_r_index, wt_f_index] = start_pop
    
    c = c0

    # Store population
    solution = [y0.copy()]
    concentration_record = [c]

    # Evolve population
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

        solution.append(y0.copy())
        concentration_record.append(c)

    return np.array(solution), np.array(concentration_record)

###--------------###
#  Run simulation  #
###--------------###
results = {}
for pop, pars in populations.items():
    solutions_list = []
    conc_list = []

    for s in range(n_replicates):
        print(f"Starting replicate {s+1}/{n_replicates} for population {pop}")
        sol, conc = run_simulation(
            pars["K"],
            pars["start_pop"],
            pars["mut_rate"],
            pars["bottleneck_strength"]
        )
        solutions_list.append(sol)
        conc_list.append(conc)

    results[pop] = {
        "solutions": np.array(solutions_list),
        "concs": np.array(conc_list)
    }

solutions_small = results["Small"]["solutions"]
solutions_large = results["Large"]["solutions"]
conc_small = results["Small"]["concs"]
conc_large = results["Large"]["concs"]

# Filter extinct populations
def filter_extinct(solutions):
    alive_mask = solutions.sum(axis=(2,3)).min(axis=1) > 0
    return solutions[alive_mask], alive_mask

solutions_small_alive, alive_mask_small = filter_extinct(solutions_small)
solutions_large_alive, alive_mask_large = filter_extinct(solutions_large)

###-----------###
#  Save output  #
###-----------###

def mean_and_95ci(arr):
    n = arr.shape[0]
    mean = np.nanmean(arr, axis=0)
    if n > 1:
        sd = np.nanstd(arr, axis=0, ddof=1)
        se = sd / np.sqrt(n)
        ci = 1.96 * se
    else:
        ci = np.zeros_like(mean)
    return mean, mean-ci, mean+ci, n

def median_and_95ci(arr):
    median = np.nanmedian(arr, axis=0)
    low = np.nanpercentile(arr, 2.5, axis=0)
    up = np.nanpercentile(arr, 97.5, axis=0)
    n = arr.shape[0]
    return median, low, up, n

MIC_mat = np.tile(MIC_bins[:,None], (1, len(fitness_bins)))
FIT_mat = np.tile(fitness_bins, (n_bins, 1))

def weighted_MIC(solutions):
    num = (solutions * MIC_mat[None,None,:,:]).sum(axis=(2,3))
    den = solutions.sum(axis=(2,3))
    return np.where(den > 0, num/den, np.nan)

def weighted_fitness(solutions):
    num = (solutions * FIT_mat[None,None,:,:]).sum(axis=(2,3))
    den = solutions.sum(axis=(2,3))
    return np.where(den > 0, num/den, np.nan)

# Population size
tot_small = solutions_small_alive.sum(axis=(2,3))
tot_large = solutions_large_alive.sum(axis=(2,3))

mean_ts, low_ts, up_ts, _ = mean_and_95ci(tot_small)
mean_tl, low_tl, up_tl, _ = mean_and_95ci(tot_large)

low_ts_p = np.maximum(low_ts, 1)
low_tl_p = np.maximum(low_tl, 1)

# Antibiotic concentration
conc_small_alive = conc_small[alive_mask_small]
conc_large_alive = conc_large[alive_mask_large]

c_sm, c_slow, c_sup, _ = mean_and_95ci(conc_small_alive)
c_lm, c_llow, c_lup, _ = mean_and_95ci(conc_large_alive)

# Resistance
w_small = weighted_MIC(solutions_small_alive)
w_large = weighted_MIC(solutions_large_alive)

m_s, l_s, u_s, _ = mean_and_95ci(w_small)
m_l, l_l, u_l, _ = mean_and_95ci(w_large)
m_s_med, l_s_med, u_s_med, _ = median_and_95ci(w_small)
m_l_med, l_l_med, u_l_med, _ = median_and_95ci(w_large)

# Fitness cost
wf_small = weighted_fitness(solutions_small_alive)
wf_large = weighted_fitness(solutions_large_alive)

m_fs, l_fs, u_fs, _ = mean_and_95ci(wf_small)
m_fl, l_fl, u_fl, _ = mean_and_95ci(wf_large)
m_fs_med, l_fs_med, u_fs_med, _ = median_and_95ci(wf_small)
m_fl_med, l_fl_med, u_fl_med, _ = median_and_95ci(wf_large)

df_results = pd.DataFrame({
    "days": days,
    "pop_mean_small": mean_ts,
    "pop_low_small": low_ts,
    "pop_up_small": up_ts,
    "pop_mean_large": mean_tl,
    "pop_low_large": low_tl,
    "pop_up_large": up_tl,
    "conc_mean_small": c_sm,
    "conc_low_small": c_slow,
    "conc_up_small": c_sup,
    "conc_mean_large": c_lm,
    "conc_low_large": c_llow,
    "conc_up_large": c_lup,
    "res_mean_small": m_s,
    "res_low_small": l_s,
    "res_up_small": u_s,
    "res_mean_large": m_l,
    "res_low_large": l_l,
    "res_up_large": u_l,
    "res_median_small": m_s_med,
    "res_low_small_median": l_s_med,
    "res_up_small_median": u_s_med,
    "res_median_large": m_l_med,
    "res_low_large_median": l_l_med,
    "res_up_large_median": u_l_med,
    "fit_mean_small": m_fs,
    "fit_low_small": l_fs,
    "fit_up_small": u_fs,
    "fit_mean_large": m_fl,
    "fit_low_large": l_fl,
    "fit_up_large": u_fl,
    "fit_median_small": m_fs_med,
    "fit_low_small_median": l_fs_med,
    "fit_up_small_median": u_fs_med,
    "fit_median_large": m_fl_med,
    "fit_low_large_median": l_fl_med,
    "fit_up_large_median": u_fl_med
})

df_results.to_excel("equal_µ_OXA", index=False)