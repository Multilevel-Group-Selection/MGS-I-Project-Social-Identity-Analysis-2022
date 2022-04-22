import os
import sys
import inspect
import time
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
# include the parent directory to the system path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
# include the package
from mgs.simulate import simulate_base_model

# parameters of the simulation
length = 21  # length of the social space
density = 0.3  # density of spots randomly occupied by agents
initial_percent = 0.3  # initial percent of contributing agents
tick_max = 200  # the maximum number of attempts at one simulation
Ngrid = 11  # number of points in ranges for synergy and pressure
use_groups = True  # if True then the simulation uses the frequency of behaviour changes in groups
epsilon = 1.0E-6  # zero at the floating numbers comparison
# create ranges for the simulation
syn = np.linspace(0, 10, Ngrid)  # grid nodes for synergy
pre = np.linspace(0, 10, Ngrid)  # grid nodes for pressure

N_points = len(syn)  # Number of points of synergy and pressure

N_runs = 3  # Number of runs of the same setup for averaging

contributors_percent = np.zeros((N_points, N_points, N_runs))
ticks_number = np.zeros((N_points, N_points, N_runs), dtype=int)

start_time = time.time()
# loops over pressure and synergy
for ip in range(N_points):
    for ie in tqdm(range(N_points), file=sys.stdout):
        for i in range(N_runs):  # loop over number of runs for the same setup
            per_cont_model1 = simulate_base_model(
                length=length,
                density=density,
                initial_percent=initial_percent,
                effort=1,
                pressure=pre[ip],
                synergy=syn[ie],
                tick_max=tick_max,
                show_plot_every=0,
                use_groups=use_groups
            )
            contributors_percent[ip, ie, i] = per_cont_model1[-1]
            ticks_number[ip, ie, i] = len(per_cont_model1)
    print(f"Completed {ip + 1} form {N_points}")

mean_contributors_percent = np.mean(contributors_percent, axis=2)

print('Averaged for MSI model')
print(mean_contributors_percent)

print('time of simulations')
print(time.time() - start_time)

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(syn, pre, mean_contributors_percent, levels=np.linspace(0, 100, 11))
fig.colorbar(cp)  # Add a color bar to a plot
ax.set_title('Averaged for the MSG model')
ax.set_xlabel('synergy')
ax.set_ylabel('pressure')
plt.show()

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(syn, pre, gaussian_filter(mean_contributors_percent, 0.5), levels=np.linspace(0, 100, 11))
fig.colorbar(cp)  # Add a color bar to a plot
ax.set_title('Averaged for the MSG model - filtered')
ax.set_xlabel('synergy')
ax.set_ylabel('pressure')
plt.show()

now = datetime.now().date()
np.savetxt(f"averaged_contributors_percent_mgs_{now}.csv", mean_contributors_percent, delimiter=",")

print("Report on each run")
for i in range(N_runs):
    run_contrib_space = contributors_percent[:, :, i]
    run_ticks_number = ticks_number[:, :, i]
    contrib_idx = np.where(run_contrib_space > 100.0 - epsilon)
    non_contrib_idx = np.where(run_contrib_space < epsilon)
    contributors_number = len(contrib_idx[0])
    non_contributors_number = len(non_contrib_idx[0])
    mean_run_contributors_percent = np.mean(run_contrib_space[np.where((epsilon <= run_contrib_space) & (run_contrib_space <= 100.0 - epsilon))])
    average_ticks_number = np.mean(run_ticks_number)
    contrib_ticks_number = ticks_number[contrib_idx]
    average_ticks_contrib_number = np.mean(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
    min_ticks_contrib_number = np.min(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
    max_ticks_contrib_number = np.max(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
    non_contrib_ticks_number = ticks_number[non_contrib_idx]
    average_ticks_non_contrib_number = np.mean(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
    min_ticks_non_contrib_number = np.min(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
    max_ticks_non_contrib_number = np.max(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
    print(f"simulation #: {i + 1}")
    print(f"number of synergy-pressure pairs with full adoption: {contributors_number}")
    print(f"number of synergy-pressure pairs with zero adoption: {non_contributors_number}")
    print(f"average percent of contributors excluding zero and full adoption: {mean_run_contributors_percent}")
    print(f"average number of ticks in the simulation: {average_ticks_number}")
    print(f"minimum number of ticks to get full adoption: {min_ticks_contrib_number}")
    print(f"average number of ticks to get full adoption: {average_ticks_contrib_number}")
    print(f"maximum number of ticks to get full adoption: {max_ticks_contrib_number}")
    print(f"minimum number of ticks to get zero adoption: {min_ticks_non_contrib_number}")
    print(f"average number of ticks to get zero adoption: {average_ticks_non_contrib_number}")
    print(f"maximum number of ticks to get zero adoption: {max_ticks_non_contrib_number}")
    print("---")
contrib_idx = np.where(contributors_percent > 100.0 - epsilon)
non_contrib_idx = np.where(contributors_percent < epsilon)
contributors_number = len(contrib_idx[0])
non_contributors_number = len(non_contrib_idx[0])
mean_run_contributors_percent = np.mean(contributors_percent[np.where((epsilon <= contributors_percent) & (contributors_percent <= 100.0 - epsilon))])
average_ticks_number = np.mean(ticks_number)
contrib_ticks_number = ticks_number[contrib_idx]
average_ticks_contrib_number = np.mean(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
min_ticks_contrib_number = np.min(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
max_ticks_contrib_number = np.max(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
non_contrib_ticks_number = ticks_number[non_contrib_idx]
average_ticks_non_contrib_number = np.mean(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
min_ticks_non_contrib_number = np.min(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
max_ticks_non_contrib_number = np.max(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
print("---")
print(f"Total in {N_runs} simulations")
print(f"number of synergy-pressure pairs with full adoption: {contributors_number}")
print(f"number of synergy-pressure pairs with zero adoption: {non_contributors_number}")
print(f"average percent of contributors excluding zero and full adoption: {mean_run_contributors_percent}")
print(f"average number of ticks in the simulation: {average_ticks_number}")
print(f"minimum number of ticks to get full adoption: {min_ticks_contrib_number}")
print(f"average number of ticks to get full adoption: {average_ticks_contrib_number}")
print(f"maximum number of ticks to get full adoption: {max_ticks_contrib_number}")
print(f"minimum number of ticks to get zero adoption: {min_ticks_non_contrib_number}")
print(f"average number of ticks to get zero adoption: {average_ticks_non_contrib_number}")
print(f"maximum number of ticks to get zero adoption: {max_ticks_non_contrib_number}")
print("---")
contrib_idx = np.where(mean_contributors_percent > 100.0 - epsilon)
non_contrib_idx = np.where(mean_contributors_percent < epsilon)
contributors_number = len(contrib_idx[0])
non_contributors_number = len(non_contrib_idx[0])
mean_run_contributors_percent = np.mean(mean_contributors_percent[np.where((epsilon <= mean_contributors_percent) & (mean_contributors_percent <= 100.0 - epsilon))])
average_ticks_number = np.mean(ticks_number)
contrib_ticks_number = ticks_number[contrib_idx]
average_ticks_contrib_number = np.mean(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
min_ticks_contrib_number = np.min(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
max_ticks_contrib_number = np.max(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
non_contrib_ticks_number = ticks_number[non_contrib_idx]
average_ticks_non_contrib_number = np.mean(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
min_ticks_non_contrib_number = np.min(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
max_ticks_non_contrib_number = np.max(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
print("---")
print(f"Averaged in {N_runs} simulations")
print(f"number of synergy-pressure pairs with full adoption: {contributors_number}")
print(f"number of synergy-pressure pairs with zero adoption: {non_contributors_number}")
print(f"average percent of contributors excluding zero and full adoption: {mean_run_contributors_percent}")
print(f"average number of ticks in the simulation: {average_ticks_number}")
print(f"minimum number of ticks to get full adoption: {min_ticks_contrib_number}")
print(f"average number of ticks to get full adoption: {average_ticks_contrib_number}")
print(f"maximum number of ticks to get full adoption: {max_ticks_contrib_number}")
print(f"minimum number of ticks to get zero adoption: {min_ticks_non_contrib_number}")
print(f"average number of ticks to get zero adoption: {average_ticks_non_contrib_number}")
print(f"maximum number of ticks to get zero adoption: {max_ticks_non_contrib_number}")
print("---")
