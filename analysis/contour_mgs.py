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
from lattice.torus import plot_matrix_colorbar, plot_matrix_values
from analysis.statistics import SeriesStatistics

# parameters of the simulation
length = 15  # length of the social space
density = 0.7  # density of spots randomly occupied by agents
initial_percent = 0.3  # initial percent of contributing agents
tick_max = 1000  # the maximum number of attempts at one simulation
Ngrid = 101  # number of points in ranges for synergy and pressure
vmax = 120000  # the maximum value at legends for frequencies
use_groups = True  # if True then the simulation uses the frequency of behavior changes in groups
epsilon = 10.0  # outliers level for statistics
stop_on_adoption = True  # if False then the simulation isn't stopped on all contributors of all non-contributors
# create ranges for the simulation
syn = np.linspace(0, 10, Ngrid)  # grid nodes for synergy
pre = np.linspace(0, 10, Ngrid)  # grid nodes for pressure

N_points = len(syn)  # Number of points of synergy and pressure

N_runs = 3  # Number of runs of the same setup for averaging

info = f"L = {length}, D = {density}, It = {initial_percent}, Tmax = {tick_max}"
print(f'MGS Model:\n{info}')

contributors_percent = np.zeros((N_points, N_points, N_runs))
ticks_number = np.zeros((N_points, N_points, N_runs), dtype=int)
f0_space = np.zeros((N_points, N_points))
f1_space = np.zeros((N_points, N_points))
f2_space = np.zeros((N_points, N_points))
f3_space = np.zeros((N_points, N_points))
condition_space = np.zeros((N_points, N_points), dtype=int)
f0_noncontrib_space = np.zeros((N_points, N_points))
f1_noncontrib_space = np.zeros((N_points, N_points))
f2_noncontrib_space = np.zeros((N_points, N_points))
f3_noncontrib_space = np.zeros((N_points, N_points))
f0_contrib_space = np.zeros((N_points, N_points))
f1_contrib_space = np.zeros((N_points, N_points))
f2_contrib_space = np.zeros((N_points, N_points))
f3_contrib_space = np.zeros((N_points, N_points))
noncontrib_condition_space = np.zeros((N_points, N_points), dtype=int)
contrib_condition_space = np.zeros((N_points, N_points), dtype=int)

start_time = time.time()
run_statistics = [SeriesStatistics(epsilon) for i in range(N_runs)]
averaged_statistics = SeriesStatistics(epsilon)
# loops over pressure and synergy
for ip in range(N_points):
    for ie in tqdm(range(N_points), file=sys.stdout):
        f0_total = []
        f1_total = []
        f2_total = []
        f3_total = []
        average_percents = np.zeros(tick_max)
        for i in range(N_runs):  # loop over number of runs for the same setup
            per_cont_model1, f0, f1, f2, f3 = simulate_base_model(
                length=length,
                density=density,
                initial_percent=initial_percent,
                effort=1,
                pressure=pre[ip],
                synergy=syn[ie],
                tick_max=tick_max,
                show_plot_every=0,
                use_groups=use_groups,
                stop_on_adoption=stop_on_adoption
            )
            if len(per_cont_model1) < tick_max:
                per_cont_model1 = per_cont_model1 + [per_cont_model1[-1]] * (tick_max - len(per_cont_model1))
            per_cont_model1 = np.array(per_cont_model1)
            average_percents = average_percents + per_cont_model1
            run_statistics[i].add_series(per_cont_model1)
            contributors_percent[ip, ie, i] = per_cont_model1[-1]
            ticks_number[ip, ie, i] = len(per_cont_model1) - 1
            f0_total.append(
                {
                    0: sum(f[0] for f in f0),
                    1: sum(f[1] for f in f0)
                }
            )
            f1_total.append(
                {
                    0: sum(f[0] for f in f1),
                    1: sum(f[1] for f in f1)
                }
            )
            f2_total.append(
                {
                    0: sum(f[0] for f in f2),
                    1: sum(f[1] for f in f2)
                }
            )
            f3_total.append(
                {
                    0: sum(f[0] for f in f3),
                    1: sum(f[1] for f in f3)
                }
            )
        average_percents /= N_runs
        averaged_statistics.add_series(average_percents)
        f0_space[ip, ie] = f0_total[-1][0] + f0_total[-1][1]  # use results of the last iteration
        f1_space[ip, ie] = f1_total[-1][0] + f1_total[-1][1]  # use results of the last iteration
        f2_space[ip, ie] = f2_total[-1][0] + f2_total[-1][1]  # use results of the last iteration
        f3_space[ip, ie] = f3_total[-1][0] + f3_total[-1][1]  # use results of the last iteration
        f0_noncontrib_space[ip, ie] = f0_total[-1][0]  # use results of the last iteration
        f1_noncontrib_space[ip, ie] = f1_total[-1][0]  # use results of the last iteration
        f2_noncontrib_space[ip, ie] = f2_total[-1][0]  # use results of the last iteration
        f3_noncontrib_space[ip, ie] = f3_total[-1][0]  # use results of the last iteration
        f0_contrib_space[ip, ie] = f0_total[-1][1]  # use results of the last iteration
        f1_contrib_space[ip, ie] = f1_total[-1][1]  # use results of the last iteration
        f2_contrib_space[ip, ie] = f2_total[-1][1]  # use results of the last iteration
        f3_contrib_space[ip, ie] = f3_total[-1][1]  # use results of the last iteration
        condition_space[ip, ie] = 1 + np.argmax(
            [
                f0_space[ip, ie],
                f1_space[ip, ie],
                f2_space[ip, ie],
                f3_space[ip, ie]
            ]
        )
        noncontrib_condition_space[ip, ie] = 1 + np.argmax(
            [
                f0_noncontrib_space[ip, ie],
                f1_noncontrib_space[ip, ie],
                f2_noncontrib_space[ip, ie],
                f3_noncontrib_space[ip, ie]
            ]
        )
        contrib_condition_space[ip, ie] = 1 + np.argmax(
            [
                f0_contrib_space[ip, ie],
                f1_contrib_space[ip, ie],
                f2_contrib_space[ip, ie],
                f3_contrib_space[ip, ie]
            ]
        )
    print(f"Completed {ip + 1} form {N_points}")

now = datetime.now().date()
filename_suffix = f"{now}_{length}_{density}_{initial_percent}_{tick_max}_{Ngrid}_{stop_on_adoption}"

mean_contributors_percent = np.mean(contributors_percent, axis=2)
# print(mean_contributors_percent)
print('time of simulations')
print(time.time() - start_time)
print("Report on each run")
with open(f'report_{filename_suffix}.txt', 'w') as f:
    for i in range(N_runs):
        print(f"Simulation #: {i + 1}")
        run_statistics[i].print_report()
        print(f"Simulation #: {i + 1}", file=f)
        run_statistics[i].print_report(f)

    print(f"Averaged in {N_runs} simulations")
    averaged_statistics.print_report()
    print(f"Averaged in {N_runs} simulations", file=f)
    averaged_statistics.print_report(f)

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(syn, pre, mean_contributors_percent, levels=np.linspace(0, 100, 11))
fig.colorbar(cp)  # Add a color bar to a plot
ax.set_title(f'MGS Model:\n{info}')
ax.set_xlabel('synergy')
ax.set_ylabel('pressure')
plt.show()

np.savetxt(f"mgs_averaged_contributors_percent_{filename_suffix}.csv", mean_contributors_percent, delimiter=",")

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(syn, pre, gaussian_filter(mean_contributors_percent, 0.5), levels=np.linspace(0, 100, 11))
fig.colorbar(cp)  # Add a color bar to a plot
ax.set_title(f'MGS Model:\n{info} - filtered')
ax.set_xlabel('synergy')
ax.set_ylabel('pressure')
plt.show()
# vmax = max(f0_space.max(), f1_space.max(), f2_space.max(), f3_space.max())
plot_matrix_colorbar(
    np.array(f0_space),
    title=f"MGS. No threat to self or group: {info}",
    mark_values=False,
    xlabel="synergy",
    ylabel="pressure",
    x=syn,
    y=pre,
    vmin=0,
    vmax=vmax
)
plot_matrix_colorbar(
    np.array(f1_space),
    title=f"MGS. Threat to self but not group: {info}",
    mark_values=False,
    xlabel="synergy",
    ylabel="pressure",
    x=syn,
    y=pre,
    vmin=0,
    vmax=vmax
)
plot_matrix_colorbar(
    np.array(f2_space),
    title=f"MGS. Threat to group but not self: {info}",
    mark_values=False,
    xlabel="synergy",
    ylabel="pressure",
    x=syn,
    y=pre,
    vmin=0,
    vmax=vmax
)
plot_matrix_colorbar(
    np.array(f3_space),
    title=f"MGS. Threat to self and group: {info}",
    mark_values=False,
    xlabel="synergy",
    ylabel="pressure",
    x=syn,
    y=pre,
    vmin=0,
    vmax=vmax
)
plot_matrix_values(
    condition_space,
    title=f"Situation-Behavior Combinations: {info}",
    xlabel="synergy",
    ylabel="pressure",
    xticks=syn,
    yticks=pre
)
np.savetxt(f"mgs_no_threat_to_self_or_group_{filename_suffix}.csv", f0_space, delimiter=",")
np.savetxt(f"mgs_threat_to_self_but_not_group_{filename_suffix}.csv", f1_space, delimiter=",")
np.savetxt(f"mgs_threat_to_group_but_not_self_{filename_suffix}.csv", f2_space, delimiter=",")
np.savetxt(f"mgs_threat_to_self_and_group_{filename_suffix}.csv", f3_space, delimiter=",")
np.savetxt(f"mgs_situation-behavior_combinations_{filename_suffix}.csv", condition_space, delimiter=",")

plot_matrix_colorbar(
    np.array(f0_noncontrib_space),
    title=f"MGS. Non-contributors. No threat to self or group:\n{info}",
    mark_values=False,
    xlabel="synergy",
    ylabel="pressure",
    x=syn,
    y=pre,
    vmin=0,
    vmax=vmax
)
plot_matrix_colorbar(
    np.array(f1_noncontrib_space),
    title=f"MGS. Non-contributors. Threat to self but not group:\n{info}",
    mark_values=False,
    xlabel="synergy",
    ylabel="pressure",
    x=syn,
    y=pre,
    vmin=0,
    vmax=vmax
)
plot_matrix_colorbar(
    np.array(f2_noncontrib_space),
    title=f"MGS. Non-contributors. Threat to group but not self:\n{info}",
    mark_values=False,
    xlabel="synergy",
    ylabel="pressure",
    x=syn,
    y=pre,
    vmin=0,
    vmax=vmax
)
plot_matrix_colorbar(
    np.array(f3_noncontrib_space),
    title=f"MGS. Non-contributors. Threat to self and group:\n{info}",
    mark_values=False,
    xlabel="synergy",
    ylabel="pressure",
    x=syn,
    y=pre,
    vmin=0,
    vmax=vmax
)
plot_matrix_values(
    noncontrib_condition_space,
    title=f"MGS. Non-contributors. Situation-Behavior Combinations:\n{info}",
    xlabel="synergy",
    ylabel="pressure",
    xticks=syn,
    yticks=pre
)
np.savetxt(f"mgs_non-contributors_no_threat_to_self_or_group_{filename_suffix}.csv", f0_noncontrib_space, delimiter=",")
np.savetxt(f"mgs_non-contributors_threat_to_self_but_not_group_{filename_suffix}.csv", f1_noncontrib_space, delimiter=",")
np.savetxt(f"mgs_non-contributors_threat_to_group_but_not_self_{filename_suffix}.csv", f2_noncontrib_space, delimiter=",")
np.savetxt(f"mgs_non-contributors_threat_to_self_and_group_{filename_suffix}.csv", f3_noncontrib_space, delimiter=",")
np.savetxt(f"mgs_non-contributors_situation-behavior_combinations_{filename_suffix}.csv", noncontrib_condition_space, delimiter=",")

plot_matrix_colorbar(
    np.array(f0_contrib_space),
    title=f"MGS. Contributors. No threat to self or group:\n{info}",
    mark_values=False,
    xlabel="synergy",
    ylabel="pressure",
    x=syn,
    y=pre,
    vmin=0,
    vmax=vmax
)
plot_matrix_colorbar(
    np.array(f1_contrib_space),
    title=f"MGS. Contributors. Threat to self but not group:\n{info}",
    mark_values=False,
    xlabel="synergy",
    ylabel="pressure",
    x=syn,
    y=pre,
    vmin=0,
    vmax=vmax
)
plot_matrix_colorbar(
    np.array(f2_contrib_space),
    title=f"MGS. Contributors. Threat to group but not self:\n{info}",
    mark_values=False,
    xlabel="synergy",
    ylabel="pressure",
    x=syn,
    y=pre,
    vmin=0,
    vmax=vmax
)
plot_matrix_colorbar(
    np.array(f3_contrib_space),
    title=f"MGS. Contributors. Threat to self and group:\n{info}",
    mark_values=False,
    xlabel="synergy",
    ylabel="pressure",
    x=syn,
    y=pre,
    vmin=0,
    vmax=vmax
)
plot_matrix_values(
    contrib_condition_space,
    title=f"MGS. Contributors. Situation-Behavior Combinations:\n{info}",
    xlabel="synergy",
    ylabel="pressure",
    xticks=syn,
    yticks=pre
)
np.savetxt(f"mgs_contributors_no_threat_to_self_or_group_{filename_suffix}.csv", f0_contrib_space, delimiter=",")
np.savetxt(f"mgs_contributors_threat_to_self_but_not_group_{filename_suffix}.csv", f1_contrib_space, delimiter=",")
np.savetxt(f"mgs_contributors_threat_to_group_but_not_self_{filename_suffix}.csv", f2_contrib_space, delimiter=",")
np.savetxt(f"mgs_contributors_threat_to_self_and_group_{filename_suffix}.csv", f3_contrib_space, delimiter=",")
np.savetxt(f"mgs_contributors_situation-behavior_combinations_{filename_suffix}.csv", contrib_condition_space, delimiter=",")
