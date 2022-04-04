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
from social_identity.simulate import simulate_social_identity_model
from lattice.torus import plot_matrix_colorbar, plot_matrix_values

# parameters of the simulation
length = 21  # length of the social space
density = 0.3  # density of spots randomly occupied by agents
initial_percent = 0.3  # initial percent of contributing agents
use_strong_commitment = False  # if True then the model applies the strong commitment else the model applies the weak commitment
tick_max = 200  # the maximum number of attempts at one simulation
Ngrid = 11  # number of points in ranges for synergy and pressure
# create ranges for the simulation
syn = np.linspace(0, 10, Ngrid)  # grid nodes for synergy
pre = np.linspace(0, 10, Ngrid)  # grid nodes for pressure

N_points = len(syn)  # Number of points of synergy and pressure

N_runs = 3  # Number of runs of the same setup for averaging

PE_MSI = np.zeros((N_points, N_points))
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
# loops over pressure and synergy
for ip in range(N_points):
    for ie in tqdm(range(N_points), file=sys.stdout):
        fin_MSI = []
        f0_total = []
        f1_total = []
        f2_total = []
        f3_total = []
        for i in range(N_runs):  # loop over number of runs for the same setup
            per_cont_model1, f0, f1, f2, f3 = simulate_social_identity_model(
                length=length,
                density=density,
                initial_percent=initial_percent,
                effort=1,
                pressure=pre[ip],
                synergy=syn[ie],
                use_strong_commitment=use_strong_commitment,  # If True then the strong commitment is applied else the weak commitment is applied
                tick_max=tick_max,
                show_plot_every=0
            )
            fin_MSI.append(per_cont_model1[-1])
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
        # mean values for the same setup
        PE_MSI[ip, ie] = np.mean(fin_MSI)
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
        # f0_space[ip, ie] = np.mean([f[0] + f[1] for f in f0_total])
        # f1_space[ip, ie] = np.mean([f[0] + f[1] for f in f1_total])
        # f2_space[ip, ie] = np.mean([f[0] + f[1] for f in f2_total])
        # f3_space[ip, ie] = np.mean([f[0] + f[1] for f in f3_total])
        # f0_noncontrib_space[ip, ie] = np.mean([f[0] for f in f0_total])
        # f1_noncontrib_space[ip, ie] = np.mean([f[0] for f in f1_total])
        # f2_noncontrib_space[ip, ie] = np.mean([f[0] for f in f2_total])
        # f3_noncontrib_space[ip, ie] = np.mean([f[0] for f in f3_total])
        # f0_contrib_space[ip, ie] = np.mean([f[1] for f in f0_total])
        # f1_contrib_space[ip, ie] = np.mean([f[1] for f in f1_total])
        # f2_contrib_space[ip, ie] = np.mean([f[1] for f in f2_total])
        # f3_contrib_space[ip, ie] = np.mean([f[1] for f in f3_total])
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

info = f"L = {length}, D = {density}, It = {initial_percent}, Tmax = {tick_max}, " \
       f"{'strong' if use_strong_commitment else 'weak'}"

print('Averaged for the Social Identity Model:\n{info}')
print(PE_MSI)
print('time of simulations')
print(time.time() - start_time)

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(syn, pre, PE_MSI, levels=np.linspace(0, 100, 11))
fig.colorbar(cp)  # Add a color bar to a plot
ax.set_title(f'Averaged for the Social Identity Model:\n{info}')
ax.set_xlabel('synergy')
ax.set_ylabel('pressure')
plt.show()
now = datetime.now().date()
np.savetxt(f"averaged_contributors_percent_{now}.csv", PE_MSI, delimiter=",")

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(syn, pre, gaussian_filter(PE_MSI, 0.5), levels=np.linspace(0, 100, 11))
fig.colorbar(cp)  # Add a color bar to a plot
ax.set_title(f'Averaged for the Social Identity Model:\n{info} - filtered')
ax.set_xlabel('synergy')
ax.set_ylabel('pressure')
plt.show()
vmax = max(f0_space.max(), f1_space.max(), f2_space.max(), f3_space.max())
plot_matrix_colorbar(
    np.array(f0_space),
    title=f"No threat to self or group: {info}",
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
    title=f"Threat to self but not group: {info}",
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
    title=f"Threat to group but not self: {info}",
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
    title=f"Threat to self and group: {info}",
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
np.savetxt(f"no_threat_to_self_or_group_{now}.csv", f0_space, delimiter=",")
np.savetxt(f"threat_to_self_but_not_group_{now}.csv", f1_space, delimiter=",")
np.savetxt(f"threat_to_group_but_not_self_{now}.csv", f2_space, delimiter=",")
np.savetxt(f"threat_to_self_and_group_{now}.csv", f3_space, delimiter=",")
np.savetxt(f"situation-behavior_combinations_{now}.csv", condition_space, delimiter=",")

vmax = max(
    f0_noncontrib_space.max(),
    f1_noncontrib_space.max(),
    f2_noncontrib_space.max(),
    f3_noncontrib_space.max(),
    f0_contrib_space.max(),
    f1_contrib_space.max(),
    f2_contrib_space.max(),
    f3_contrib_space.max()
)
plot_matrix_colorbar(
    np.array(f0_noncontrib_space),
    title=f"Non-contributors. No threat to self or group: {info}",
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
    title=f"Non-contributors. Threat to self but not group: {info}",
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
    title=f"Non-contributors. Threat to group but not self: {info}",
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
    title=f"Non-contributors. Threat to self and group: {info}",
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
    title=f"Non-contributors. Situation-Behavior Combinations: {info}",
    xlabel="synergy",
    ylabel="pressure",
    xticks=syn,
    yticks=pre
)
np.savetxt(f"non-contributors_no_threat_to_self_or_group_{now}.csv", f0_noncontrib_space, delimiter=",")
np.savetxt(f"non-contributors_threat_to_self_but_not_group_{now}.csv", f1_noncontrib_space, delimiter=",")
np.savetxt(f"non-contributors_threat_to_group_but_not_self_{now}.csv", f2_noncontrib_space, delimiter=",")
np.savetxt(f"non-contributors_threat_to_self_and_group_{now}.csv", f3_noncontrib_space, delimiter=",")
np.savetxt(f"non-contributors_situation-behavior_combinations_{now}.csv", noncontrib_condition_space, delimiter=",")

plot_matrix_colorbar(
    np.array(f0_contrib_space),
    title=f"Contributors. No threat to self or group: {info}",
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
    title=f"Contributors. Threat to self but not group: {info}",
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
    title=f"Contributors. Threat to group but not self: {info}",
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
    title=f"Contributors. Threat to self and group: {info}",
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
    title=f"Contributors. Situation-Behavior Combinations: {info}",
    xlabel="synergy",
    ylabel="pressure",
    xticks=syn,
    yticks=pre
)
np.savetxt(f"contributors_no_threat_to_self_or_group_{now}.csv", f0_contrib_space, delimiter=",")
np.savetxt(f"contributors_threat_to_self_but_not_group_{now}.csv", f1_contrib_space, delimiter=",")
np.savetxt(f"contributors_threat_to_group_but_not_self_{now}.csv", f2_contrib_space, delimiter=",")
np.savetxt(f"contributors_threat_to_self_and_group_{now}.csv", f3_contrib_space, delimiter=",")
np.savetxt(f"contributors_situation-behavior_combinations_{now}.csv", contrib_condition_space, delimiter=",")
