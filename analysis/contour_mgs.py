import os
import sys
import inspect
import time

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
Ngrid = 5  # number of points in ranges for synergy and pressure
# create ranges for the simulation
syn = np.linspace(0, 10, Ngrid)  # grid nodes for synergy
pre = np.linspace(0, 10, Ngrid)  # grid nodes for pressure

N_points = len(syn)  # Number of points of synergy and pressure

N_runs = 3  # Number of runs of the same setup for averaging

PE_MSI = np.zeros((N_points, N_points))

start_time = time.time()
# loops over pressure and synergy
for ip in range(N_points):
    for ie in tqdm(range(N_points), file=sys.stdout):
        fin_MSI = []

        for i in range(N_runs):  # loop over number of runs for the same setup
            per_cont_model1 = simulate_base_model(
                length=length,
                density=density,
                initial_percent=initial_percent,
                effort=1,
                pressure=pre[ip],
                synergy=syn[ie],
                tick_max=tick_max,
                show_plot_every=0
            )

            fin_MSI.append(per_cont_model1[-1])

        # mean values for the same setup
        PE_MSI[ip, ie] = np.mean(fin_MSI)
    print(f"Completed {ip + 1} form {N_points}")

#        PE_MSI_WEAK[ip,ie] = np.mean(fin_MSI_WEAK)

print('Averaged for MSI model')
print(PE_MSI)

print('time of simulations')
print(time.time() - start_time)

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(syn, pre, PE_MSI, levels=np.linspace(0, 100, 11))
fig.colorbar(cp)  # Add a color bar to a plot
ax.set_title('Averaged for the MSG model')
ax.set_xlabel('synergy')
ax.set_ylabel('pressure')
plt.show()

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(syn, pre, gaussian_filter(PE_MSI, 0.5), levels=np.linspace(0, 100, 11))
fig.colorbar(cp)  # Add a color bar to a plot
ax.set_title('Averaged for the MSG model - filtered')
ax.set_xlabel('synergy')
ax.set_ylabel('pressure')
plt.show()


