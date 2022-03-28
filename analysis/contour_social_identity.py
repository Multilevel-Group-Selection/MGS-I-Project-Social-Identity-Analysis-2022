import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from social_identity.simulate import simulate_social_identity_model

start_time = time.time()

# -------
Ngrid = 20
syn = np.linspace(0, 10, Ngrid)  # grid nodes for synergy
pre = np.linspace(0, 10, Ngrid)  # grid nodes for pressure

N_points = len(syn)  # Number of points of synergy and pressure

N_runs = 3  # Number of runs of the same setup for averaging

PE_MSI = np.zeros((N_points, N_points))

# loops over pressure and synergy
for ip in range(N_points):
    for ie in tqdm(range(N_points), file=sys.stdout):
        fin_MSI = []

        for i in range(N_runs):  # loop over number of runs for the same setup
            per_cont_model1 = simulate_social_identity_model(
                length=21,
                density=0.3,
                initial_percent=0.3,
                effort=1,
                pressure=pre[ip],
                synergy=syn[ie],
                use_strong_commitment=False,  # If True then the strong commitment is applied else the weak commitment is applied
                tick_max=200,
                show_plot_every=0
            )

            fin_MSI.append(per_cont_model1[-1])

        # mean values for the same setup
        PE_MSI[ip, ie] = np.mean(fin_MSI)
    print(f"Completed {ip + 1} form {N_points}")

#        PE_MSI_WEAK[ip,ie] = np.mean(fin_MSI_WEAK)

print('Averaged for the Social Identity Model')
print(PE_MSI)

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(syn, pre, PE_MSI)
fig.colorbar(cp)  # Add a color bar to a plot
ax.set_title('Averaged for the Social Identity Model')
ax.set_xlabel('synergy')
ax.set_ylabel('pressure')
plt.show()

print('time of simulations')
print(time.time() - start_time)


