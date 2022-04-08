import inspect
import os
import sys

import matplotlib.pyplot as plt
# include the parent directory to the system path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from mgs.simulate import simulate_base_model

# Parameters of the simulation
length = 21
density = 0.3
initial_percent = 0.3
effort = 1
pressure = 8
synergy = 9.5
use_strong_commitment = False  # If True then the strong commitment is applied else the weak commitment is applied
tick_max = 200
show_plot_every = 0  # if > 0 then the social space is plotted every show_plot_every iteration
use_groups = True  # if True then groups are applied to estimate behavior

percent_of_contributors = simulate_base_model(
    length=length,
    density=density,
    initial_percent=initial_percent,
    effort=effort,
    pressure=pressure,
    synergy=synergy,
    tick_max=tick_max,
    show_plot_every=show_plot_every,
    use_groups=use_groups
)

plt.plot(percent_of_contributors)
plt.title("Percent of Contributors")
plt.xlabel("tick")
plt.ylabel("percent")
plt.show()
