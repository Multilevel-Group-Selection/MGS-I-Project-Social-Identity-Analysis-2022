import inspect
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
# include the parent directory to the system path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from social_identity.simulate import simulate_social_identity_model

# Parameters of the simulation
length = 20
density = 0.3
initial_percent = 0.3
effort = 1
pressure = 6
synergy = 8
use_strong_commitment = False  # If True then the strong commitment is applied else the weak commitment is applied
tick_max = 20
stop_on_adoption = True
show_plot_every = 0  # if > 0 then the social space is plotted every show_plot_every iteration
save_animation = True
if save_animation:
    animation_filepath = f"model_{'strong' if use_strong_commitment else 'weak'}_L{length}_P{pressure}_S{synergy}_{datetime.now().date()}"
else:
    animation_filepath = ""
# Simulation
percent_of_contributors, not_focal_agent_threat_to_self_not_threat_group_freq, \
focal_agent_threat_to_self_not_threat_group_freq, not_focal_agent_threat_to_self_threat_group_freq, \
focal_agent_threat_to_self_threat_group_freq = simulate_social_identity_model(
    length=length,
    density=density,
    initial_percent=initial_percent,
    effort=effort,
    pressure=pressure,
    synergy=synergy,
    use_strong_commitment=use_strong_commitment,
    tick_max=tick_max,
    show_plot_every=show_plot_every,
    stop_on_adoption=stop_on_adoption,
    animation_filepath=animation_filepath
)

info = f"L = {length}, D = {density}, It = {initial_percent}, P = {pressure}, S = {synergy}, " \
       f"{'strong' if use_strong_commitment else 'weak'}"

plt.plot(percent_of_contributors)
plt.title(f"Percent of Contributors:\n{info}")
plt.xlabel("tick")
plt.ylabel("percent")
plt.show()


# not_focal_agent_threat_to_self_not_threat_group_freq = np.array(not_focal_agent_threat_to_self_not_threat_group_freq)
# focal_agent_threat_to_self_not_threat_group_freq = np.array(focal_agent_threat_to_self_not_threat_group_freq)
# not_focal_agent_threat_to_self_threat_group_freq = np.array(not_focal_agent_threat_to_self_threat_group_freq)
# focal_agent_threat_to_self_threat_group_freq = np.array(focal_agent_threat_to_self_threat_group_freq)
# total_changes = not_focal_agent_threat_to_self_not_threat_group_freq + focal_agent_threat_to_self_not_threat_group_freq + not_focal_agent_threat_to_self_threat_group_freq + focal_agent_threat_to_self_threat_group_freq

series = [
    not_focal_agent_threat_to_self_not_threat_group_freq,
    focal_agent_threat_to_self_not_threat_group_freq,
    not_focal_agent_threat_to_self_threat_group_freq,
    focal_agent_threat_to_self_threat_group_freq
]
titles = [
    "No threat to self or group",
    "Threat to self but not group",
    "Threat to group but not self",
    "Threat to self and group"
]
for s, t in zip(series, titles):
    plt.plot(
        [a[0] for a in s],
        label=f"{t}: non-contributors"
    )
    plt.plot(
        [a[effort] for a in s],
        label=f"{t}: contributors"
    )
plt.title(f"Situation-Behavior Combinations:\n{info}")
plt.xlabel("tick")
plt.ylabel("number of cases")
plt.legend()
plt.show()

contrib_total = 0
non_contrib_total = 0

for s in series:
    for a in s:
        non_contrib_total += a[0]
        contrib_total += a[effort]

for s, t in zip(series, titles):
    non_contrib = np.array([a[0] for a in s])
    contrib = np.array([a[effort] for a in s])
    plt.plot(
        100 * non_contrib / non_contrib_total,
        label=f"{t}: non-contributors"
    )
    plt.plot(
        100 * contrib / contrib_total,
        label=f"{t}: contributors"
    )
plt.title(f"Situation-Behavior Combinations:\n{info}")
plt.xlabel("tick")
plt.ylabel("percent")
plt.legend()
plt.show()
