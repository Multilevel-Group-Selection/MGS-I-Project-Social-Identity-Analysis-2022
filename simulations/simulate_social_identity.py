import matplotlib.pyplot as plt
import numpy as np

from social_identity.simulate import simulate_social_identity_model

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
    show_plot_every=show_plot_every
)

info = f"L = {length}, D = {density}, It = {initial_percent}, P = {pressure}, S = {synergy}, " \
       f"{'strong' if use_strong_commitment else 'weak'}"

plt.plot(percent_of_contributors)
plt.title(f"Percent of Contributors:\n{info}")
plt.xlabel("tick")
plt.ylabel("percent")
plt.show()


not_focal_agent_threat_to_self_not_threat_group_freq = np.array(not_focal_agent_threat_to_self_not_threat_group_freq)
focal_agent_threat_to_self_not_threat_group_freq = np.array(focal_agent_threat_to_self_not_threat_group_freq)
not_focal_agent_threat_to_self_threat_group_freq = np.array(not_focal_agent_threat_to_self_threat_group_freq)
focal_agent_threat_to_self_threat_group_freq = np.array(focal_agent_threat_to_self_threat_group_freq)
total_changes = not_focal_agent_threat_to_self_not_threat_group_freq + focal_agent_threat_to_self_not_threat_group_freq + not_focal_agent_threat_to_self_threat_group_freq + focal_agent_threat_to_self_threat_group_freq

plt.plot(not_focal_agent_threat_to_self_not_threat_group_freq, label="No threat to self or group")
plt.plot(focal_agent_threat_to_self_not_threat_group_freq, label="Threat to self but not group")
plt.plot(not_focal_agent_threat_to_self_threat_group_freq, label="Threat to group but not self")
plt.plot(focal_agent_threat_to_self_threat_group_freq, label="Threat to self and group")
plt.title(f"Situation-Behavior Combinations:\n{info}")
plt.xlabel("tick")
plt.ylabel("number of cases")
plt.legend()
plt.show()


plt.plot(100 * not_focal_agent_threat_to_self_not_threat_group_freq / total_changes, label="No threat to self or group")
plt.plot(100 * focal_agent_threat_to_self_not_threat_group_freq / total_changes, label="Threat to self but not group")
plt.plot(100 * not_focal_agent_threat_to_self_threat_group_freq / total_changes, label="Threat to group but not self")
plt.plot(100 * focal_agent_threat_to_self_threat_group_freq / total_changes, label="Threat to self and group")
plt.title(f"Situation-Behavior Combinations:\n{info}")
plt.xlabel("tick")
plt.ylabel("percent")
plt.legend()
plt.show()
