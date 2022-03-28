import matplotlib.pyplot as plt

from social_identity.simulate import simulate_social_identity_model

percent_of_contributors = simulate_social_identity_model(
    length=21,
    density=0.3,
    initial_percent=0.3,
    effort=1,
    pressure=8,
    synergy=7,
    use_strong_commitment=True,  # If True then the strong commitment is applied else the weak commitment is applied
    tick_max=200,
    show_plot_every=10
)

plt.plot(percent_of_contributors)
plt.title("Percent of Contributors")
plt.xlabel("tick")
plt.ylabel("percent")
plt.show()
