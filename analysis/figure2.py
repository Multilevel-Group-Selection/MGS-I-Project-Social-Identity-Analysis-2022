import matplotlib.pyplot as plt

population = [i for i in range(20, 501)]

max_network_connection = [pop*(pop-1)/2 for pop in population]

max_moore_connection = [8*pop for pop in population]

max_density = [max_moore/max_network for max_moore, max_network in
               zip(max_moore_connection, max_network_connection)]

plt.figure(figsize=(4, 4))
plt.plot(population, max_density, "k")
plt.ylabel('Network Density', fontsize=10)
plt.xlabel('Population', fontsize=10)
plt.title('Maximum Social Network Density', fontsize=11)
plt.plot(68, 0.23880597014925373, 'ko', ms=6)
plt.plot(158, 0.10191082802547771, 'ko', ms=6)
plt.annotate('low population density case', xy=(70, 0.3), fontsize=9)
plt.annotate('high population density case', xy=(160, 0.15), fontsize=9)
plt.tight_layout()
plt.show()
