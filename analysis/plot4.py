import numpy as np
import matplotlib.pyplot as plt

base03_filename = "l15/base_03/mgs_averaged_contributors_percent_2022-08-08_15_0.3_0.3_1000_101_True.csv"
base07_filename = "l15/base_07/mgs_averaged_contributors_percent_2022-08-10_15_0.7_0.3_1000_101_True.csv"
new_base03_filename = "l15/new_base_03/mgs_averaged_contributors_percent_2022-07-25_15_0.3_0.3_1000_101_True.csv"
new_base07_filename = "l15/new_base_07/mgs_averaged_contributors_percent_2022-07-29_15_0.7_0.3_1000_101_True.csv"

if __name__ == "__main__":
    base03 = np.loadtxt(
        base03_filename,
        delimiter=","
    )
    base07 = np.loadtxt(
        base07_filename,
        delimiter=","
    )
    new_base03 = np.loadtxt(
        new_base03_filename,
        delimiter=","
    )
    new_base07 = np.loadtxt(
        new_base07_filename,
        delimiter=","
    )
    fig, axs = plt.subplots(2, 2, sharex="all", sharey="all")
    Ngrid = 101
    syn = np.linspace(0, 10, Ngrid)  # grid nodes for synergy
    pre = np.linspace(0, 10, Ngrid)  # grid nodes for pressure
    cp = axs[0, 0].contourf(syn, pre, base03, levels=np.linspace(0, 100, 11))
    axs[0, 0].set_title("Base, d = 0.3")
    cp = axs[1, 0].contourf(syn, pre, base07, levels=np.linspace(0, 100, 11))
    axs[1, 0].set_title("Base, d = 0.7")
    cp = axs[0, 1].contourf(syn, pre, new_base03, levels=np.linspace(0, 100, 11))
    axs[0, 1].set_title("New Base, d = 0.3")
    cp = axs[1, 1].contourf(syn, pre, new_base07, levels=np.linspace(0, 100, 11))
    axs[1, 1].set_title("New Base, d = 0.7")

    fig.text(0.5, 0.04, 'synergy', ha='center')
    fig.text(0.01, 0.5, 'pressure', va='center', rotation='vertical')

    fig.subplots_adjust(left=0.08, right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(cp, cax=cbar_ax)
    plt.show()
