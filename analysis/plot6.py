import numpy as np
import matplotlib.pyplot as plt

base03_filename = "l15/new_base_03/mgs_averaged_contributors_percent_2022-07-25_15_0.3_0.3_1000_101_True.csv"
base07_filename = "l15/new_base_07/mgs_averaged_contributors_percent_2022-07-29_15_0.7_0.3_1000_101_True.csv"
si_strong_03_filename = "l15/strong_03/averaged_contributors_percent_2022-08-03_15_0.3_1000_strong.csv"
si_strong_07_filename = "l15/strong_07/averaged_contributors_percent_2022-08-04_15_0.7_1000_strong.csv"
si_weak_03_filename = "l15/weak_03/averaged_contributors_percent_2022-07-30_15_0.3_1000_weak.csv"
si_weak_07_filename = "l15/weak_07/averaged_contributors_percent_2022-08-01_15_0.7_1000_weak.csv"

if __name__ == "__main__":
    base03 = np.loadtxt(
        base03_filename,
        delimiter=","
    )
    base07 = np.loadtxt(
        base07_filename,
        delimiter=","
    )
    si_strong_03 = np.loadtxt(
        si_strong_03_filename,
        delimiter=","
    )
    si_strong_07 = np.loadtxt(
        si_strong_07_filename,
        delimiter=","
    )
    si_weak_03 = np.loadtxt(
        si_weak_03_filename,
        delimiter=","
    )
    si_weak_07 = np.loadtxt(
        si_weak_07_filename,
        delimiter=","
    )
    fig, axs = plt.subplots(2, 3, sharex="all", sharey="all", figsize=(12, 6))
    Ngrid = 101
    syn = np.linspace(0, 10, Ngrid)  # grid nodes for synergy
    pre = np.linspace(0, 10, Ngrid)  # grid nodes for pressure
    cp = axs[0, 0].contourf(syn, pre, base03, levels=np.linspace(0, 100, 11))
    axs[0, 0].set_title("New Base, d = 0.3")
    cp = axs[1, 0].contourf(syn, pre, base07, levels=np.linspace(0, 100, 11))
    axs[1, 0].set_title("New Base, d = 0.7")
    cp = axs[0, 1].contourf(syn, pre, si_weak_03, levels=np.linspace(0, 100, 11))
    axs[0, 1].set_title("SI Low, d = 0.3")
    cp = axs[0, 2].contourf(syn, pre, si_strong_03, levels=np.linspace(0, 100, 11))
    axs[0, 2].set_title("SI High, d = 0.3")
    cp = axs[1, 1].contourf(syn, pre, si_weak_07, levels=np.linspace(0, 100, 11))
    axs[1, 1].set_title("SI Low, d = 0.7")
    cp = axs[1, 2].contourf(syn, pre, si_strong_07, levels=np.linspace(0, 100, 11))
    axs[1, 2].set_title("SI High, d = 0.7")

    fig.text(0.5, 0.04, 'synergy', ha='center')
    fig.text(0.006666666667, 0.5, 'pressure', va='center', rotation='vertical')

    fig.subplots_adjust(left=0.0533333333, right=0.865+0.0533333333333)
    cbar_ax = fig.add_axes([0.9533333333333, 0.15, 0.01, 0.7])
    fig.colorbar(cp, cax=cbar_ax)
    plt.savefig('plot6.png', dpi=300)
    plt.show()
