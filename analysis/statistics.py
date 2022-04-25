import numpy as np


def print_stats(contrib_space: np.ndarray, ticks_space: np.ndarray, epsilon: float = 10.0):
    contrib_idx = np.where(contrib_space > 100.0 - epsilon)
    non_contrib_idx = np.where(contrib_space < epsilon)
    contributors_number = len(contrib_idx[0])
    non_contributors_number = len(non_contrib_idx[0])
    mean_run_contributors_percent = np.mean(contrib_space[np.where((epsilon <= contrib_space) & (contrib_space <= 100.0 - epsilon))])
    average_ticks_number = np.mean(ticks_space)
    contrib_ticks_number = ticks_space[contrib_idx]
    average_ticks_contrib_number = np.mean(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
    min_ticks_contrib_number = np.min(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
    max_ticks_contrib_number = np.max(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
    non_contrib_ticks_number = ticks_space[non_contrib_idx]
    average_ticks_non_contrib_number = np.mean(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
    min_ticks_non_contrib_number = np.min(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
    max_ticks_non_contrib_number = np.max(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
    print(f"number of synergy-pressure pairs with less than {epsilon} adoption: {non_contributors_number}")
    print(f"number of synergy-pressure pairs with more than {100.0 - epsilon} adoption: {contributors_number}")
    print(f"average percent of contributors excluding with adoption in [{epsilon}; {100.0 - epsilon}]: {mean_run_contributors_percent}")
    print(f"average number of ticks in the simulation: {average_ticks_number}")
    print(f"minimum number of ticks to get less than {epsilon} adoption: {min_ticks_non_contrib_number}")
    print(f"average number of ticks to get less than {epsilon} adoption: {average_ticks_non_contrib_number}")
    print(f"maximum number of ticks to get less than {epsilon} adoption: {max_ticks_non_contrib_number}")
    print(f"minimum number of ticks to get more than {100.0 - epsilon} adoption: {min_ticks_contrib_number}")
    print(f"average number of ticks to get more than {100.0 - epsilon} adoption: {average_ticks_contrib_number}")
    print(f"maximum number of ticks to get more than {100.0 - epsilon} adoption: {max_ticks_contrib_number}")
    print("---")
