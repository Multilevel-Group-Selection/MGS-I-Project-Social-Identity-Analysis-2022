import numpy as np


class SeriesStatistics:
    def __init__(self, epsilon: float):
        self.below_epsilon = 0
        self.upper_epsilon = 0
        self.non_outlier = 0
        self.number_of_series = 0
        self.epsilon = epsilon
        self.below_epsilon_starting_tick = []
        self.upper_epsilon_starting_tick = []
        self.non_outlier_percent = []

    def add_series(self, series: np.ndarray):
        self.number_of_series += 1
        if series[-1] < self.epsilon:
            self.below_epsilon += 1
            i = len(series) - 1
            while i >= 0 and series[i] < self.epsilon:
                i -= 1
            self.below_epsilon_starting_tick.append(i + 1)
        elif series[-1] > 100.0 - self.epsilon:
            self.upper_epsilon += 1
            i = len(series) - 1
            while i >= 0 and series[i] > 100.0 - self.epsilon:
                i -= 1
            self.upper_epsilon_starting_tick.append(i + 1)
        else:
            self.non_outlier += 1
            self.non_outlier_percent.append(series[-1])

    def print_report(self):
        print(f"number of synergy-pressure pairs with less than {self.epsilon}% adoption: {self.below_epsilon}")
        print(f"number of synergy-pressure pairs with more than {100.0 - self.epsilon}% adoption: {self.upper_epsilon}")
        print(f"number of synergy-pressure pairs with adoption in [{self.epsilon}%; {100.0 - self.epsilon}%]: {self.non_outlier}")
        print(f"average percent of contributors excluding with adoption in [{self.epsilon}%; {100.0 - self.epsilon}%]: {np.mean(self.non_outlier_percent)}")
        print(f"the standard deviation for the percent of contributors excluding with adoption in [{self.epsilon}%; {100.0 - self.epsilon}%]: {np.std(self.non_outlier_percent)}")
        print(f"minimum number of ticks to get the equilibrium with less than {self.epsilon}% adoption: {np.min(self.below_epsilon_starting_tick) if self.below_epsilon_starting_tick else None}")
        print(f"average number of ticks to get the equilibrium with less than {self.epsilon}% adoption: {np.mean(self.below_epsilon_starting_tick) if self.below_epsilon_starting_tick else None}")
        print(f"the standard deviation for the number of ticks to get the equilibrium with less than {self.epsilon}% adoption: {np.std(self.below_epsilon_starting_tick) if self.below_epsilon_starting_tick else None}")
        print(f"maximum number of ticks to get the equilibrium with less than {self.epsilon}% adoption: {np.max(self.below_epsilon_starting_tick) if self.below_epsilon_starting_tick else None}")
        print(f"minimum number of ticks to get the equilibrium with more than {100.0 - self.epsilon}% adoption: {np.min(self.upper_epsilon_starting_tick) if self.upper_epsilon_starting_tick else None}")
        print(f"average number of ticks to get the equilibrium with more than {100.0 - self.epsilon}% adoption: {np.mean(self.upper_epsilon_starting_tick) if self.upper_epsilon_starting_tick else None}")
        print(f"the standard deviation for the number of ticks to get the equilibrium with more than {100.0 - self.epsilon}% adoption: {np.std(self.upper_epsilon_starting_tick) if self.upper_epsilon_starting_tick else None}")
        print(f"maximum number of ticks to get the equilibrium with more than {100.0 - self.epsilon}% adoption: {np.max(self.upper_epsilon_starting_tick) if self.upper_epsilon_starting_tick else None}")
        print("---")


def print_stats(contrib_space: np.ndarray, ticks_space: np.ndarray, epsilon: float = 10.0):
    contrib_idx = np.where(contrib_space > 100.0 - epsilon)
    non_contrib_idx = np.where(contrib_space < epsilon)
    contributors_number = len(contrib_idx[0])
    non_contributors_number = len(non_contrib_idx[0])
    mean_run_contributors_percent = np.mean(contrib_space[np.where((epsilon <= contrib_space) & (contrib_space <= 100.0 - epsilon))])
    average_ticks_number = np.mean(ticks_space)
    std_ticks_number = np.std(ticks_space)
    contrib_ticks_number = ticks_space[contrib_idx]
    average_ticks_contrib_number = np.mean(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
    std_ticks_contrib_number = np.std(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
    min_ticks_contrib_number = np.min(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
    max_ticks_contrib_number = np.max(contrib_ticks_number) if len(contrib_ticks_number) > 0 else None
    non_contrib_ticks_number = ticks_space[non_contrib_idx]
    average_ticks_non_contrib_number = np.mean(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
    std_ticks_non_contrib_number = np.std(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
    min_ticks_non_contrib_number = np.min(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
    max_ticks_non_contrib_number = np.max(non_contrib_ticks_number) if len(non_contrib_ticks_number) > 0 else None
    print(f"number of synergy-pressure pairs with less than {epsilon} adoption: {non_contributors_number}")
    print(f"number of synergy-pressure pairs with more than {100.0 - epsilon} adoption: {contributors_number}")
    print(f"average percent of contributors excluding with adoption in [{epsilon}; {100.0 - epsilon}]: {mean_run_contributors_percent}")
    print(f"average number of ticks in the simulation: {average_ticks_number}")
    print(f"the standard deviation of the number of ticks in the simulation: {std_ticks_number}")
    print(f"minimum number of ticks to get less than {epsilon} adoption: {min_ticks_non_contrib_number}")
    print(f"average number of ticks to get less than {epsilon} adoption: {average_ticks_non_contrib_number}")
    print(f"the standard deviation of the number of ticks to get less than {epsilon} adoption: {std_ticks_non_contrib_number}")
    print(f"maximum number of ticks to get less than {epsilon} adoption: {max_ticks_non_contrib_number}")
    print(f"minimum number of ticks to get more than {100.0 - epsilon} adoption: {min_ticks_contrib_number}")
    print(f"average number of ticks to get more than {100.0 - epsilon} adoption: {average_ticks_contrib_number}")
    print(f"the standard deviation of the number of ticks to get more than {100.0 - epsilon} adoption: {std_ticks_contrib_number}")
    print(f"maximum number of ticks to get more than {100.0 - epsilon} adoption: {max_ticks_contrib_number}")
    print("---")
