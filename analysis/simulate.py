import numpy as np


class Simulation:
    def __init__(self, grid_size: int, runs_number: int):
        self.grid_size = grid_size
        self.runs_number = runs_number
        self.synergy = np.linspace(0, 10, self.grid_size)
        self.pressure = np.linspace(0, 10, self.grid_size)
        self.points_number = len(self.synergy)