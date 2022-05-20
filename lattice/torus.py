import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from typing import Optional

from matplotlib.animation import FuncAnimation


def get_animation(frames, title: str, vmin: int, vmax: int, map_values):
    fig, ax = plt.subplots()
    anim = FuncAnimation(
        fig,
        lambda f: plot_matrix_values(frames[f], f"{title}#{f}", vmin=vmin, vmax=vmax, map_values=map_values, fig=fig, ax=ax, is_show=False),
        repeat=False
    )
    return anim


def save_frames(frames, base_name: str, title: str, vmin: int, vmax: int, map_values):
    fig, ax = plt.subplots()
    for i, frame in enumerate(frames):
        plot_matrix_values(frame, f"{title}#{i}", vmin=vmin, vmax=vmax, map_values=map_values,
                           fig=fig, ax=ax, is_show=False)
        plt.savefig(f"{base_name}_{i}.png")


def plot_matrix_values(
        matrix: np.ndarray,
        title: str,
        cmap="Reds",
        vmin=1,
        vmax=4,
        xlabel="",
        ylabel="",
        xticks=None,
        yticks=None,
        map_values=None,
        fig=None,
        ax=None,
        is_show=True
):
    """
    This function plots the matrix printing values in cells
    :param matrix: numpy ndarray to be plotted
    :param title: title of the plot
    :param cmap: name of or reference to the colormap
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    else:
        ax.clear()
    rows, cols = matrix.shape
    ms = ax.matshow(matrix, cmap=plt.get_cmap(cmap, vmax - vmin + 1), origin='lower', vmin=vmin, vmax=vmax)
    if rows < 21:
        for i in range(rows):
            for j in range(cols):
                v = str(matrix[j, i]) if map_values is None else str(map_values[matrix[j, i]])
                ax.text(i, j, v, va='center', ha='center', fontsize=6)
    else:
        fig.colorbar(ms, ticks=np.arange(vmin, vmax + 1))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        plt.title(title, loc='center', wrap=True, fontsize=8)
    if xticks is not None:
        if len(xticks) < 21:
            t = np.arange(0, len(xticks), 1)
        else:
            t = np.linspace(0, len(xticks) - 1, round(len(xticks) / 10), endpoint=True, dtype=int)
        ax.set_xticks(t)
        ax.set_xticklabels(map("{:.1f}".format, xticks[t]), fontsize=7)
    if yticks is not None:
        if len(xticks) < 21:
            t = np.arange(0, len(yticks), 1)
        else:
            t = np.linspace(0, len(yticks) - 1, round(len(yticks) / 10), endpoint=True, dtype=int)
        ax.set_yticks(t)
        ax.set_yticklabels(map("{:.1f}".format, yticks[t]), fontsize=7)
    if is_show:
        plt.show()


def plot_matrix_colorbar(
        matrix: np.ndarray,
        title: str,
        cmap="viridis",
        edgecolors='k',
        linewidths=1,
        mark_values=True,
        xlabel="",
        ylabel="",
        x=None,
        y=None,
        vmin=None,
        vmax=None
):
    """
    This function plots the matrix using the color bar to denote colors of cells
    :param matrix: numpy ndarray to be plotted
    :param title: title of the plot
    :param cmap: name of or reference to the colormap
    :param linewidths: width of cell's edges
    :param edgecolors: color of cell's edges
    """
    unique = np.unique(matrix)
    if x is not None and y is not None:
        plt.pcolor(
            x,
            y,
            matrix,
            cmap=plt.get_cmap(cmap, len(unique)) if mark_values else cmap,
            edgecolors=edgecolors,
            linewidths=linewidths,
            vmin=vmin,
            vmax=vmax
        )
    else:
        plt.pcolor(
            matrix,
            cmap=plt.get_cmap(cmap, len(unique)) if mark_values else cmap,
            edgecolors=edgecolors,
            linewidths=linewidths,
            vmin=vmin,
            vmax=vmax
        )
    cbar = plt.colorbar()
    if mark_values:
        cbar.set_ticks(unique)
    if title:
        plt.title(title, loc='center', wrap=True, fontsize=8)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()


class TorusLattice:
    """
    This class is an abstraction of the torus lattice to model the social space
    """
    def __init__(self, order: int, empty_node_value: int = -1) -> None:
        """
        Constructs the squared lattice with order x order spots and initializes each by the empty_node_value value

        :param order: The number of spots per one side
        :param empty_node_value: The value for empty spots
        """
        self.order = order
        self.empty_node_value = empty_node_value
        self.field = np.full(shape=(order, order), fill_value=empty_node_value, dtype=np.integer)

    def __getitem__(self, key):
        r = key[0] % self.order
        c = key[1] % self.order
        return self.field[r, c]

    def __setitem__(self, key, value):
        r = key[0] % self.order
        c = key[1] % self.order
        self.field[r, c] = value

    def plot(self, title: str, print_values_in_cells: bool = True):
        """
        Plots the lattice

        :param title: The title of the plot
        :param print_values_in_cells: if True then the field is plotted printing values in cells
        :param edgecolors: Name of edge color
        :param linewidths: Width of spots borders
        """
        if print_values_in_cells:
            plot_matrix_values(
                np.array(self.field),
                title,
                vmin=self.field.min(),
                vmax=self.field.max(),
                map_values={
                    self.empty_node_value: "",
                    self.empty_node_value + 1: "n",
                    self.empty_node_value + 2: "c"
                }
            )
        else:
            plot_matrix_colorbar(np.array(self.field), title)

    def nonempty_number(self):
        """
        Calculates the number of nonempty cells

        :return: The number of nonempty cells
        """
        return sum(1 for r in self.field for c in r if c != self.empty_node_value)

    def nonempty_number_in_radius(self, row: int, col: int, radius: int, value: Optional[int] = None):
        """
        Calculates the number of nonempty neighbors cells for the (row, col) cell.
        Cells from the square [-radius + row; radius + row] x [-radius + col; radius + col] are neighbors.

        :param row: Index of the row
        :param col: Index of the column
        :param radius: The radius value
        :param value: If this values is not None then the number of cells containing values is returned
        :return: the number of nonempty neighbors cells
        """
        count = 0
        offset_list = list(range(-radius, radius + 1))
        for r in offset_list:
            for c in offset_list:
                i = (row + r) % self.order
                j = (col + c) % self.order
                e = self.field[i, j]
                if e != self.empty_node_value and (value is None or value == e):
                    count += 1
        return count

    def nonempty(self, value: Optional[int] = None, shuffle=True):
        """
        Returns a list of coordinates of nonempty neighbor cells for the (row, col) cell.
        If this values is not None then cells containing values are returned.

        :param value: Optional integer to filter cells
        :param shuffle: If true then the list is returned shuffled.
        :return: The list of coordinates
        """
        a = [(r, c) for r in range(self.order) for c in range(self.order) if
             (value is None and self.field[r][c] != self.empty_node_value) or self.field[r][c] == value]
        if shuffle:
            random.shuffle(a)
        return a

    def nonempty_in_radius(self, row: int, col: int, radius: int, shuffle=True):
        """
        Returns a list of coordinates of nonempty cells.
        Cells from the square [-radius + row; radius + row] x [-radius + col; radius + col] are neighbors.

        :param row: Index of the row
        :param col: Index of the column
        :param radius: The radius value
        :param shuffle: If true then the list is returned shuffled.
        :return: The list of coordinates
        """
        offset_list = list(range(-radius, radius + 1))
        agents = []
        for r in offset_list:
            for c in offset_list:
                i = (row + r) % self.order
                j = (col + c) % self.order
                e = self.field[i, j]
                if e != self.empty_node_value:
                    agents.append((i, j))
        if shuffle:
            random.shuffle(agents)
        return agents

    def move_from(self, row: int, col: int):
        """
        Moves the value from the specific cell the random closest empty cell.

        :param row: Index of the row
        :param col: Index of the column
        :return: new row, new column
        """
        if self.__getitem__((row, col)) != self.empty_node_value:
            for offset in range(1, int(self.order / 2)):
                can_move = []
                dirs = [(i, j) for i in range(-offset, offset + 1, 1) for j in range(-offset, offset + 1, 1)]
                for d in dirs:
                    x = (row + d[0]) % self.order
                    y = (col + d[1]) % self.order
                    if self.field[x, y] == self.empty_node_value:
                        can_move.append((x, y))
                if can_move:
                    x, y = random.choice(can_move)
                    self.field[x, y] = self.field[row, col]
                    self.field[row, col] = self.empty_node_value
                    return x, y
        return row, col

    def land_agents(self, density: float, initial_percent: float, agent_value: int = 0, effort: int = 1):
        spots_number = self.order * self.order
        population_size = round(density * spots_number)
        # Generate a random list of integers from the range(0; order**2 - 1) without duplicates.
        occupied_spots = random.sample(range(spots_number), population_size)
        # land agens in spots
        for oc in occupied_spots:
            self.field[int(oc / self.order), oc % self.order] = agent_value
        contrib_initial = round(initial_percent * population_size)
        # randomly select contributors
        for oc in occupied_spots[:contrib_initial]:
            self.field[int(oc / self.order), oc % self.order] = effort
        return population_size, contrib_initial
