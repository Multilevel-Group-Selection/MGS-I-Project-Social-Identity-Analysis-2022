import random

import numpy as np

from lattice.torus import TorusLattice


def potentially_moving(field: TorusLattice, effort: int, synergy: float, pressure: float):
    agents_list = field.nonempty()
    for agent in agents_list:
        row, col = agent
        if field[row, col] == effort:
            # contributors that are going to leave the group
            if synergy * effort * field.nonempty_number_in_radius(row, col, radius=1, value=effort) / \
                    field.nonempty_number_in_radius(row, col, radius=1) <= pressure:
                field.move_from(row, col)
        else:
            # non-contributors that are going to leave the group
            if effort + synergy * effort * field.nonempty_number_in_radius(row, col, radius=1, value=effort) / \
                    field.nonempty_number_in_radius(row, col, radius=1) <= pressure:
                field.move_from(row, col)


def potentially_changing_behavior(field: TorusLattice, effort: int, synergy: float, pressure: float):
    agents_list = field.nonempty()
    for agent in agents_list:
        row, col = agent
        if field.nonempty_number_in_radius(row, col, radius=1) == 1:
            if effort <= pressure:
                if field[row, col] == effort:
                    field[row, col] = 0
                else:
                    field[row, col] = effort
        else:
            if field[row, col] == effort:
                # Vulnerable contributors that are in a group reconsider
                if synergy * effort * field.nonempty_number_in_radius(row, col, radius=1, value=effort) / \
                        field.nonempty_number_in_radius(row, col, radius=1) <= pressure:
                    field[row, col] = 0
            else:
                # Vulnerable non-contributors that are in a group reconsider
                if effort + synergy * effort * field.nonempty_number_in_radius(row, col, radius=1, value=effort) / \
                        field.nonempty_number_in_radius(row, col, radius=1) <= pressure:
                    field[row, col] = effort


def potentially_changing_behavior_groups(field: TorusLattice, effort: int, synergy: float, pressure: float):
    agents_list = field.nonempty()
    length = field.order
    contributor = np.zeros((length, length))
    non_contributor = np.zeros((length, length))
    for agent in agents_list:
        row, col = agent
        if field.nonempty_number_in_radius(row, col, radius=1) == 1:
            if effort <= pressure:
                if field[row, col] == effort:
                    non_contributor[row, col] += 1
                else:
                    contributor[row, col] += 1
            else:
                if field[row, col] == effort:
                    contributor[row, col] += 1
                else:
                    non_contributor[row, col] += 1
        else:
            group = field.nonempty_in_radius(row, col, radius=1)
            for group_agent in group:
                group_agent_row, group_agent_col = group_agent
                aggents_with_effort = len([1 for (r, c) in group if field[r, c] == field[group_agent_row, group_agent_col]])
                benefit = synergy * effort * aggents_with_effort / len(group)
                if field[group_agent_row, group_agent_col] == effort:
                    # Vulnerable contributors that are in a group reconsider
                    if benefit <= pressure:
                        non_contributor[group_agent_row, group_agent_col] += 1
                    else:
                        contributor[group_agent_row, group_agent_col] += 1
                else:
                    # Vulnerable non-contributors that are in a group reconsider
                    if effort + benefit <= pressure:
                        contributor[group_agent_row, group_agent_col] += 1
                    else:
                        non_contributor[group_agent_row, group_agent_col] += 1
    for agent in agents_list:
        row, col = agent
        if contributor[row, col] > non_contributor[row, col] and field[row, col] == 0:
            field[row, col] = effort
        elif contributor[row, col] < non_contributor[row, col] and field[row, col] == effort:
            field[row, col] = 0


def simulate_base_model(
        length: int,
        density: float,
        initial_percent: float,
        effort: int,
        pressure: int,
        synergy: int,
        tick_max: int = 200,
        show_plot_every: int = 0,
        use_groups: bool = False
):
    field = TorusLattice(length)
    population, contrib_initial = field.land_agents(
        density=density,
        initial_percent=initial_percent,
        agent_value=0,
        effort=effort
    )
    # print(f"{population} agents, {contrib_initial} contributors in the initial social space")
    if show_plot_every > 0:
        field.plot("The Initial Social Space")
    # Simulation
    effort_agents_number = len(field.nonempty(value=effort))
    percent_of_contributors = [effort_agents_number / population]
    tick = 0
    while 0 < effort_agents_number < population and tick < tick_max:
        potentially_moving(field, effort, synergy, pressure)
        if use_groups:
            potentially_changing_behavior_groups(field, effort, synergy, pressure)
        else:
            potentially_changing_behavior(field, effort, synergy, pressure)
        effort_agents_number = len(field.nonempty(value=effort))
        percent_of_contributors.append(100 * effort_agents_number / population)
        if show_plot_every > 0 and tick % show_plot_every == 0:
            field.plot(f"The Social Space # {tick}")
        tick += 1
    return percent_of_contributors
