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
            field.move_from(row, col)  # the agent moves when it changes the behaviour
        elif contributor[row, col] < non_contributor[row, col] and field[row, col] == effort:
            field[row, col] = 0
            field.move_from(row, col)  # the agent moves when it changes the behaviour


def simulate_base_model(
        length: int,
        density: float,
        initial_percent: float,
        effort: int,
        pressure: float,
        synergy: float,
        tick_max: int = 200,
        show_plot_every: int = 0,
        use_groups: bool = False,
        stop_on_adoption=True
):
    field = TorusLattice(length, contrib_value=effort)
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
    percent_of_contributors = []
    not_focal_agent_threat_to_self_not_threat_group_freq = []
    focal_agent_threat_to_self_not_threat_group_freq = []
    not_focal_agent_threat_to_self_threat_group_freq = []
    focal_agent_threat_to_self_threat_group_freq = []
    tick = 0
    while (0 < effort_agents_number < population or not stop_on_adoption) and tick < tick_max:
        agents_list = field.nonempty()
        contributor = np.zeros((length, length))
        non_contributor = np.zeros((length, length))
        not_focal_agent_threat_to_self_not_threat_group = {0: 0, effort: 0}
        focal_agent_threat_to_self_not_threat_group = {0: 0, effort: 0}
        not_focal_agent_threat_to_self_threat_group = {0: 0, effort: 0}
        focal_agent_threat_to_self_threat_group = {0: 0, effort: 0}
        for agent in agents_list:
            row, col = agent
            group = field.nonempty_in_radius(row, col, radius=1)
            agents_in_radius = len(group)
            focal_agent_benefit = synergy * effort * field.nonempty_number_in_radius(row, col, radius=1, value=effort) / agents_in_radius
            is_focal_agent_threat_to_self = (
                    field[row, col] == effort and focal_agent_benefit <= pressure or
                    field[row, col] == 0 and effort + focal_agent_benefit <= pressure
            )
            # Calculate each group member’s (including the focal agent’s) retained effort and benefit-from-group.
            retained_efforts_and_benefits = []
            for group_agent in group:
                group_agent_row, group_agent_col = group_agent
                agents_with_effort = len([1 for (r, c) in group if field[r, c] == effort])
                benefit = synergy * effort * agents_with_effort / agents_in_radius
                effort_and_benefit = benefit + (0 if field[group_agent_row, group_agent_col] == effort else effort)
                retained_efforts_and_benefits.append(effort_and_benefit)
            # Calculate the group’s average retained effort and benefit-from-group
            retained_efforts_and_benefits_average = sum(retained_efforts_and_benefits) / len(retained_efforts_and_benefits)
            threat_to_group = (retained_efforts_and_benefits_average <= pressure)
            if not is_focal_agent_threat_to_self and not threat_to_group:
                not_focal_agent_threat_to_self_not_threat_group[field[row, col]] += 1
            elif is_focal_agent_threat_to_self and not threat_to_group:
                focal_agent_threat_to_self_not_threat_group[field[row, col]] += 1
            elif not is_focal_agent_threat_to_self and threat_to_group:
                not_focal_agent_threat_to_self_threat_group[field[row, col]] += 1
            else:
                focal_agent_threat_to_self_threat_group[field[row, col]] += 1
        # change behaviour and move
        if use_groups:
            potentially_changing_behavior_groups(field, effort, synergy, pressure)
        else:
            potentially_moving(field, effort, synergy, pressure)
            potentially_changing_behavior(field, effort, synergy, pressure)
        effort_agents_number = len(field.nonempty(value=effort))
        percent_of_contributors.append(100 * effort_agents_number / population)
        not_focal_agent_threat_to_self_not_threat_group_freq.append(not_focal_agent_threat_to_self_not_threat_group)
        focal_agent_threat_to_self_not_threat_group_freq.append(focal_agent_threat_to_self_not_threat_group)
        not_focal_agent_threat_to_self_threat_group_freq.append(not_focal_agent_threat_to_self_threat_group)
        focal_agent_threat_to_self_threat_group_freq.append(focal_agent_threat_to_self_threat_group)
        if show_plot_every > 0 and tick % show_plot_every == 0:
            field.plot(f"The Social Space # {tick}")
        tick += 1
    return percent_of_contributors, not_focal_agent_threat_to_self_not_threat_group_freq, focal_agent_threat_to_self_not_threat_group_freq, not_focal_agent_threat_to_self_threat_group_freq, focal_agent_threat_to_self_threat_group_freq
