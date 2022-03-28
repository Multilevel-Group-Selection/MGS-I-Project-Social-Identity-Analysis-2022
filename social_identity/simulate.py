import random

import numpy as np

from lattice.torus import TorusLattice


def simulate_social_identity_model(
        length: int,
        density: float,
        initial_percent: float,
        effort: int,
        pressure: int,
        synergy: int,
        use_strong_commitment: bool = True,
        tick_max: int = 200,
        show_plot_every: int = 0
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
    contributors_number = len(field.nonempty(value=effort))
    percent_of_contributors = [contributors_number / population]
    tick = 0
    while 0 < contributors_number < population and tick < tick_max:
        agents_list = field.nonempty()
        contributor = np.zeros((length, length))
        non_contributor = np.zeros((length, length))
        for agent in agents_list:
            row, col = agent
            group = field.nonempty_in_radius(row, col, radius=1)
            agents_in_radius = len(group)
            neighbors_number = agents_in_radius - 1
            focal_agent_benefit = synergy * effort * field.nonempty_number_in_radius(row, col, radius=1,
                                                                                     value=effort) / agents_in_radius
            focal_agent_effort_and_benefit = field[row, col] + focal_agent_benefit
            is_focal_agent_threat_to_self = (focal_agent_effort_and_benefit <= pressure)
            if neighbors_number <= 1:
                # If the agent’s number of neighbors is ≤ 1, follow the base model’s existing logic for movement and type change.
                if is_focal_agent_threat_to_self:
                    new_row, new_col = field.move_from(row, col)
                    contributor[new_row, new_col] = contributor[row, col]
                    non_contributor[new_row, new_col] = non_contributor[row, col]
                    contributor[row, col] = 0
                    non_contributor[row, col] = 0
            else:  # If the agent’s number of neighbors is > 1
                # Calculate each group member’s (including the focal agent’s) retained effort and benefit-from-group.
                retained_efforts_and_benefits = []
                benefit_from_group = []
                threat_to_self_values = []
                for group_agent in group:
                    group_agent_row, group_agent_col = group_agent
                    # benefit = synergy * effort * field.nonempty_number_in_radius(group_agent_row, group_agent_col, radius=1, value=effort) / field.nonempty_number_in_radius(group_agent_row, group_agent_col, radius=1)
                    aggents_with_effort = len(
                        [1 for (r, c) in group if field[r, c] == field[group_agent_row, group_agent_col]])
                    benefit = synergy * effort * aggents_with_effort / agents_in_radius
                    effort_and_benefit = field[group_agent_row, group_agent_col] + benefit
                    retained_efforts_and_benefits.append(effort_and_benefit)
                    benefit_from_group.append(benefit)
                    threat_to_self_values.append(
                        effort_and_benefit <= pressure
                    )
                # Calculate the group’s average retained effort and benefit-from-group
                retained_efforts_and_benefits_average = sum(retained_efforts_and_benefits) / len(
                    retained_efforts_and_benefits)
                benefit_from_group_average = sum(benefit_from_group) / len(benefit_from_group)
                threat_to_group = (retained_efforts_and_benefits_average <= pressure)
                members_need_help = any(threat_to_self_values) and (field[row, col] == 0)  # True if at least one group's memeber "retained effort and benefit" below the pressure level
                if use_strong_commitment:
                    # strong conditions
                    # change behavior
                    if (not is_focal_agent_threat_to_self and not threat_to_group) or (is_focal_agent_threat_to_self and not threat_to_group) or (not is_focal_agent_threat_to_self and threat_to_group and members_need_help):
                        if field[row, col] == 0:
                            contributor[row, col] += 1  # non-contributor -> contributor
                        else:
                            non_contributor[row, col] += 1  # contributor -> non-contributor
                    is_focal_agent_leads_to_leave_group = (is_focal_agent_threat_to_self and threat_to_group)
                else:
                    # weak conditions
                    if is_focal_agent_threat_to_self and not threat_to_group:
                        # Change behavior if not group’s
                        if field[row, col] == 0:
                            contributor[row, col] += 1  # non-contributor -> contributor
                        else:
                            non_contributor[row, col] += 1  # contributor -> non-contributor
                    is_focal_agent_leads_to_leave_group = threat_to_group
                if not is_focal_agent_leads_to_leave_group:
                    for group_agent, threat_to_self in zip(group, threat_to_self_values):
                        group_agent_row, group_agent_col = group_agent
                        members_need_help = any(threat_to_self_values) and (
                                    field[group_agent_row, group_agent_col] == 0)
                        if group_agent_row != row or group_agent_col != col:
                            # the agent in not focal in the group
                            if use_strong_commitment:
                                # strong
                                if threat_to_group and threat_to_self:
                                    # If the group member’s behavior resulted in a move, then remove/hide that member from that group but do not actually move them (since they are a core member in the neighboring group)
                                    continue
                                if (not threat_to_self and not threat_to_group) or (threat_to_self and not threat_to_group) or (not threat_to_self and threat_to_group and members_need_help):
                                    # Change behavior if not group’s
                                    if field[group_agent_row, group_agent_col] == 0:
                                        contributor[group_agent_row, group_agent_col] += 1  # non-contributor -> contributor
                                    else:
                                        non_contributor[group_agent_row, group_agent_col] += 1  # contributor -> non-contributor
                            else:
                                # weak
                                if threat_to_group:
                                    # If the group member’s behavior resulted in a move, then remove/hide that member from that group but do not actually move them (since they are a core member in the neighboring group)
                                    continue
                                elif threat_to_self and not threat_to_group:
                                    # Change behavior if not group’s
                                    if field[group_agent_row, group_agent_col] == 0:
                                        contributor[group_agent_row, group_agent_col] += 1  # non-contributor -> contributor
                                    else:
                                        non_contributor[group_agent_row, group_agent_col] += 1  # contributor -> non-contributor
                else:
                    new_row, new_col = field.move_from(row, col)
                    contributor[new_row, new_col] = contributor[row, col]
                    non_contributor[new_row, new_col] = non_contributor[row, col]
                    contributor[row, col] = 0
                    non_contributor[row, col] = 0
        # change behaviour of agents
        # Determine each agent’s most frequent type (contributor vs. non-contributor) based on all the groups it’s in (ignoring an agent’s type in the group(s) that it’s been removed/hidden from).
        to_contrib = 0
        to_noncontrib = 0
        agents_list = field.nonempty()
        for agent in agents_list:
            row, col = agent
            group = field.nonempty_in_radius(row, col, radius=1)
            agents_in_radius = len(group)
            neighbors_number = agents_in_radius - 1
            if neighbors_number <= 1:
                # If the agent’s number of neighbors is ≤ 1, follow the base model’s existing logic for movement and type change
                if field.nonempty_number_in_radius(row, col, radius=1) == 1:
                    if effort <= pressure:
                        if field[row, col] == effort:
                            field[row, col] = 0
                            to_contrib += 1
                        else:
                            field[row, col] = effort
                            to_contrib += 1
                else:
                    if field[row, col] + synergy * effort * field.nonempty_number_in_radius(row, col, radius=1, value=effort) / field.nonempty_number_in_radius(row, col, radius=1) <= pressure:
                        if field[row, col] == effort:
                            field[row, col] = 0
                            to_contrib += 1
                        else:
                            field[row, col] = effort
                            to_contrib += 1
            else:
                # If the agent’s number of neighbors is > 1
                if contributor[row, col] > non_contributor[row, col] and field[row, col] == 0:
                    field[row, col] = effort
                    to_contrib += 1
                elif contributor[row, col] < non_contributor[row, col] and field[row, col] == effort:
                    field[row, col] = 0
                    to_noncontrib += 1
        # calc the percentage of contributing agents
        contributors_number = len(field.nonempty(value=effort))
        percent_of_contributors.append(100 * contributors_number / population)
        if show_plot_every > 0 and tick % show_plot_every == 0:
            field.plot(f"The Social Space # {tick}")
        tick += 1
    return percent_of_contributors