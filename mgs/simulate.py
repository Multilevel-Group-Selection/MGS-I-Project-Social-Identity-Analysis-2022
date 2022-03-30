import random

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


def simulate_base_model(
        length: int,
        density: float,
        initial_percent: float,
        effort: int,
        pressure: int,
        synergy: int,
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
    effort_agents_number = len(field.nonempty(value=effort))
    percent_of_contributors = [effort_agents_number / population]
    tick = 0
    while 0 < effort_agents_number < population and tick < tick_max:
        potentially_moving(field, effort, synergy, pressure)
        potentially_changing_behavior(field, effort, synergy, pressure)
        effort_agents_number = len(field.nonempty(value=effort))
        percent_of_contributors.append(100 * effort_agents_number / population)
        if show_plot_every > 0 and tick % show_plot_every == 0:
            field.plot(f"The Social Space # {tick}")
        tick += 1
    return percent_of_contributors
