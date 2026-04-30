import math
import random
import copy


def simulated_annealing(
    initial_state,
    energy_fn,
    neighbor_fn,
    initial_temp=2.0,
    min_temp=1e-4,
    cooling_rate=0.99,
    max_iter=100_000,
):
    current_state = initial_state
    temperature = initial_temp
    current_energy = energy_fn(current_state)

    for _ in range(max_iter):
        if temperature < min_temp:
            break

        new_state = neighbor_fn(current_state)
        new_energy = energy_fn(new_state)
        delta_energy = new_energy - current_energy

        if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
            current_state = new_state
            current_energy = new_energy

        temperature *= cooling_rate

    return current_state


def simulated_annealing_log(
    initial_state,
    energy_fn,
    neighbor_fn,
    initial_temp=2.0,
    min_temp=1e-4,
    cooling_rate=0.99,
    max_iter=100_000,
    log_interval=10,
):
    current_state = initial_state
    temperature = initial_temp
    current_energy = energy_fn(current_state)

    best_state = current_state
    best_energy = current_energy
    history_log = [(0, best_energy)]

    for step in range(1, max_iter + 1):
        if temperature < min_temp:
            break

        new_state = neighbor_fn(current_state)
        new_energy = energy_fn(new_state)
        delta_energy = new_energy - current_energy

        if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
            current_state = new_state
            current_energy = new_energy

        if current_energy < best_energy:
            best_energy = current_energy
            best_state = copy.deepcopy(current_state)

        if step % log_interval == 0:
            history_log.append((step, best_energy))

        temperature *= cooling_rate

    return best_state, history_log
