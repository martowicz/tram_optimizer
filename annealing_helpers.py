import math
import random
from simulated_annealing import simulated_annealing_log

class ProblemData:
    def __init__(self, P, C=40, R=30, alpha=1.0, beta=30.0):
        self.num_stops = len(P)
        self.num_time_windows = len(P[0][0]) if self.num_stops > 0 else 0
        self.num_segments = self.num_stops - 1

        self.P = P
        self.C = C
        self.R = R
        self.alpha = alpha
        self.beta = beta

        self.OD = [(s, d) for s in range(self.num_stops) for d in range(s + 1, self.num_stops)]


def build_problem_data(P, C=40, R=30, alpha=1.0, beta=30.0):
    return ProblemData(P, C, R, alpha, beta)

def evaluate_schedule(f_vec, data, return_details=False):
    if len(f_vec) != data.num_time_windows:
        raise ValueError("f_vec length must equal num_time_windows")

    wait_prev = {(s, d): 0.0 for s, d in data.OD}
    b_sol = {}
    w_sol = {}
    V_sol = {}

    total_wait = 0.0
    total_fleet = 0.0

    for t in range(data.num_time_windows):
        f_t = max(0, f_vec[t])
        seg_cap = [f_t * data.C] * data.num_segments
        wait_now = {}

        for s, d in data.OD:
            demand_sd = wait_prev[(s, d)] + data.P[s][d][t]
            segs = range(s, d)

            board_cap = min([seg_cap[seg] for seg in segs], default=0)
            boarded = min(demand_sd, board_cap)

            for seg in segs:
                seg_cap[seg] -= boarded

            waiting = demand_sd - boarded
            b_sol[(s, d, t)] = boarded
            w_sol[(s, d, t)] = waiting
            wait_now[(s, d)] = waiting

        V_t = math.ceil(f_t * data.R / 60)
        V_sol[t] = V_t

        total_wait += sum(wait_now.values())
        total_fleet += V_t
        wait_prev = wait_now

    obj = data.alpha * total_wait + data.beta * total_fleet

    if return_details:
        return {
            "objective": obj,
            "total_wait": total_wait,
            "total_fleet": total_fleet,
            "f": list(f_vec),
            "V": V_sol,
            "b": b_sol,
            "w": w_sol
        }

    return obj

def compute_hourly_demand(data):
    return [sum(data.P[s][d][t] for s, d in data.OD) for t in range(data.num_time_windows)]

def build_initial_frequency(data, hourly_demand):
    return [max(1, math.ceil(h / data.C)) for h in hourly_demand]

def run_schedule_optimization(data, initial_temp=5.0, min_temp=1e-3, cooling_rate=0.999,
                              max_iter=1000, log_interval=250,
                              freq_delta_choices=None, max_frequency=30):
    if freq_delta_choices is None:
        freq_delta_choices = [-2, -1, 1, 2]

    hourly_demand = compute_hourly_demand(data)
    f_init = build_initial_frequency(data, hourly_demand)

    def neighbor_frequency(f):
        f_new = list(f)
        idx = random.randrange(len(f_new))
        delta = random.choice(freq_delta_choices)
        f_new[idx] = max(0, min(f_new[idx] + delta, max_frequency))
        return f_new

    def energy_fn(f):
        return evaluate_schedule(f, data)

    best_f, log = simulated_annealing_log(
        f_init,
        energy_fn,
        neighbor_frequency,
        initial_temp =initial_temp,
        min_temp = min_temp,
        cooling_rate = cooling_rate,
        max_iter = max_iter,
        log_interval = log_interval
    )

    best_solution = evaluate_schedule(best_f, data, return_details=True)

    return {
        "best_f": best_f,
        "log": log,
        "best_solution": best_solution,
        "hourly_demand": hourly_demand,
        "f_init": f_init,
        "e_init": energy_fn(f_init),
    }

def print_results(results):
    print("                 --- Annealing Results ---\n")

    print("           Initial frequencies:   Hourly demands:   Best solution:")
    for i in range(len(results["f_init"])):
        print("Hour:", i, "         Freq:",results["f_init"][i], "         Demand:",results["hourly_demand"][i], "         Freq:", results["best_f"][i])

    print("\nInitial energy:", results["e_init"])
    print("Final energy:", results["best_solution"]["objective"])