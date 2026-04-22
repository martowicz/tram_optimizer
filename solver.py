import os
from dataclasses import dataclass

import pandas as pd
import pulp

from config import ALPHA, BETA, C, R
from data_loader import load_data_as_matrices


@dataclass
class SolverResult:
    objective_value: float
    total_waiting: int
    total_fleet: int
    frequencies: dict[int, int]
    fleets: dict[int, int]


def _build_and_solve_model(P, num_stops: int, num_hours: int) -> SolverResult:
    od_pairs = [(s, d) for s in range(num_stops) for d in range(s + 1, num_stops)]
    hours = list(range(num_hours))

    problem = pulp.LpProblem("tram_frequency_optimization", pulp.LpMinimize)

    f = pulp.LpVariable.dicts("f", hours, lowBound=0, cat=pulp.LpInteger)
    v = pulp.LpVariable.dicts("V", hours, lowBound=0, cat=pulp.LpInteger)
    b = pulp.LpVariable.dicts("b", (range(num_stops), range(num_stops), hours), lowBound=0, cat=pulp.LpInteger)
    w = pulp.LpVariable.dicts("w", (range(num_stops), range(num_stops), hours), lowBound=0, cat=pulp.LpInteger)

    problem += (
        ALPHA * pulp.lpSum(w[s][d][t] for (s, d) in od_pairs for t in hours)
        + BETA * pulp.lpSum(v[t] for t in hours)
    )

    for s, d in od_pairs:
        for t in hours:
            prev_waiting = w[s][d][t - 1] if t > 0 else 0
            problem += w[s][d][t] == prev_waiting + int(P[s, d, t]) - b[s][d][t]

    for segment_end in range(num_stops - 1):
        for t in hours:
            onboard_on_segment = pulp.lpSum(
                b[i][d][t]
                for i in range(segment_end + 1)
                for d in range(segment_end + 1, num_stops)
            )
            problem += onboard_on_segment <= f[t] * C

    for t in hours:
        problem += 60 * v[t] >= f[t] * R

    solve_status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[solve_status] != "Optimal":
        raise RuntimeError(f"Solver did not find an optimal solution. Status: {pulp.LpStatus[solve_status]}")

    frequencies = {t + 1: int(pulp.value(f[t])) for t in hours}
    fleets = {t + 1: int(pulp.value(v[t])) for t in hours}
    total_waiting = int(sum(pulp.value(w[s][d][t]) for s, d in od_pairs for t in hours))
    total_fleet = int(sum(fleets.values()))
    objective_value = float(pulp.value(problem.objective))

    return SolverResult(
        objective_value=objective_value,
        total_waiting=total_waiting,
        total_fleet=total_fleet,
        frequencies=frequencies,
        fleets=fleets,
    )


def _forward_travel_cumsum(W, num_stops: int) -> list[int]:
    cumulative = [0]
    for stop in range(num_stops - 1):
        travel_time = int(W[stop, stop + 1])
        if travel_time <= 0:
            raise ValueError(f"Missing or invalid travel time between stop {stop + 1} and {stop + 2}.")
        cumulative.append(cumulative[-1] + travel_time)
    return cumulative


def _format_clock(total_minutes: int) -> str:
    hh = (total_minutes // 60) % 24
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"


def _generate_timetable(frequencies: dict[int, int], cumulative_forward_minutes: list[int]) -> pd.DataFrame:
    rows = []
    trip_id = 1

    for hour in sorted(frequencies):
        trips_in_hour = frequencies[hour]
        if trips_in_hour <= 0:
            continue

        interval = 60 / trips_in_hour
        hour_start_minute = (hour - 1) * 60

        for k in range(trips_in_hour):
            departure_minute = int(round(hour_start_minute + k * interval))

            for stop_idx, offset in enumerate(cumulative_forward_minutes, start=1):
                minute_at_stop = departure_minute + offset
                rows.append(
                    {
                        "TripID": trip_id,
                        "HourWindow_t": hour,
                        "Stop": stop_idx,
                        "Time_min_from_start_of_day": minute_at_stop,
                        "ClockTime": _format_clock(minute_at_stop),
                    }
                )

            trip_id += 1

    return pd.DataFrame(rows)


def _save_outputs(result: SolverResult, timetable: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    summary_df = pd.DataFrame(
        [
            {"Metric": "ObjectiveValue", "Value": result.objective_value},
            {"Metric": "TotalWaitingPassengers", "Value": result.total_waiting},
            {"Metric": "TotalFleetAcrossHours", "Value": result.total_fleet},
            {"Metric": "Alpha", "Value": ALPHA},
            {"Metric": "Beta", "Value": BETA},
            {"Metric": "Capacity_C", "Value": C},
            {"Metric": "CycleTime_R", "Value": R},
        ]
    )

    frequency_df = pd.DataFrame(
        [
            {
                "TimeWindow_t": hour,
                "Frequency_f_t": result.frequencies[hour],
                "Fleet_V_t": result.fleets[hour],
            }
            for hour in sorted(result.frequencies)
        ]
    )

    summary_df.to_csv(os.path.join(output_dir, "cost.csv"), index=False)
    frequency_df.to_csv(os.path.join(output_dir, "frequency_and_fleet.csv"), index=False)
    timetable.to_csv(os.path.join(output_dir, "timetable.csv"), index=False)


def solve_and_export(
    num_stops: int = 5,
    num_hours: int = 24,
    output_dir: str = "data/output",
) -> SolverResult:
    P, W = load_data_as_matrices(num_stops=num_stops, num_hours=num_hours)
    result = _build_and_solve_model(P=P, num_stops=num_stops, num_hours=num_hours)
    cumulative_forward_minutes = _forward_travel_cumsum(W=W, num_stops=num_stops)
    timetable = _generate_timetable(result.frequencies, cumulative_forward_minutes)
    _save_outputs(result=result, timetable=timetable, output_dir=output_dir)
    return result


if __name__ == "__main__":
    solve_and_export()
