
import enum
import pandas as pd
import numpy as np


def generate_timetable(results, W_matrix):


    best_f = results["best_f"]
    num_stops = W_matrix.shape[0]

    cumulative_offsets = [0.0]
    for i in range(num_stops-1):
        cumulative_offsets.append(cumulative_offsets[-1] + W_matrix[i,i+1])
    
    total_trips = sum(best_f)

    if total_trips == 0:
        return pd.DataFrame()
    
    cumulative_trips_at_hour = [0.0]
    current_sum = 0.0
    for f_t in best_f:
        current_sum+=f_t
        cumulative_trips_at_hour.append(current_sum)

    hours = len(best_f)

    departure_minutes = []
    for k in range(total_trips):
        target = k + 0.5

        for t in range(hours):
            if cumulative_trips_at_hour[t] <= target < cumulative_trips_at_hour[t+1]:
                trips_in_hour = best_f[t]
                fraction = (target - cumulative_trips_at_hour[t]) / trips_in_hour
                exact_min = (t*60) + (fraction * 60)
                departure_minutes.append(int(round(exact_min)))
                break

    schedule_data = []

    for trip_id, start_min in enumerate(departure_minutes, start=1):
        for stop_idx, offset in enumerate(cumulative_offsets, start=1):
            arrival_time = start_min + offset

            hh = int((arrival_time // 60) % 24)
            mm = int(arrival_time % 60)
            
            schedule_data.append({
                "TripID": trip_id,
                "Stop": stop_idx,
                "DepartureTime_Min": arrival_time,
                "ClockTime": f"{hh:02d}:{mm:02d}",
                "HourWindow": (start_min // 60)
            })
    return pd.DataFrame(schedule_data)