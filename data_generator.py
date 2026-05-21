import pandas as pd
import random
import os

def generate_dummy_data(num_stops=5, num_hours=24):
    
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)

    
    travel_times = []
    for s in range(1, num_stops):
        time_minutes = random.randint(2, 5)
        travel_times.append({
            "Origin": s,
            "Destination": s + 1,
            "TravelTime_min": time_minutes
        })
    
    df_travel = pd.DataFrame(travel_times)
    df_travel.to_csv("data/input/travel_times.csv", index=False)

    demand_data = []
    
    for t in range(1, num_hours + 1):
        if t%24 in [7, 8, 9]:
            peak_multiplier = 3.0
        elif t%24 in [15, 16, 17]:
            peak_multiplier = 2.5
        elif t%24 in [23, 24, 1, 2, 3, 4]:
            peak_multiplier = 0.1
        else:
            peak_multiplier = 1.0

        for s in range(1, num_stops):
            for d in range(s + 1, num_stops + 1):
                
                base_demand = random.randint(10, 40)
                
                distance_penalty = 1.0 / (d - s)
                
                final_demand = int(base_demand * peak_multiplier * distance_penalty)
                
                demand_data.append({
                    "TimeWindow_t": t,
                    "Origin_s": s,
                    "Destination_d": d,
                    "Demand_P": final_demand
                })

    df_demand = pd.DataFrame(demand_data)
    df_demand.to_csv("data/input/demand_od.csv", index=False)

if __name__ == "__main__":
    generate_dummy_data(num_stops=5, num_hours=24)