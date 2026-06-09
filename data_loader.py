# plik: src/data_loader.py
import pandas as pd
import numpy as np

def load_data_as_matrices(travel_times_filepath: str, demand_filepath: str, num_stops=5, num_hours=24):
    """
    Wczytuje dane z CSV i konwertuje je na macierze NumPy.
    """
    
    df_travel = pd.read_csv(travel_times_filepath)
    W = np.zeros((num_stops, num_stops), dtype=int)
    
    for _, row in df_travel.iterrows():
        i = int(row['Origin']) - 1
        j = int(row['Destination']) - 1
        W[i, j] = int(row['TravelTime_min'])
        
    
    df_demand = pd.read_csv(demand_filepath)
    P = np.zeros((num_stops, num_stops, num_hours), dtype=int)
    
    for _, row in df_demand.iterrows():
        s = int(row['Origin_s']) - 1
        d = int(row['Destination_d']) - 1
        t = int(row['TimeWindow_t']) - 1
        P[s, d, t] = int(row['Demand_P'])

    print("✅ Pomyślnie załadowano dane wejściowe w odpowieniej formie numpy")
        
    return P, W


def load_stop_names(stop_names_filepath: str, num_stops):
    df_names = pd.read_csv(stop_names_filepath)
    stop_names_dictionary = dict(zip(df_names["stop_id"], df_names["stop_name"]))
    if len(stop_names_dictionary) != num_stops:
        print("❌ Ilość przystanków nie zgadza się z wprowadzonymi danymi")
    print("✅ Pomyślnie załadowano nazwy przystanków")
    return stop_names_dictionary
