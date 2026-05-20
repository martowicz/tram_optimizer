# plik: src/data_loader.py
import pandas as pd
import numpy as np

def load_data_as_matrices(input_file: str, num_stops=5, num_hours=24):
    """
    Wczytuje dane z CSV i konwertuje je na macierze NumPy.
    """
    
    df_travel = pd.read_csv(input_file)
    W = np.zeros((num_stops, num_stops), dtype=int)
    
    for _, row in df_travel.iterrows():
        i = int(row['Origin']) - 1
        j = int(row['Destination']) - 1
        W[i, j] = int(row['TravelTime_min'])
        
    
    df_demand = pd.read_csv("data/input/demand_od.csv")
    P = np.zeros((num_stops, num_stops, num_hours), dtype=int)
    
    for _, row in df_demand.iterrows():
        s = int(row['Origin_s']) - 1
        d = int(row['Destination_d']) - 1
        t = int(row['TimeWindow_t']) - 1
        P[s, d, t] = int(row['Demand_P'])
        
    return P, W

if __name__ == "__main__":
    P, W = load_data_as_matrices("data/input/travel_times.csv")
    print("Kształt macierzy W (przystanki, przystanki):", W.shape)
    print("Kształt macierzy P (przystanki, przystanki, godziny):", P.shape)
    print("Popyt z przystanku 0 do 2 w oknie czasowym 7:", P[0, 2, 7])
