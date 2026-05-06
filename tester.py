import os
import matplotlib.pyplot as plt
from data_generator import generate_dummy_data
from data_loader import load_data_as_matrices
from timetable_phase import generate_timetable
from annealing_helpers import build_problem_data, run_schedule_optimization

# Get timetable df for given parameters
def get_result(C, R, alpha, beta, P, W):
    data = build_problem_data(P, C, R, alpha, beta)

    initial_temp = 10.0
    min_temp = 1e-3
    cooling_rate = 0.999
    max_iter = 1000
    log_every = 10

    results = run_schedule_optimization(data, initial_temp, min_temp, cooling_rate, max_iter, log_every)

    return generate_timetable(results, W)

# Save chart for given timetable
def save_chart(dir, timetable_df, C, R, alpha, beta):
    file_name = f"C{C}_R{R}_alpha{alpha}_beta{beta}.png"
    file_path = dir + file_name
    title = f"C = {C}, R = {R}, alpha = {alpha}, beta = {beta}"

    df = timetable_df.groupby("TripID").max().groupby("HourWindow").count()
    df = df.rename(columns={"Stop": "trips"})

    fig, ax = plt.subplots()
    df.plot.bar(y="trips", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("hour")
    ax.set_ylabel("trips in hour")
    plt.savefig(file_path)
    plt.close(fig)

# Run tests for different parameters
def test_and_chart(cs, rs, alphas, betas, P, W):
    dir = "test_charts/"
    os.makedirs(dir, exist_ok=True)
    for C in cs:
        for R in rs:
            for alpha in alphas:
                for beta in betas:
                    timetable_df = get_result(C, R, alpha, beta, P, W)
                    save_chart(dir, timetable_df, C, R, alpha, beta)


if __name__ == "__main__":
    num_stops = 5
    num_hours = 24
    generate_dummy_data(num_stops, num_hours)
    P, W = load_data_as_matrices(num_stops, num_hours)

    cs = [40, 50]
    rs = [15, 30]
    alphas = [1.0, 2.0, 5.0]
    betas = [5.0, 50.0, 100.0]
    test_and_chart(cs, rs, alphas, betas, P, W)
