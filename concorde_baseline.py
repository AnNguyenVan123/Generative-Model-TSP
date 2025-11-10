import numpy as np
import torch
import pickle
import os
import argparse
import networkx as nx
from networkx.algorithms import approximation as approx
from utils.local_search import tsp_length_batch

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Filename of the dataset to evaluate")
    opts = parser.parse_args()

    assert opts.data_path is not None, "Need to specify data path"

    # Read dataset
    name = os.path.splitext(os.path.basename(opts.data_path))[0]
    with open(opts.data_path, "rb") as f:
        dataset = pickle.load(f)
    positions = np.array(dataset)  # M x N x 2

    M, N, _ = positions.shape
    tours = np.zeros((M, N), dtype=np.int64)
    costs = []

    for i in range(M):
        coords = positions[i]  # shape: (N,2)

        # Build complete graph
        G = nx.complete_graph(N)
        for u in range(N):
            for v in range(u + 1, N):
                dist = np.linalg.norm(coords[u] - coords[v])
                G[u][v]['weight'] = dist
                G[v][u]['weight'] = dist

        # Solve TSP approximately
        tour = approx.traveling_salesman_problem(G, weight='weight', cycle=True)
        tour = np.array(tour[:-1])  # remove repeated start/end node
        # Ensure tour has exactly N nodes
        tour = np.array(tour)
        if len(tour) > N:
            tour = tour[:N]  # take only the first N nodes
        tours[i] = tour


        # Compute cost precisely
        cost = tsp_length_batch(torch.from_numpy(coords[None, :, :]), torch.from_numpy(tour[None, :]))
        costs.append(cost.item())

        print(f"Finished instance {i+1}/{M} with cost {cost.item()}")

    # Save results
    result_dir = f"results/tsp/{name}"
    os.makedirs(result_dir, exist_ok=True)

    with open(f"{result_dir}/concorde_costs.pkl", "wb") as f:
        pickle.dump(costs, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved NetworkX costs to {result_dir}/concorde_costs.pkl")

    with open(f"{result_dir}/concorde_tours.pkl", "wb") as f:
        pickle.dump(tours.tolist(), f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved NetworkX tours to {result_dir}/concorde_tours.pkl")
