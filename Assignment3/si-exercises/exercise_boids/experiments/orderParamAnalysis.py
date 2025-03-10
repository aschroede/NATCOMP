import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def main(data_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    order_df = pd.read_csv(os.path.join(data_dir, "orderParameter.csv"))
    nn_df = pd.read_csv(os.path.join(data_dir, "nearestNeighborParameter.csv"), header=0)

    # Ensure proper column names assignment
    #nn_df.columns = ["time"] + [f"nn_{i}" for i in range(1, len(nn_df.columns))]

    # Plot order parameter over time
    plt.figure(figsize=(8, 5))
    plt.plot(order_df["time"], order_df["orderParameter"], marker='o', linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Order Parameter")
    plt.title("Order Parameter Over Time")
    plt.grid()
    plt.savefig(os.path.join(output_dir, "order_parameter_over_time.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot order parameter and nearest neighbor distributions.")
    parser.add_argument("data_dir", type=str, help="Directory containing the data files")
    parser.add_argument("output_dir", type=str, help="Directory to save the output figures")
    args = parser.parse_args()
    
    main(args.data_dir, args.output_dir)
