import pandas as pd
from matplotlib.animation import FuncAnimation
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def load_data(filename):
    df = pd.read_csv(filename)
    df.columns = ['time', 'nn_distances']  # Rename columns to be clear
    df['nn_distances'] = df['nn_distances'].apply(lambda x: json.loads(x.strip('"')))
    return df   
    
def makeNnHistograms(df, output_dir):

    # Time intervals to plot (adjust these as needed)
    time_points = [0, 100, 300, 500, 1000]  # Selected time points to visualize

    # Create a directory for saving plots if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)  

    # Set up the figure and subplots
    fig, axes = plt.subplots(len(time_points), 1, figsize=(10, 3*len(time_points)))
    if len(time_points) == 1:
        axes = [axes]  # Make sure axes is always a list

    # Plot histogram for each selected time point
    for i, time_point in enumerate(time_points):
        # Find the row with the closest time point
        closest_row = df.iloc[(df['time'] - time_point).abs().argsort()[0]]
        actual_time = closest_row['time']
        nn_distances = closest_row['nn_distances']
        
        # Plot histogram
        axes[i].hist(nn_distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Nearest Neighbor Distance Distribution at Time {actual_time}')
        axes[i].set_xlabel('Distance')
        axes[i].set_ylabel('Frequency')
        
        # Add some statistics as text
        stats_text = (f"Mean: {np.mean(nn_distances):.2f}\n"
                    f"Median: {np.median(nn_distances):.2f}\n"
                    f"Min: {min(nn_distances):.2f}, Max: {max(nn_distances):.2f}")
        axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/nn_histograms_comparison.png", dpi=300)


def create_histogram_animation(df, output_dir):
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    def update(frame):
        ax.clear()
        row = df.iloc[frame]
        time = row['time']
        nn_distances = row['nn_distances']
        
        ax.hist(nn_distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'Nearest Neighbor Distance Distribution at Time {time}')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Frequency')
        #ax.set_xlim(0, df['nn_distances'].apply(max).max() * 1)  # Consistent x-axis
        #ax.set_ylim(0, df['nn_distances'].apply(len).max() * 1)  # Consistent y-axis
        
        stats_text = (f"Mean: {np.mean(nn_distances):.2f}\n"
                     f"Median: {np.median(nn_distances):.2f}")
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        return ax,
    
    # Create animation (use every 5th frame to speed up animation)
    frames = range(0, len(df), 1)
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    
    # Save animation
    anim.save(f"{output_dir}/{"nn_histogram_animation.mp4"}", writer='ffmpeg', dpi=100)
    plt.close()

def makeDistanceTrends(df, output_dir):

    # Bonus: Plot how the mean and median nearest neighbor distance changes over time
    plt.figure(figsize=(10, 6))
    mean_distances = df['nn_distances'].apply(np.mean)
    median_distances = df['nn_distances'].apply(np.median)

    plt.plot(df['time'], mean_distances, label='Mean NN Distance', color='blue')
    plt.plot(df['time'], median_distances, label='Median NN Distance', color='red')
    plt.xlabel('Time')
    plt.ylabel('Distance')
    plt.title('Mean and Median Nearest Neighbor Distance Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/nn_distance_trends.png", dpi=300)


def main(data_dir, output_dir):
    df = load_data(data_dir) 
    makeNnHistograms(df, output_dir)
    create_histogram_animation(df, output_dir)
    makeDistanceTrends(df, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot nearest neighbor distance histograms.")
    parser.add_argument("filename", type=str, help="Path to the nearest neighbor distance data file")   
    parser.add_argument("output_dir", type=str, help="Directory to save the output figures")
    args = parser.parse_args()
    
    main(args.filename, args.output_dir)