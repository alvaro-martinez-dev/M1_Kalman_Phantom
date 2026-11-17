# scripts/plot_bragg.py
"""
Bragg Peak Validation Plot.
This script reads the output of the Monte Carlo propagation (states.parquet) and plots the mean kinetic energy of the protons as a function of depth (z) to validate the Bethe-Bloch energy loss implementation.
Usage: python -m scripts.plot_bragg
"""
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Find the latest propagation dataset by searching the states.parquet files and picking the most recently modified one.
    # We do this because the user might have run the simulation multiple times with different parameters or seeds.
    list_of_files = glob.glob('data/propagation/*/states.parquet')
    if not list_of_files:
        print("No propagation dataset found in data/propagation/. Please run 'python -m scripts.generate_propagation' first.")
        return
    
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading data from: {latest_file}")

    # Load the parquet file into a pandas DataFrame. The 'T' column is the kinetic energy in MeV and 'z' is the depth in mm.
    # We use parquet because it is a highly efficient binary format for large datasets, much faster than CSV.
    df = pd.read_parquet(latest_file)

    # Group by the depth 'z' and calculate the mean kinetic energy 'T' of all protons at that position.
    # This statistical averaging removes the random scattering noise of individual tracks and reveals the smooth Bragg curve.
    energy_profile = df.groupby('z')['T'].mean().reset_index()

    # Create the plot with a size of 10x6 inches.
    plt.figure(figsize=(10, 6))
    
    # Plot the mean energy. The curve should be flat at the entrance, drop slightly in the water, and drop sharply at the end (Bragg Peak) where protons deposit the rest of their energy.
    plt.plot(energy_profile['z'], energy_profile['T'], color='blue', linewidth=2, label='Mean Proton Energy')

    # Shade the phantom layers based on the geometry defined in config.yaml (Water 50mm, Bone 20mm, Water 50mm).
    # The z_start of the phantom is 50 mm, so the layers are: 50-100 (Water), 100-120 (Bone), 120-170 (Water).
    # We use axvspan to draw vertical bands and highlight where the bone is, since the energy loss should be steeper there due to higher density.
    plt.axvspan(50, 100, color='cyan', alpha=0.2, label='Water (50 mm)')
    plt.axvspan(100, 120, color='gray', alpha=0.4, label='Bone (20 mm)')
    plt.axvspan(120, 170, color='cyan', alpha=0.2)

    # Set labels and title. Use scientific naming for clarity.
    plt.title('Bragg Peak: Proton Energy Loss in Water-Bone-Water Phantom')
    plt.xlabel('Depth z [mm]')
    plt.ylabel('Mean Kinetic Energy [MeV]')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()

    # Save the figure in the same output folder as the dataset so it is kept next to the data it came from, and also show it interactively.
    output_dir = os.path.dirname(latest_file)
    save_path = os.path.join(output_dir, 'bragg_peak_validation.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()