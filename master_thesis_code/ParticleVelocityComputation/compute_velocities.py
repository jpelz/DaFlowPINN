import os
import numpy as np
import pandas as pd
import glob

def compute_velocites(dt, file_pattern, output_file):
    def read_particle_file(filename):
        """Reads a single particle data file and returns a DataFrame."""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Extract time step from filename
        timestep = int(filename.split("_")[-1].split(".")[0])
        time = timestep * dt
        
        # Extract data (skip header lines)
        data = np.loadtxt(lines[3:])  # Skip 3 header lines
        df = pd.DataFrame(data, columns=["X", "Y", "Z", "ID"])

        # Convert positions from mm to m
        df[["X", "Y", "Z"]] /= 1000.0

        df["T"] = time
        return df

    # Get sorted list of files
    data_files = sorted(glob.glob(file_pattern))
    all_data = {}

    # Read all files into a dictionary {timestep: dataframe}
    for file in data_files:
        df = read_particle_file(file)
        timestep = int(file.split("_")[-1].split(".")[0])
        all_data[timestep] = df.set_index("ID")  # Use ID as index for easy matching

    # Compute velocities
    data_with_velocity = []
    time_steps = sorted(all_data.keys())

    for i, t in enumerate(time_steps):
        df = all_data[t]
        #df["U"], df["V"], df["W"] = np.nan, np.nan, np.nan  # Initialize velocity columns
        
        if i > 0 and i < len(time_steps) - 1:
            # Central difference for most cases
            df_prev = all_data[time_steps[i - 1]]
            df_next = all_data[time_steps[i + 1]]
            common_ids = df.index.intersection(df_prev.index).intersection(df_next.index)
            df = df.loc[common_ids].copy()  # Keep only common IDs
            df.loc[common_ids, "U"] = (df_next.loc[common_ids, "X"] - df_prev.loc[common_ids, "X"]) / (2 * dt)
            df.loc[common_ids, "V"] = (df_next.loc[common_ids, "Y"] - df_prev.loc[common_ids, "Y"]) / (2 * dt)
            df.loc[common_ids, "W"] = (df_next.loc[common_ids, "Z"] - df_prev.loc[common_ids, "Z"]) / (2 * dt)
        
        if i == 0:
            # Forward difference for first time step
            df_next = all_data[time_steps[i + 1]]
            common_ids = df.index.intersection(df_next.index)
            df = df.loc[common_ids].copy()  # Keep only common IDs
            df.loc[common_ids, "U"] = (df_next.loc[common_ids, "X"] - df.loc[common_ids, "X"]) / dt
            df.loc[common_ids, "V"] = (df_next.loc[common_ids, "Y"] - df.loc[common_ids, "Y"]) / dt
            df.loc[common_ids, "W"] = (df_next.loc[common_ids, "Z"] - df.loc[common_ids, "Z"]) / dt
        
        if i == len(time_steps) - 1:
            # Backward difference for last time step
            df_prev = all_data[time_steps[i - 1]]
            common_ids = df.index.intersection(df_prev.index)
            df = df.loc[common_ids].copy()  # Keep only common IDs
            df.loc[common_ids, "U"] = (df.loc[common_ids, "X"] - df_prev.loc[common_ids, "X"]) / dt
            df.loc[common_ids, "V"] = (df.loc[common_ids, "Y"] - df_prev.loc[common_ids, "Y"]) / dt
            df.loc[common_ids, "W"] = (df.loc[common_ids, "Z"] - df_prev.loc[common_ids, "Z"]) / dt
        
        data_with_velocity.append(df.reset_index())

    # Combine all time steps and save
    final_df = pd.concat(data_with_velocity, ignore_index=True)
    final_df = final_df[["ID", "X", "Y", "Z", "T", "U", "V", "W"]]  # Ensure correct column order
    final_df.to_csv(output_file, sep=' ', index=False, float_format='%.6e')

    print(f"Velocity data saved to {output_file}")

if __name__=="__main__":
    # Constants
    dt = 5.916e-3  # Time step in seconds
    ppp=["001", "010", "050", "200"]

    for i in ppp:
        file_pattern = f"/scratch/jpelz/da-challenge/DA_CASE01/DA_CASE01_TR/DA_CASE01_TR_ppp_0_{i}/DA_CASE01_TR_ppp_0_{i}_PartFile_*.dat"
        output_file = f"/scratch/jpelz/da-challenge/DA_CASE01/velocity_files/DA_CASE01_TR_ppp_0_{i}_velocities.dat"
        compute_velocites(dt, file_pattern, output_file) 