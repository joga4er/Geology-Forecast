import os
import glob
import random
from typing import Optional

import pandas as pd
import hashlib
import numpy as np
import matplotlib.pyplot as plt

import warnings

NEGATIVE_PART = -299
LARGEST_CHUNK = 600
SMALLEST_CHUNK = 350
TOTAL_REALIZATIONS = 1

#The filterwarnings () method is used to set warning filters, which can control the output method and level of warning information.
warnings.filterwarnings('ignore')

def get_sequence_length_distribution(test_data_filename):
    print(os.getcwd())
    test_data = pd.read_csv(test_data_filename)
    sequence_length = 300 - test_data.isna().sum(axis=1)
    sequence_length.plot.hist(bins=300)
    full_length_ratio = len(sequence_length[sequence_length==300])/len(sequence_length)
    print(f'The probability of full length input sequence in test data is {100 * round(full_length_ratio, 2)} %.')
    plt.xlabel('Sequence length of test data')
    plt.show()
    return round(full_length_ratio, 2)

def create_example_plot(input_profile_filename):
    input_profile = pd.read_csv(input_profile_filename)
    new_vs_grid = np.arange(
        input_profile['VS_APPROX_adjusted'].min(), input_profile['VS_APPROX_adjusted'].max() + 1, step=1
        )
    new_horizon_z = np.interp(
        new_vs_grid, input_profile['VS_APPROX_adjusted'], input_profile['HORIZON_Z_adjusted']
        )
    fig, ax = plt.subplots(2)

    ax[0].plot(new_vs_grid, new_horizon_z, color='b')
    ax[0].set_ylabel('vertical distance [feet]')
    ax[0].plot([1200, 1200, 1500, 1500, 1200], [20, 100, 100, 20, 20], 'k')
    ax[0].plot([1500, 1800, 1800, 1500], [100, 100, 20, 20], color='k', linestyle='--')
    ax[1].plot([1200, 1200, 1500, 1500, 1200], [20, 100, 100, 20, 20], 'k')
    ax[1].plot([1500, 1800, 1800, 1500], [100, 100, 20, 20], color='k', linestyle='--')
    ax[1].plot(new_vs_grid[1200:1500], new_horizon_z[1200:1500], color='b', )
    ax[1].plot(new_vs_grid[1500:1800], new_horizon_z[1500:1800], color='b', linestyle='--')
    for distorted in [-0.02, -0.015, -0.01, -0.05, 0.02, 0.08, 0.012, 0.018]:
        horizon_z_prob = new_horizon_z[1500:1800].copy()
        for y in range(300):
            horizon_z_prob[y] += y * distorted
        ax[1].plot(new_vs_grid[1500:1800], horizon_z_prob, color='b', linestyle='--')
         
    ax[1].set_ylabel('vertical distance [feet]')
    ax[1].set_xlabel('horizontal distance [feet]')
    plt.show()

def add_training_window(
        df: pd.DataFrame, input_array: list, start_index: int, sequence_length: int,
        file_name: str,
        ):

    # Calculate padding with NaNs
    padding_length = LARGEST_CHUNK - sequence_length
    padded_array = [np.nan] * padding_length + list(input_array[start_index:(sequence_length + start_index)])
    # offset profile so that 0th position is zero
    center_value = padded_array[NEGATIVE_PART*(-1)]
    padded_array = [elem - center_value for elem in padded_array]
    # Create a row with 'geology_id' and the padded array
    # TODO compute geology_id
    # Generate an 8-digit hash
    full_id = f"{file_name}_{str(start_index)}"
    hash_object = hashlib.md5(full_id.encode('utf-8'))
    hash_hex_id = f"g_{hash_object.hexdigest()[:10]}"  # Take the first 10 characters of the hash

    row = [hash_hex_id] + padded_array

    # Append to the DataFrame
    df.loc[len(df)] = row

def process_folder(
        path_to_process: str, output_file_name: str=None, data_augmentation: bool=False,
        sequence_stride: int=10, DO_PLOT: bool=False,
        my_rnd: Optional[random.Random]=None, random_state: int=0,
        full_length_ratio: float=0.5,
        ):
    if my_rnd is None:
        my_rnd = random.Random(random_state)
    random.seed(random_state)
    # Get a list of all CSV files in the current directory
    csv_files = glob.glob(f"{path_to_process}/*.csv")
    data_overview = pd.DataFrame({'data points':[], 'length in feet':[]})

    # Create column names with the first column as 'geology_id' and the rest as numbers
    # columns = ['geology_id'] + [NEGATIVE_PART + i for i in range(abs(NEGATIVE_PART)+1)]  # Columns: 'geology_id', -299 to 0
    columns = ['geology_id'] + [NEGATIVE_PART + i for i in range(LARGEST_CHUNK)]  # Columns: 'geology_id', -299 to 300
    # for k in range(1,TOTAL_REALIZATIONS):
    #     columns += [f"r_{k}_pos_{i}" for i in np.arange(1, LARGEST_CHUNK + NEGATIVE_PART)]
    #     # print([f"r_{k}_pos_{i}" for i in np.arange(1, LARGEST_CHUNK + NEGATIVE_PART)])

    # Create an empty DataFrame with these column names
    total_df = pd.DataFrame(columns=columns)
    if DO_PLOT:
        _, ax = plt.subplots()
    # Read
    for file_path in csv_files:
        # Extract the file name (excluding directories)
        file_name = file_path.split('/')[-1]
        print(f"Processing {file_path}; File name: {file_name}")
        df = pd.read_csv(file_path)

        # Define the new grid for VS_APPROX_adjusted with a fixed step of 1
        new_vs_grid = np.arange(df['VS_APPROX_adjusted'].min(), df['VS_APPROX_adjusted'].max() + 1, step=1)
        data_overview.loc[file_name] = [len(df), len(new_vs_grid), ]  # save sequence length of profile to overview

        # Interpolate HORIZON_Z_adjusted values to the new grid
        new_horizon_z = np.interp(new_vs_grid, df['VS_APPROX_adjusted'], df['HORIZON_Z_adjusted'])

        # mirrow interpolated values and set origin to zero
        if data_augmentation:
            new_horizon_z_mirrowed = new_horizon_z[::-1] - new_horizon_z[-1]

        # Plot results
        if DO_PLOT:
            # Plot the original data
            ax.plot(df['VS_APPROX_adjusted'], df['HORIZON_Z_adjusted'],
                        label=file_name, color='k')
            if data_augmentation:
                ax.plot(new_vs_grid, new_horizon_z_mirrowed,
                    label=file_name, linestyle=':', color='k')
        # iterate over profile and mirrowed profile
        for profile in [new_horizon_z, new_horizon_z_mirrowed]:
            # iterate over start of training window
            for start_index in np.arange(0, len(profile) - LARGEST_CHUNK, sequence_stride):
                # initialize sequence length
                sequence_length = LARGEST_CHUNK
                # decide if input length is maximum length based on random
                if random.random() > full_length_ratio:
                    # get sequence length from random integer
                    sequence_length = random.randint(SMALLEST_CHUNK, LARGEST_CHUNK)
                add_training_window(
                    df=total_df, input_array=profile, start_index=start_index,
                    sequence_length=sequence_length, file_name=file_name,
                )

    if DO_PLOT:
        ax.set_xlabel('horizontal distance from origin [feet]')
        ax.set_ylabel('vertical distance from origin [feet]')
        plt.show()

    for k in range(1, TOTAL_REALIZATIONS):
        for i in range(1, LARGEST_CHUNK + NEGATIVE_PART):
            total_df[f"r_{k}_pos_{i}"] = total_df[i]

    # Reshuffle rows while keeping the original index
    reshuffled_df = total_df.sample(frac=1, random_state=random_state)
    # Save the reshuffled DataFrame as a CSV file in UTF-8 encoding
    reshuffled_df.to_csv(output_file_name, encoding='utf-8', index=False)
    print(f'There are {len(data_overview)} geological profiles available for traiing and validation')

    if DO_PLOT:
        fig, ax = plt.subplots(2)

        data_overview['data points'].plot.hist(ax=ax[0], bins=40)
        data_overview['length in feet'].plot.hist(ax=ax[1], bins=40)
        ax[0].legend()
        ax[1].legend()
        plt.show()

    print(data_overview)


if __name__ == "__main__":
    my_rnd = random.Random(42)
    full_length_ratio = get_sequence_length_distribution(os.path.join("data","test.csv"))
    create_example_plot(os.path.join("data","train_raw", "3c9b6dea.csv"))
    process_folder(
        path_to_process=r"data\train_raw",
        output_file_name=r"data\train_augmented.csv",
        data_augmentation=True, 
        sequence_stride=100,
        my_rnd=my_rnd, random_state=42,
        full_length_ratio=full_length_ratio,
        )


