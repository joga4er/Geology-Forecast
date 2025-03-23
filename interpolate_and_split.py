import glob
import random
from random import Random

import pandas as pd
import hashlib
import numpy as np
import matplotlib.pyplot as plt

import warnings

NEGATIVE_PART = -299
LARGEST_CHUNK = 600
SMALLEST_CHUNK = 350
TOTAL_REALIZATIONS = 10

#The filterwarnings () method is used to set warning filters, which can control the output method and level of warning information.
warnings.filterwarnings('ignore')

def remove_chunk(array, length):
    """
    Splits a numpy array into a chunk of the given length and the rest.

    Parameters:
    array (numpy.ndarray): The input array to split.
    length (int): The size of the chunk to remove.

    Returns:
    tuple: A tuple containing:
        - chunk (numpy.ndarray): The removed chunk of the specified length.
        - rest (numpy.ndarray): The remaining part of the array.
    """
    if length > len(array):
        raise ValueError("Length exceeds the size of the input array.")

    print(f"Chunk len: {length}")

    # Split the array
    chunk = array[:length]
    rest = array[length:]

    return chunk, rest


def add_chunk_to_df(df, input_array, chunk_length, file_name, initial_length):
    position = initial_length - len(input_array)
    chunk, shortened_array = remove_chunk(input_array, chunk_length)

    # subtracting the -300th element for normalization
    chunk -= chunk[-(LARGEST_CHUNK+NEGATIVE_PART)]

    # Calculate padding with NaNs
    padding_length = LARGEST_CHUNK - chunk_length
    padded_array = [np.nan] * padding_length + list(chunk)

    # Create a row with 'geology_id' and the padded array
    # TODO compute geology_id
    # Generate an 8-digit hash
    full_id = f"{file_name}_{str(position)}"
    hash_object = hashlib.md5(full_id.encode('utf-8'))
    hash_hex_id = f"g_{hash_object.hexdigest()[:10]}"  # Take the first 10 characters of the hash

    row = [hash_hex_id] + padded_array

    # Append to the DataFrame
    df.loc[len(df)] = row

    return shortened_array



def process_folder(
        path_to_process, output_file_name=None, data_augmentation=False, DO_PLOT = True, my_rnd=None, random_state=0
        ):
    if my_rnd is None:
        my_rnd = random.Random()
    # Get a list of all CSV files in the current directory
    csv_files = glob.glob(f"{path_to_process}/*.csv")
    number_of_files = len(csv_files)
    number_of_original_values = 0
    number_of_interpolated_values = 0

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
        number_of_original_values += len(df)

        print(df.head())

        # Define the new grid for VS_APPROX_adjusted with a fixed step of 1
        new_vs_grid = np.arange(df['VS_APPROX_adjusted'].min(), df['VS_APPROX_adjusted'].max() + 1, step=1)
        number_of_original_values += len(df)

        # Interpolate HORIZON_Z_adjusted values to the new grid
        new_horizon_z = np.interp(new_vs_grid, df['VS_APPROX_adjusted'], df['HORIZON_Z_adjusted'])
        number_of_interpolated_values += len(new_horizon_z)

        # mirrow interpolated values and set origin to zero
        if data_augmentation:
            new_horizon_z_mirrowed = new_horizon_z[::-1] - new_horizon_z[-1]
            #! TODO include mirrowed data into training data chunks!

        # Plot results
        if DO_PLOT:
            # Plot the original data
            ax.plot(df['VS_APPROX_adjusted'], df['HORIZON_Z_adjusted'],
                        label=file_name)
            if data_augmentation:
                ax.plot(new_vs_grid, new_horizon_z_mirrowed,
                    label=file_name)

            # # Plot the interpolated data
            # plt.plot(new_vs_grid, new_horizon_z,
            #          label='Interpolated Data', zorder=2)

        remaining_array = new_horizon_z

        # take about half in large chunks
        array_len = len(remaining_array)
        initial_len = array_len
        total_large = array_len // LARGEST_CHUNK // 2
        for i in range(total_large):
            remaining_array = add_chunk_to_df(total_df, remaining_array, LARGEST_CHUNK, file_name, initial_len)

        # for the remaining part (as long as chung is longer than largest chunk times 2.5)
        # take smaller chunks by randomely selecting from smallest possible to largest possible length
        while len(remaining_array) >= LARGEST_CHUNK*2.5:
            chunk_len = my_rnd.randint(SMALLEST_CHUNK, LARGEST_CHUNK)
            remaining_array = add_chunk_to_df(total_df, remaining_array, chunk_len, file_name, initial_len)

        array_len = len(remaining_array)
        remaining_len = array_len // 3
        for i in range(2):
            remaining_array = add_chunk_to_df(total_df, remaining_array, remaining_len, file_name, initial_len)

        remaining_array = add_chunk_to_df(total_df, remaining_array, len(remaining_array), file_name, initial_len)
        # print(remaining_array)

        # now we filled in the row
    # plt.legend()
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

    print(f'Number of profiles:{number_of_files}')
    print(f'Number of data points:{number_of_original_values}')
    print(f'Number of interpolated data points:{number_of_interpolated_values}')


if __name__ == "__main__":
    my_rnd = random.Random(42)
    process_folder(
        path_to_process=r"geology-forecast-challenge-open\data\train_raw",
        output_file_name='train_augmented.csv', data_augmentation=True, my_rnd=my_rnd
        )


