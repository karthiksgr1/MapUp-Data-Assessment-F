import datetime
import pandas as pd
import numpy as np
from datetime import time

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    unique_ids = set(df['id_start'].unique()) | set(df['id_end'].unique())
    
    # Convert the set to a list for DataFrame creation
    unique_ids_list = sorted(list(unique_ids))

    # Create an empty DataFrame for the distance matrix
    distance_matrix = pd.DataFrame(index=unique_ids_list, columns=unique_ids_list)

    # Fill in the distance matrix based on known routes
    for _, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[id_start, id_end] = distance
        distance_matrix.at[id_end, id_start] = distance  # Symmetric matrix

    # Set diagonal values to 0
    np.fill_diagonal(distance_matrix.values, 0)

    # Apply the Floyd-Warshall algorithm for all-pairs shortest path
    for k in unique_ids_list:
        for i in unique_ids_list:
            for j in unique_ids_list:
                if pd.notna(distance_matrix.at[i, k]) and pd.notna(distance_matrix.at[k, j]):
                    if pd.isna(distance_matrix.at[i, j]):
                        distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
                    else:
                        distance_matrix.at[i, j] = min(distance_matrix.at[i, j], distance_matrix.at[i, k] + distance_matrix.at[k, j])

    return distance_matrix

def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    unrolled_data = []

    # Iterate over all combinations of unique IDs
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:
                distance = df.at[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    reference_rows = df[df['id_start'] == reference_id]

    # Calculate the average distance for the reference value
    average_distance = reference_rows['distance'].mean()

    # Calculate the threshold range (within 10% of the average distance)
    lower_threshold = average_distance - (0.1 * average_distance)
    upper_threshold = average_distance + (0.1 * average_distance)

    # Filter rows where distances are within the threshold range
    within_threshold = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]

    # Get unique values from the 'id_start' column and sort them
    result_ids = sorted(within_threshold['id_start'].unique())

    return result_ids

def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Add new columns for toll rates based on vehicle types
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    weekday_time_ranges = [
        {'start_time': time(0, 0, 0), 'end_time': time(10, 0, 0), 'discount_factor': 0.8},
        {'start_time': time(10, 0, 0), 'end_time': time(18, 0, 0), 'discount_factor': 1.2},
        {'start_time': time(18, 0, 0), 'end_time': time(23, 59, 59), 'discount_factor': 0.8}
    ]

    # Define constant discount factor for weekends
    weekend_discount_factor = 0.7

    # Create empty columns for start_day, start_time, end_day, and end_time
    df['start_day'] = ''
    df['start_time'] = pd.to_datetime(df['distance']).dt.time
    df['end_day'] = ''
    df['end_time'] = pd.to_datetime(df['distance']).dt.time

    # Iterate over each row
    for idx, row in df.iterrows():
        # Extract time information
        start_time, end_time = row['start_time'], row['end_time']

        # Update the corresponding columns
        df.at[idx, 'start_time'] = start_time
        df.at[idx, 'end_time'] = end_time

        # Determine the day for start_time
        for time_range in weekday_time_ranges:
            if time_range['start_time'] <= start_time <= time_range['end_time']:
                df.at[idx, 'start_day'] = pd.to_datetime(row['distance']).strftime('%A')
                break
        else:
            df.at[idx, 'start_day'] = 'Saturday' if pd.to_datetime(row['distance']).weekday() == 5 else 'Sunday'

        # Determine the day for end_time
        for time_range in weekday_time_ranges:
            if time_range['start_time'] <= end_time <= time_range['end_time']:
                df.at[idx, 'end_day'] = pd.to_datetime(row['distance']).strftime('%A')
                break
        else:
            df.at[idx, 'end_day'] = 'Saturday' if pd.to_datetime(row['distance']).weekday() == 5 else 'Sunday'

        # Apply discount factors based on time ranges
        for time_range in weekday_time_ranges:
            if time_range['start_time'] <= start_time <= time_range['end_time']:
                df.at[idx, 'moto'] *= time_range['discount_factor']
                df.at[idx, 'car'] *= time_range['discount_factor']
                df.at[idx, 'rv'] *= time_range['discount_factor']
                df.at[idx, 'bus'] *= time_range['discount_factor']
                df.at[idx, 'truck'] *= time_range['discount_factor']

        # Apply constant discount factor for weekends
        if df.at[idx, 'start_day'] in ['Saturday', 'Sunday']:
            df.at[idx, 'moto'] *= weekend_discount_factor
            df.at[idx, 'car'] *= weekend_discount_factor
            df.at[idx, 'rv'] *= weekend_discount_factor
            df.at[idx, 'bus'] *= weekend_discount_factor
            df.at[idx, 'truck'] *= weekend_discount_factor

    return df
