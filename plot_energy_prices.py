#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
This script extracts, combines, processes, and plots wholesale energy
prices from the National Energy Market (NEM). It scrapes historical
pricing data contained in daily AEMO Dispatch Reports at:
https://nemweb.com.au/Reports/Archive/DispatchIS_Reports/

Each report is an outer ZIP file, containing nested inner ZIP files with
CSV data. Each CSV file contains tables of varying widths and headers,
stacked within the same spreadsheet. Since changing from 30-minute to
5-minute settlement periods on 1 October 2021, each daily report has 288
CSV files, representing each 5-minute period.

The script follows these steps:
    1. Generates a list of AEMO Dispatch Report URLs within a specified
       date range using the get_report_urls() function.
    2. For each URL, extracts data from the AEMO Dispatch Report using
       the extract_report_data() function.
        - Iterates through the outer ZIP file, inner ZIP files, and CSV
          files.
        - Filters data by the specified table name and extracts data
          within the specified column index range.
        - Combines all the extracted CSV data from the 288 files.
    3. Combines all the extracted data from each report using the
       combine_report_data() function.
        - Provides progress updates using print(). Given 336 reports,
          there are 336 x 288 = 96,768 CSV files to extract from, which
          may take 30 minutes or more.
    4. Formats the combined data using the format_data() function and
       checks the formatted data for cleanliness using the
       check_data_cleanliness() function.
        - Note: Data cleaning functions have not yet been implemented as
          unclean data has not yet been encountered.
    5. Processes the formatted data, resampling price data from 5-minute
       to 30-minute intervals using the process_data() function.
    6. Plots the processed data using the plot_data() function.
        - Calculates various percentiles for energy prices.
        - For each state, generates percentile plots by hour of the day,
          saving them as a PNG image in the current directory.

How to run:
    1. Ensure Python 3.6 or later is installed.
    2. Install dependencies: pip install -r requirements.txt
    3. Place the script in a directory.
    4. Run the script: python plot_energy_prices.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO, TextIOWrapper
from zipfile import ZipFile, BadZipFile
from csv import reader
import time

# Constants
BASE_URL = 'https://nemweb.com.au/Reports/Archive/DispatchIS_Reports/PUBLIC_DISPATCHIS_{}.zip'
DATE_FORMAT = '%Y%m%d'
START_DATE = '2022-12-01'
END_DATE = '2023-11-01'
TABLE_NAME_FILTER = 'PRICE'
SETTLEMENTDATE_INDEX = 4
RRP_INDEX = 9
INTERVAL_FREQUENCY = '5 min'
PRICE_FLOOR = -1000
PRICE_CEILING = 16600
OUTPUT_FILE = 'plots.png'

# Functions
def table_name(row, name_column=2):
    """
    Returns the table name in the name_column index, default is the
    third column.
    """
    return row[name_column]

def row_type(row, type_column=0):
    """
    Returns the AEMO CSV data format row type in the type_column index,
    default is the first column.
    """
    return {'C': 'comment', 'I': 'header', 'D': 'data'}[row[type_column]]

def get_report_urls(base_url, date_format, start_date, end_date):
    """
    Generates a list of AEMO Dispatch Report URLs within a given date
    range.

    Args:
        base_url (str): Template for report URLs, with a replacement
        field for the date.
        date_format (str): Format for the replacement field.
        start_date (str or datetime-like): Starting date for the date
        range.
        end_date (str or datetime-like): Ending date for the date range

    Returns:
        List[str]: List of the report URLs.
    """
    date_range = pd.date_range(start=start_date, end=end_date)
    return [base_url.format(date.strftime(date_format)) for date in date_range]

def extract_report_data(url, table_name_filter, start_column_index, end_column_index):
    """
    Extracts and combines all data from an AEMO Dispatch Report, within
    a given column index range and filtered by a given table name.

    Note:
        AEMO 'PRICE' tables do not have a consistent number of columns.
        At 12/07/2023 11:00, 10 new columns were added. If new columns
        do not appear within the column index range, we avoid slower use
        of DataFrame merges. Else if they do, this code will need to be
        modified.

    Args:
        url (str): URL of the ZIP file, containing nested ZIP files with
        CSV data.
        table_name_filter (str): Table name to filter by.
        start_column_index (int): Starting column index for extraction.
        end_column_index (int): Ending column index for extraction.

    Returns:
        pd.DataFrame: Pandas DataFrame containing the combined data.
    """
    
    # Initialise a list to store combined data
    combined_data = []
    header = None

    try:
        # Request the outer ZIP file
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if unsuccessful

        # Open the outer ZIP file as a BytesIO buffer
        with ZipFile(BytesIO(response.content), 'r') as outer_zipfile:
            # Loop through each inner ZIP file
            for inner_filename in outer_zipfile.namelist():
                # Open the inner ZIP file as a BytesIO buffer
                with ZipFile(BytesIO(outer_zipfile.read(inner_filename))) as inner_zipfile:
                    # Loop through each CSV file
                    for csv_filename in inner_zipfile.namelist():
                        # Open the CSV file
                        with inner_zipfile.open(csv_filename) as csv_file:            
                            # Decode the CSV data, to process row by row as text
                            csv_reader = reader(TextIOWrapper(csv_file, encoding='utf-8'))

                            # Combine the extracted data within the column index range and filtered by table name.
                            for row in csv_reader:
                                if table_name(row) == table_name_filter:
                                    # Keep the first header row found
                                    if header is None and row_type(row) == 'header':
                                        header = row[start_column_index: end_column_index + 1]

                                    elif row_type(row) == 'data':
                                        combined_data.append(row[start_column_index: end_column_index + 1])
    except requests.RequestException as e:
        print(f'An error occured while requesting data: {e}')
    except BadZipFile as e:
        print(f'An error occured with the ZIP file: {e}')
    
    # Return a DataFrame with all the combined data
    return pd.DataFrame(combined_data, columns=header)

def combine_report_data(report_urls, table_name_filter, start_column_index, end_column_index):
    """
    Combines all data from AEMO Dispatch Reports, from a list of their
    URLs. Prints progress updates of the number of URLs processed and
    the elapsed time.

    Parameters:
        report_urls (List[str]): List of report URLs to extract from.
        table_name_filter (str): Table name to filter by.
        start_column_index (int): Starting column index for extraction.
        end_column_index (int): Ending column index for extraction.

    Returns:
        pd.DataFrame: Pandas DataFrame containing combined data from all
        dispatch reports.
    """
    # Initialise a DataFrame to store combined data
    combined_data = pd.DataFrame()
    
    # Count the number of URLs
    total_urls = len(report_urls)
    start_time = time.time() # Record the start time for progress updates
    
    # Loop through each report URL
    for i, url in enumerate(report_urls, 1):        
        # Extract and combine the data from the report
        extracted_data = extract_report_data(url, table_name_filter, start_column_index, end_column_index)
        combined_data = pd.concat([combined_data, extracted_data])

        # Time and print progress update
        elapsed_time = time.time() - start_time
        print(f'Processed URL {i} of {total_urls}. Elapsed time: {elapsed_time:.2f} seconds')
        
    return combined_data

def format_data(combined_data):
    """Formats the combined data."""
    # Rename columns
    formatted_data = combined_data.rename(columns={'SETTLEMENTDATE': 'Timestamp', 'REGIONID': 'State',
                                                  'INTERVENTION': 'Intervention', 'RRP': 'Price'})

    # Convert the 'Timestamp' column to datetime type and reduce
    # timestamps by 5 minutes to the start of the interval
    formatted_data['Timestamp'] = pd.to_datetime(formatted_data['Timestamp']) - pd.DateOffset(minutes=5)

    # Remove the trailing '1' in state names
    formatted_data['State'] = formatted_data['State'].str.rstrip('1')

    # Convert the 'Intervention' and 'Price' columns to numeric type
    formatted_data['Intervention'] = pd.to_numeric(formatted_data['Intervention'])
    formatted_data['Price'] = pd.to_numeric(formatted_data['Price'])

    return formatted_data

def check_data_cleanliness(formatted_data):
    """Checks the formatted data for cleanliness."""
    cleanliness_results = []
    
    # Check for missing values
    missing_values_count = formatted_data[['Timestamp', 'State', 'Price']].isnull().sum().sum()
    if missing_values_count > 0:
        cleanliness_results.append(f'{missing_values_count} missing values found.')

    # Check 'State' column for correct labels.
    states = set(formatted_data['State'].unique())
    expected_states = set(['NSW', 'QLD', 'SA', 'TAS', 'VIC'])
    if states != expected_states:
        cleanliness_results.append('Incomplete or incorrect state labels found.')

    # Check 'Timestamp' column is complete and of consistent frequency
    # Exclude intervention case duplicates
    timestamps = formatted_data[formatted_data['Intervention'] == 0]['Timestamp']
    # Calculate number of intervals within date range
    days_count = ((pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days + 1)
    intervals_per_day = pd.to_timedelta('1 day') / pd.to_timedelta(INTERVAL_FREQUENCY)
    intervals_count = days_count * intervals_per_day
    # Generate expected timestamps within date range
    expected_timestamps = pd.date_range(start=START_DATE, periods=intervals_count, freq=INTERVAL_FREQUENCY)
    # Repeat each timestamp once per state
    states_count = len(states)
    expected_timestamps = expected_timestamps.repeat(states_count)
    # Check if timestamps are not equal to expected timestamps
    if not np.array_equal(timestamps, expected_timestamps):
        cleanliness_results.append('Incomplete or inconsistent timestamps found.')

    # Check 'Price' column for outliers outside of the current cap and
    # floor range
    outliers_count = ((formatted_data['Price'] < PRICE_FLOOR) | (formatted_data['Price'] > PRICE_CEILING)).sum()
    if outliers_count > 0:
        cleanliness_results.append(f'{outliers_count} outliers found.')
    
    if cleanliness_results:
        print('Errors occurred while checking data cleanliness.', ' '.join(cleanliness_results))
        return False
    else:
        print('Successfully checked data for cleanliness.')
        return True
        
def process_data(formatted_data):
    """Processes the formatted data."""
    # Filter out rows where an intervention has occured
    processed_data = formatted_data[formatted_data['Intervention'] == 0]

    # Select relevant columns
    processed_data = processed_data[['Timestamp', 'State', 'Price']]

    # Set 'Timestamp' as the index
    processed_data = processed_data.set_index('Timestamp')
    
    # Resample the data from 5-minute intervals to 30-minute intervals and
    # aggregate by the mean
        # Note: check_data_cleanliness() ensures the data has complete
        # and consistent 5-minute intervals. Therefore, the mean will be
        # equivalent to the time-weighted average. If intervals are of
        # inconsistent frequency, a new time-weighted average function
        # will need to be implemented here.
    processed_data = processed_data.groupby('State').resample('30 min').mean()
    
    return processed_data.reset_index()

def plot_data(processed_data, sharey=False):
    """Plots the processed data."""
    # Create an 'Hour' column for plotting
    processed_data['Hour'] = processed_data['Timestamp'].dt.hour + processed_data['Timestamp'].dt.minute / 60

    # Group by 'State' and 'Hour' to calculate various percentiles
    percentile_levels = [0.01, 0.1, 0.5, 0.9, 0.99]
    df_percentiles = processed_data.groupby(['State', 'Hour'])['Price'].quantile(percentile_levels).unstack(level=-1)
    df_percentiles.columns = ['1st percentile', '10th percentile', '50th percentile',
                              '90th percentile', '99th percentile']
    
    # Get the unique states for individual plots
    states = processed_data['State'].unique()

    # Create a colormap with 5 distinguishable colors
    colormap = plt.cm.get_cmap('tab10', 5)

    # Create subplots for each state with individual x-axis labels
        # Note: set sharey=True if you would like the subplots to have
        # the same y-axis range for clearer comparison of prices between
        # states
    fig, axes = plt.subplots(nrows=len(states), ncols=1, figsize=(16, 8 * len(states)), sharex=False, sharey=sharey)

    for i, state in enumerate(states):
        axis = axes[i]
        state_data = df_percentiles.loc[state]  # Extract data for the state

        # Plot each percentile with a color from the colormap
        for j, percentile in enumerate(df_percentiles.columns):
            axis.plot(state_data.index, state_data[percentile], label=percentile, color=colormap(j))

        axis.set_title(f'Percentile Wholesale Energy Prices for {state}')
        axis.set_xlabel('Hour of Day')
        axis.set_ylabel('Price ($/MWh)')
        axis.set_xlim(0, 23.5)
        axis.set_xticks(np.arange(0, 24, 0.5))
        axis.set_xticklabels([f'{int(hour)}:{int((hour-int(hour))*60):02d}' for hour in np.arange(0, 24, 0.5)], rotation=90)
        axis.legend()
        axis.grid(True)
    
    # Adjust the layout to prevent overlap
    fig.tight_layout()
    # Save the plots to current directory
    fig.savefig(OUTPUT_FILE, facecolor='white')
    print(f'Saved {OUTPUT_FILE} to current directory.')
    
    # df_percentiles.to_csv('percentiles.csv', index=False)

if __name__ == '__main__':
    report_urls = get_report_urls(BASE_URL, DATE_FORMAT, START_DATE, END_DATE)
    combined_data = combine_report_data(report_urls, TABLE_NAME_FILTER, SETTLEMENTDATE_INDEX, RRP_INDEX)
    formatted_data = format_data(combined_data)
    if check_data_cleanliness(formatted_data) is True:
        processed_data = process_data(formatted_data)
        plot_data(processed_data)
        
        # combined_data.to_csv('combined_data.csv', index=False)
        # processed_data.to_csv('processed_data.csv', index=False)
    else:
        raise Exception("An error occured: Data was incorrect for processing.")

