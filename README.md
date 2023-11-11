This script extracts, combines, processes, and plots wholesale energy prices from the National Energy Market (NEM). It scrapes historical pricing data contained in daily AEMO Dispatch Reports at: https://nemweb.com.au/Reports/Archive/DispatchIS_Reports/

Each report is an outer ZIP file, containing nested inner ZIP files with CSV data. Each CSV file contains tables of varying widths and headers, stacked within the same spreadsheet. Since changing from 30-minute to 5-minute settlement periods on 1 October 2021, each daily report has 288 CSV files, representing each 5-minute period.

The script follows these steps:  
1. Generates a list of AEMO Dispatch Report URLs within a specified date range using the get_report_urls() function.
2. For each URL, extracts data from the AEMO Dispatch Report using the extract_report_data() function.
	- Iterates through the outer ZIP file, inner ZIP files.
	- Iterates through the outer ZIP file, inner ZIP files, and CSV files.
	- Filters data by the specified table name and extracts data within the specified column index range.
	- Combines all the extracted CSV data from the 288 files.
3. Combines all the extracted data from each report using the combine_report_data() function.
	- Provides progress updates using print(). Given 336 reports, there are 336 x 288 = 96,768 CSV files to extract from, which may take 30 minutes or more.
4. Formats the combined data using the format_data() function and checks the formatted data for cleanliness using the check_data_cleanliness() function.
	- Note: Data cleaning functions have not yet been implemented as unclean data has not yet been encountered.
5. Processes the formatted data, resampling price data from 5-minute to 30-minute intervals using the process_data() function.
6. Plots the processed data using the plot_data() function.
	- Calculates various percentiles for energy prices.
	- For each state, generates percentile plots by hour of the day, saving them as a PNG image in the current directory.

How to run:
1. Ensure Python 3.6 or later is installed.
2. Install dependencies: pip install -r requirements.txt
3. Place the script in a directory.
4. Run the script: python plot_energy_prices.py
