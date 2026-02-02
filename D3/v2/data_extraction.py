


file_path = '/Users/gapaza/Documents/projects/data-driver/sim-to-real/DDERCompression1218_initial/DDERCompression1218Data.csv'

import pandas as pd


def get_csv_column_as_list(file_path, column_name):
    # 1. Load the CSV file
    # CSVs don't have "sheets," so we remove the sheet_name parameter
    df = pd.read_csv(file_path)

    # Print column names


    # 2. Extract the column and convert to list
    # .dropna() is added here optionally to remove empty rows
    column_list = df[column_name].dropna().tolist()



    return column_list


# --- Example Usage ---
file = file_path  # Path to your file
sheet = 'DDERCompression1218Data'  # Name of the tab
column = 'Force_N'  # The exact header name of the column

try:
    my_list = get_csv_column_as_list(file, column)
    print(my_list)
except KeyError:
    print(f"Error: The column '{column}' was not found in the file.")
except Exception as e:
    print(f"An error occurred: {e}")



