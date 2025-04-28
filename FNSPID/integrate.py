import os
import pandas as pd

def convert_to_utc(df, date_column):
    """
    Convert the DataFrame's date column to UTC.
    """
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    if df[date_column].dt.tz is None:
        df[date_column] = df[date_column].dt.tz_localize('UTC')
    return df

def custom_agg(group):
    # Average for numeric columns
    avg_values = group.select_dtypes(include='number').mean()
    
    # Concatenate 'New_text' with a separator
    concatenated_text = " New Article ".join(group['New_text'])
    
    # Combine results using pd.concat
    result = pd.concat([avg_values, pd.Series({'New_text': concatenated_text})])
    
    return result

def integrate_files(file_path_folder1, file_path_folder2):
    """
    Integrate two CSV files (one from folder1 and one from folder2) based on the 'Date' column.
    """
    # Read CSV files
    df1 = pd.read_csv(file_path_folder1)
    df2 = pd.read_csv(file_path_folder2)
    
    # Standardize column names (capitalize each column name)
    df1.columns = df1.columns.str.capitalize()
    df2.columns = df2.columns.str.capitalize()
    
    # Convert Date columns to UTC and ensure proper datetime format
    df1 = convert_to_utc(df1, 'Date')
    df2 = convert_to_utc(df2, 'Date')
    
    df1['Date'] = pd.to_datetime(df1['Date'], errors='coerce')
    df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
    
    # Normalize the Date columns to the start of the day (midnight)
    df1['Date'] = df1['Date'].dt.normalize()
    df2['Date'] = df2['Date'].dt.normalize()
    
    # Set Date as the index and sort the DataFrames
    df1.set_index('Date', inplace=True)
    df2.set_index('Date', inplace=True)
    
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)
    
    # Merge on Date (inner join ensures only dates present in both files are included)
    merged_df = pd.merge(df1, df2, on='Date', how='inner')
    
    # Filter rows where "New_text" column is not empty
    merged_df = merged_df[merged_df['New_text'].notna() & (merged_df['New_text'] != '')]
    merged_df = merged_df.drop(columns=['Url'])
    result_df = merged_df.groupby(merged_df.index).apply(custom_agg)
    return result_df

def integrate_folders(folder1, folder2, output_folder):
    """
    For every file in folder1, find the corresponding file in folder2 (with the same name),
    merge them based on the Date column, and save the merged file to output_folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process only CSV files
    files = [f for f in os.listdir(folder1) if f.endswith('.csv')]
    
    for filename in files:
        file1 = os.path.join(folder1, filename)
        file2 = os.path.join(folder2, filename)
        
        if not os.path.isfile(file2):
            print(f"Skipping {filename}: corresponding file not found in {folder2}.")
            continue
        
        print(f"Processing {filename}...")
        merged_df = integrate_files(file1, file2)
        
        # Reset index to include Date as a column in the final dataset
        #merged_df.reset_index(inplace=True)
        
        # Save the merged DataFrame to the output folder with the same filename
        output_file = os.path.join(output_folder, filename)
        merged_df.to_csv(output_file)
        print(f"Saved merged file: {output_file}")

if __name__ == "__main__":
    # Folder paths
    folder1 = "news_data_summarized"   # Files with Date, Url, New_text
    folder2 = "data_authors"   # Files with many columns (including Date)
    output_folder = "stock_news_author_integrated"  # Destination for the merged files
    
    integrate_folders(folder1, folder2, output_folder)