import os
import time
import pandas as pd

def from_csv_copy_summaries(folder_path, saving_path):
    # List all CSV files in the input folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    start_time = time.time()
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        file_path = os.path.join(folder_path, csv_file)
        
        # Try to read the file with utf-8 encoding first, then fallback if needed
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="Windows-1252")
        
        # Extract stock symbol from filename and ensure uppercase
        symbol = csv_file.split(".")[0].upper()
        
        # Capitalize column names for consistency (e.g., 'date' -> 'Date', 'url' -> 'Url')
        df.columns = df.columns.str.capitalize()
        
        # Copy the summary column 'Lsa_summary' into a new column 'New_text'
        df['New_text'] = df['Lsa_summary']
        
        # If a column 'Mark' exists, keep only rows where Mark equals 1
        if 'Mark' in df.columns:
            df = df[df['Mark'] == 1]
        
        # Keep only the required columns: Date, Url, and New_text
        df = df[['Date', 'Url', 'New_text']]
        
        # Save the processed DataFrame to the saving folder
        output_file = os.path.join(saving_path, f"{symbol}.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {output_file} in {time.time() - start_time:.2f} s")
    
if __name__ == "__main__":
    headline_path = "news_data_preprocessed"
    headline_saving_path = "news_data_summarized"
    os.makedirs(headline_saving_path, exist_ok=True)
    from_csv_copy_summaries(headline_path, headline_saving_path)
