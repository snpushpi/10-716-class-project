import os
from datetime import timedelta, datetime
import pandas as pd

# 反推相对时间
def convert_to_utc(time_str):
    if pd.isnull(time_str):
        return None  # 或返回 "Invalid date format" 根据需要

    # 去除多余的空格
    time_str = time_str.strip()

    # 如果字符串以 "UTC" 结尾，则去除它，并确保不做时区偏移
    if time_str.endswith("UTC"):
        time_str = time_str.replace("UTC", "").strip()
        offset = timedelta(hours=0)
    elif " EDT" in time_str:
        time_str = time_str.replace(" EDT", "").strip()
        offset = timedelta(hours=-4)
    elif " EST" in time_str:
        time_str = time_str.replace(" EST", "").strip()
        offset = timedelta(hours=-5)
    else:
        offset = timedelta(hours=0)

    # 定义可接受的日期时间格式，包括处理没有时区文本的情况
    formats = [
        '%Y-%m-%d %H:%M:%S',     # e.g., "2023-12-16 22:00:00"
        '%B %d, %Y — %I:%M %p',   # e.g., "September 12, 2023 — 06:15 pm"
        '%b %d, %Y %I:%M%p',      # e.g., "Nov 14, 2023 7:35AM"
        '%d-%b-%y',              # e.g., "6-Jan-22"
        '%Y-%m-%d',              # e.g., "2021-4-5"
        '%Y/%m/%d',              # e.g., "2021/4/5"
        '%b %d, %Y'              # e.g., "DEC 7, 2023"
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(time_str, fmt)
            # 如果格式只包含日期（例如 '%d-%b-%y'），不进行时区调整
            if fmt == '%d-%b-%y':
                offset = timedelta(hours=0)
            dt_utc = dt + offset
            return dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
        except ValueError:
            continue

    # 如果所有格式都不匹配，返回错误信息
    return "Invalid date format"

def date_inte(folder_path, saving_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    for csv_file in csv_files:
        print('Starting: ' + csv_file)
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path, on_bad_lines="warn")
        df.columns = df.columns.str.capitalize()
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
        print("Before conversion:")
        print(df["Date"])
        df['Date'] = df['Date'].apply(convert_to_utc)
        print("After conversion:")
        print(df["Date"])
        # 将 "Invalid date format" 替换为 NaT，并转换为 datetime 格式
        df['Date'] = df['Date'].replace("Invalid date format", pd.NaT)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        # 按照 Date 列降序排序
        df = df.sort_values(by='Date', ascending=False)
        print(df)
        df.to_csv(os.path.join(saving_path, csv_file), index=False)
        print('Done: ' + csv_file)

if __name__ == "__main__":
    news_folder_path = 'news_data_raw'
    news_saving_path = 'news_data_preprocessed'

    stock_folder_path = 'stock_price_data_raw'
    stock_saving_path = 'stock_price_data_preprocessed'

    date_inte(news_folder_path, news_saving_path)
    date_inte(stock_folder_path, stock_saving_path)