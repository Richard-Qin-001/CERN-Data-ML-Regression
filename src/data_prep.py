import pandas as pd
import numpy as np
import config
import sqlite3

def load_csv():
    try:
        df = pd.read_csv(config.DATA_PATH)
        print('Data loaded successful')
    except FileExistsError as e:
        print(f"Error: {e}")
    df.head()
    df.info()
    df_cleaned = df.dropna(subset=['M']).copy()
    df_cleaned['Q_prod'] = df_cleaned['Q1'] * df_cleaned['Q2']
    df_cleaned['dR'] = np.sqrt(
        (df_cleaned['eta1'] - df_cleaned['eta2'])**2 + 
        (df_cleaned['phi1'] - df_cleaned['phi2'])**2
    )
    print(df_cleaned.shape)
    return df_cleaned

def df2db(df):
    conn = sqlite3.connect(config.DATABASE_PATH)
    df.to_sql('electron_events', conn, if_exists='replace', index=False)
    conn.close()
    print("The data has been successfully stored in the 'electron_events' table in cern_data.db.")

def load_data_from_db():
    conn = sqlite3.connect(config.DATABASE_PATH)
    try:
        df_loaded = pd.read_sql_query("SELECT * FROM electron_events", conn)
        print(f"Data has been successfully loaded from the database. Total rows: {df_loaded.shape[0]}")
    except Exception as e:
        print(f"Error: Unable to load data from the database. Please check the table name and database file. Error message: {e}")
        exit(0)
    conn.close()
    return df_loaded

if __name__ == "__main__":
    df = load_csv()
    df2db(df)
