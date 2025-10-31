import pandas as pd
import numpy as np
import config
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

def data_split(df):
    features_to_exclude = ['Run', 'Event', 'M']
    X = df.drop(features_to_exclude, axis=1)
    y = df['M']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"\nTraining set X_train shape: {X_train.shape}")
    print(f"\nTesting set X_test shape: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    print("Data segmentation and normalization completed.")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    df = load_csv()
    df2db(df)
