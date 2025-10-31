import os

CURRENT_DIR =  os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
DATA_PATH = os.path.join(DATA_DIR, 'dielectron.csv')
DATABASE_DIR = os.path.join(PROJECT_DIR, 'database')
DATABASE_PATH = os.path.join(DATABASE_DIR, 'cern_data.db')