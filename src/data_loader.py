import pandas as pd

def load_data(path):
    return pd.read_csv(path, delim_whitespace=True)
