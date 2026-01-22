
import pandas as pd

def load_cmapss(path):
    columns = (
        ["engine_id", "cycle"] +
        [f"op_setting_{i}" for i in range(1, 4)] +
        [f"sensor_{i}" for i in range(1, 22)]
    )
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = columns
    return df
