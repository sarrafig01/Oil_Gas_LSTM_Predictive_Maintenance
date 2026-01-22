
import numpy as np

def create_sequences(df, sensors, window_size=30):
    X, y = [], []
    for engine_id in df["engine_id"].unique():
        engine_df = df[df["engine_id"] == engine_id]
        values = engine_df[sensors].values
        rul = engine_df["RUL"].values

        for i in range(len(values) - window_size):
            X.append(values[i:i + window_size])
            y.append(rul[i + window_size])

    return np.array(X), np.array(y)
