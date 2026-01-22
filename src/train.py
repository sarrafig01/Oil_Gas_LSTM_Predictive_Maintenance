
from data_loader import load_cmapss
from preprocessing import add_rul, normalize
from sequence_generator import create_sequences
from lstm_model import build_lstm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

DATA_PATH = "data/train_FD001.txt"
WINDOW_SIZE = 30

df = load_cmapss(DATA_PATH)
sensor_cols = [c for c in df.columns if "sensor" in c]

df = add_rul(df)
df = normalize(df, sensor_cols)

X, y = create_sequences(df, sensor_cols, WINDOW_SIZE)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = build_lstm((X_train.shape[1], X_train.shape[2]))

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("lstm_model.h5", save_best_only=True)
]

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks
)

print("Training finished")
