import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class StockPredictor:
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.model = self.build_model()

    def preprocess(self):
        scaled_data = self.scaler.fit_transform(self.data)
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

    def build_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train(self, X, y, epochs=1, batch_size=1):
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs)

if __name__ == "__main__":
    print("LSTM Stock Predictor Initialized.")
    print("Ready to load market data and begin training.")
