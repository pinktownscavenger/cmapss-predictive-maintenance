import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, header=None, delimiter=' ')
    df = df.iloc[:, :-2]  # Remove empty columns
    df.columns = ['Unit', 'Cycles', 'OpSet1', 'OpSet2', 'OpSet3'] + \
                 [f'Sensor{i}' for i in range(1, 22)]

    df['EOL'] = df.groupby('Unit')['Cycles'].transform('max')  # End of Life for each unit
    df['RUL'] = df['EOL'] - df['Cycles']
    df = df.drop(columns=['EOL', 'Unit'])  # Drop unnecessary columns

    return df

def visualize_correlations(df):
    corrmat = df.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(corrmat, cmap="RdYlGn", linewidths=0.1, annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.show()

def feature_selection(df, threshold=0.5):
    del_cols = [col for col in df.columns if abs(df[col].corr(df['RUL'])) < threshold]
    return df.drop(columns=del_cols)

def prepare_data(df, seq_length=10):
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]  # Target (RUL)

    # Scale data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences for GRU
    X_reshaped, y_seq = [], []
    for i in range(len(X_scaled) - seq_length + 1):
        X_reshaped.append(X_scaled[i:i + seq_length, :])
        y_seq.append(y.values[i + seq_length - 1])

    return np.array(X_reshaped), np.array(y_seq), scaler

# Build the GRU model
def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(GRU(units=100, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(units=100))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))  # Output layer for RUL

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

if __name__ == "__main__":
    data_path = "CMAPSSDATA/train_FD001.txt"
    df = load_and_preprocess_data(data_path)

    visualize_correlations(df)

    df_selected = feature_selection(df, threshold=0.5)

    seq_length = 10
    X, y, scaler = prepare_data(df_selected, seq_length=seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = build_gru_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error on Test Set: {mse:.4f}")
