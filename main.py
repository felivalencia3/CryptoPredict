import pandas as pd
import numpy
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/Weekly-DOGE-USD.csv")


def plot_prices():
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df['Date']
    plt.figure(figsize=(16, 8))
    plt.plot(df["Close"], label='Close Price history')
    plt.show()


if __name__ == '__main__':
    # Move data into Panda's DataFram
    data = df.sort_index(ascending=True, axis=0)
    new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
    for i in range(0, len(data)):
        new_dataset["Date"][i] = data['Date'][i]
        new_dataset["Close"][i] = data["Close"][i]

    # Normalize Dataset, "Squish" it.
    final_dataset = new_dataset.values

    train_data = final_dataset[0:93, :]
    valid_data = final_dataset[93:, :]

    new_dataset.index = new_dataset.Date
    new_dataset.drop("Date", axis=1, inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data, y_train_data = [], []

    for i in range(116, len(train_data)):
        x_train_data.append(scaled_data[i - 116:i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = numpy.array(x_train_data), numpy.array(y_train_data)

    x_train_data = numpy.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))
