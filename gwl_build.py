# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:17:08 2024

@author: rahul
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('well_data.csv')

# Ensure all data is numeric and handle missing values
# Replace commas with periods in numerical columns
data['GWL'] = data['GWL'].astype(str).str.replace(',', '.').astype(float)
data['Average temperature'] = data['Average temperature'].astype(str).str.replace(',', '.').astype(float)
data['Precipititation'] = data['Precipititation'].astype(str).str.replace(',', '.').astype(float)
data['Meteo station number'] = data['Meteo station number'].astype(str).astype(float)

# Handle missing values
data = data.dropna()

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['GWL', 'Average temperature', 'Precipititation', 'Meteo station number']])

# Create sequences
def create_sequences(data, seq_length, well_num_col, date_col):
    sequences = []
    for well_num in data[well_num_col].unique():
        well_data = data[data[well_num_col] == well_num]
        well_data = well_data.sort_values(by=date_col)
        for i in range(len(well_data) - seq_length):
            seq = well_data.iloc[i:i+seq_length].drop(columns=[well_num_col, date_col]).values
            label = well_data.iloc[i+seq_length]['GWL']
            sequences.append((seq, label))
    return sequences

seq_length = 4  # Use past 4 months to predict the next month
sequences = create_sequences(data, seq_length, 'Well number', 'Date')

# Split the data into features and labels
X, y = zip(*sequences)
X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, X.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Save the model
model.save('groundwater_lstm_model.keras')


# Load the data
# data_new = pd.read_csv('prediction_new.csv')

# # Ensure all data is numeric and handle missing values
# data_new['GWL'] = data_new['GWL'].astype(str).str.replace(',', '.').astype(float)
# data_new['Average temperature'] = data_new['Average temperature'].astype(str).str.replace(',', '.').astype(float)
# data_new['Precipititation'] = data_new['Precipititation'].astype(str).str.replace(',', '.').astype(float)
# data_new['Meteo station number'] = data_new['Meteo station number'].astype(str).astype(float)


# # Function to preprocess new data
# def preprocess_data(data):
#     # Replace commas with periods in numerical columns
#     data['GWL'] = data['GWL'].astype(str).str.replace(',', '.').astype(float)
#     data['Average temperature'] = data['Average temperature'].astype(str).str.replace(',', '.').astype(float)
#     data['Precipititation'] = data['Precipititation'].astype(str).str.replace(',', '.').astype(float)
#     data['Meteo station number'] = data['Meteo station number'].astype(str).astype(float)
    
#     return data

# # Function to create sequences
# def create_sequences(data, seq_length, well_num_col, date_col):
#     sequences = []
#     for well_num in data[well_num_col].unique():
#         well_data = data[data[well_num_col] == well_num]
#         well_data = well_data.sort_values(by=date_col)
#         for i in range(len(well_data)+1 - seq_length):
#             seq = well_data.iloc[i:i+seq_length].drop(columns=[well_num_col, date_col]).values
#             sequences.append(seq)
#     return np.array(sequences)

# # Load new data
# new_data = pd.read_csv('prediction_new.csv')

# # Preprocess the new data
# new_data = preprocess_data(new_data)

# # Create sequences
# seq_length = 4  # Use past 4 months to predict the next month
# X_new = create_sequences(new_data, seq_length, 'Well number', 'Date')

# predictions = model.predict(X_new)

# # Save predictions to a CSV file
# predictions_df = pd.DataFrame(predictions, columns=['Predicted_GWL'])

# [17.87,9.04,23.89]