# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:10:33 2024

@author: rahul
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('groundwater_lstm_model.keras')

# Function to preprocess new data
def preprocess_data(data):
    # Replace commas with periods in numerical columns
    data['GWL'] = data['GWL'].astype(str).str.replace(',', '.').astype(float)
    data['Average temperature'] = data['Average temperature'].astype(str).str.replace(',', '.').astype(float)
    data['Precipititation'] = data['Precipititation'].astype(str).str.replace(',', '.').astype(float)
    data['Meteo station number'] = data['Meteo station number'].astype(str).astype(float)
    return data
    
    

# Function to create sequences
def create_sequences(data, seq_length, well_num_col, date_col):
    sequences = []
    for well_num in data[well_num_col].unique():
        well_data = data[data[well_num_col] == well_num]
        well_data = well_data.sort_values(by=date_col)
        for i in range(len(well_data)+1 - seq_length):
            seq = well_data.iloc[i:i+seq_length].drop(columns=[well_num_col, date_col]).values
            sequences.append(seq)
    return np.array(sequences)

# Load new data
new_data = pd.read_csv('prediction_new.csv')

# Preprocess the new data
new_data = preprocess_data(new_data)

seq_length = 4  # Use past 4 months to predict the next month
X_new = create_sequences(new_data, seq_length, 'Well number', 'Date')

predictions = model.predict(X_new)

# Save predictions to a CSV file
predictions_df = pd.DataFrame(predictions, columns=['Predicted_GWL'])
print(predictions_df)
predictions_df.to_csv("predictions_gwl.csv")
print('Predictions saved to predicted_groundwater_levels.csv')


