import PySimpleGUI as sg
import cardata 
      
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.python.keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
   
   
layout = [[sg.Text('Gender:'),
           sg.Text(size=(15,1), key='-OUTPUT-')],
           [sg.Radio("Male", "RADIO", key='-MALE-'), sg.Radio("Female", "RADIO", key='-FEMALE-')],
          [sg.Text('Age:')],
          [sg.Input(key='-IN-')],
          [sg.Text('Annual Salary:')],
          [sg.Input(key='-IN-')],
          [sg.Text('Credit_Card_Debt:')],
          [sg.Input(key='-IN-')],
          [sg.Text('Net Worth Purchase Amount:')],
          [sg.Input(key='-IN-')],
          [sg.Button('Display'), sg.Button('Exit')]]
          
          
         

  
window = sg.Window('Introduction', layout)

while True:
    event, values = window.read()
    print(event, values)
      
    if event in  (None, 'Exit'):
        break
    gender = '\nGender: ' 
    if values['-FEMALE-']: 
        gender += 'Female'
    else: 
     if event == 'Display':
            input_test_sample = np.array([[0, 41.8,  62812.09, 11609.38, 238961.25]])
#input_test_sample2 = np.array([[1, 46.73, 61370.67, 9391.34, 462946.49]])
#Scale input test sample data
input_test_sample_scaled = scaler_in.transform(input_test_sample)
#Predict output
output_predict_sample_scaled = model.predict(input_test_sample_scaled)
#Print predicted output
print('Predicted Output (Scaled) =', output_predict_sample_scaled)
#Unscale output
output_predict_sample = scaler_out.inverse_transform(output_predict_sample_scaled)
print('Predicted Output / Purchase Amount ', output_predict_sample)
        # Update the "output" text element
        # to be the value of "input" element
        
        window['-OUTPUT-'].update(values['-IN-'])