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
        # Update the "output" text element
        # to be the value of "input" element
        window['-OUTPUT-'].update(values['-IN-'])